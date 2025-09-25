import torch
import wandb
import os

from torch.nn.functional import scaled_dot_product_attention
from typing import Any, Iterable, List, Tuple, Dict
from torch import nn
# from transformers.models.qwen2.modeling_qwen2 import *
from transformers.models.qwen3.modeling_qwen3 import *
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Optional, Cache, Unpack, FlashAttentionKwargs, BaseModelOutputWithPast,  # noqa
    DynamicCache, Qwen3Attention, apply_rotary_pos_emb, Qwen3MLP  # noqa
)
from transformers.loss.loss_utils import ForCausalLMLoss

from modeling.shared_func_module import get_flash_attention_args_tensorized, UniQwen3Config, get_vision_mask
from modeling import shared_func_module
from accelerate import Accelerator
from tools.log import main_logger

accelerator = Accelerator()


class UniShareQwen3MLP(Qwen3MLP):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.config = config
        self.has_share_expert = (config.ffn_share_size != 0)
        self.has_vis_expert = (config.ffn_vision_size != 0)

        assert self.has_share_expert or self.has_vis_expert

        if self.has_share_expert:
            self.share_gate_proj = nn.Linear(self.hidden_size, config.ffn_share_size, bias=False)
            self.share_up_proj = nn.Linear(self.hidden_size, config.ffn_share_size, bias=False)
            self.share_down_proj = nn.Linear(config.ffn_share_size, self.hidden_size, bias=False)
        if self.has_vis_expert:
            self.vision_gate_proj = nn.Linear(self.hidden_size, config.ffn_vision_size, bias=False)
            self.vision_up_proj = nn.Linear(self.hidden_size, config.ffn_vision_size, bias=False)
            self.vision_down_proj = nn.Linear(config.ffn_vision_size, self.hidden_size, bias=False)

        self.buffer_vis_mask = None

    def set_buffer_vis_mask(self, vis_mask):
        self.buffer_vis_mask = vis_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vision_mask = self.buffer_vis_mask
        text_mask = ~vision_mask
        assert vision_mask is not None
        assert vision_mask.dim() == 2

        has_vis_token, has_text_token = vision_mask.sum().item(), text_mask.sum().item()
        assert has_vis_token or has_text_token

        final_output = torch.empty_like(x)
        share_output = text_expert_output = vision_expert_output = 0
        if self.has_share_expert:
            share_output = self.share_down_proj(self.act_fn(self.share_gate_proj(x)) * self.share_up_proj(x))

        if has_text_token:
            text_hidden_states = x[text_mask, :]
            text_expert_output = self.down_proj(self.act_fn(self.gate_proj(text_hidden_states)) * self.up_proj(text_hidden_states))
        final_output[text_mask, :] = (text_expert_output if has_text_token else 0) + (share_output[text_mask, :] if self.has_share_expert else 0)

        if self.has_vis_expert and has_vis_token:
            vision_hidden_states = x[vision_mask, :]
            vision_expert_output = self.vision_down_proj(self.act_fn(self.vision_gate_proj(vision_hidden_states)) * self.vision_up_proj(vision_hidden_states))
        final_output[vision_mask, :] = ((vision_expert_output if has_vis_token else 0) +
                                        (share_output[vision_mask, :] if self.has_share_expert else 0))

        return final_output


class UniQwen3Model(Qwen3Model):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        # if accelerator.is_main_process:
        #     print("[debug] " * 3, config)
        # self.vision_embed_tokens = nn.Embedding(config.vision_vocab_size, config.hidden_size, self.padding_idx)

        self.vision_encode_layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_vision_encode_layers)]
        )
        self.vision_decode_layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_vision_decode_layers)]  ## 这里设定的 layer_idx 有问题啊。。不过不影响训练
        )
        if getattr(config, 'add_share_ffn', 0):
            for l_idx in range(config.num_vision_encode_layers, config.num_hidden_layers - config.num_vision_decode_layers):
                self.layers[l_idx].mlp = UniShareQwen3MLP(config)

        self.config = config
        self.vis_tokens, self.total_tokens, self.step = 0, 0, 0
        self.skip_loc = getattr(config, "skip_connection_loc", -1)

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        text_vocab_size = self.config.real_text_vocab_size
        vision_mask = shared_func_module.get_vision_mask(input_ids, self.config.real_text_vocab_size, self.config.vision_start_id)
        text_mask = ~vision_mask
        # set vis_mask buffer
        for n, mod in self.named_modules():
            if isinstance(mod, UniShareQwen3MLP):
                mod.set_buffer_vis_mask(vision_mask)

        if vision_mask.sum().item() == 0:
            # 如果没有 vis 数据，导致 vis 分支没有梯度，就会训练时 多卡间 无法正常同步
            raise ValueError("存在 batch 里面只有文本数据")

        output_attentions = False
        output_hidden_states = False
        use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        assert past_key_values is None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        assert self.config._attn_implementation == "flash_attention_2"
        assert input_ids is not None, "不支持输入 input_embeds"
        assert not output_hidden_states, "不支持 output hs"
        bs, seq_len = input_ids.shape
        assert bs == 1, "使用 data packing 来让 bs == 1"
        assert not self.config.all_modal_visible

        local_rank = os.environ.get("LOCAL_RANK", -1)

        if accelerator.is_main_process:
            self.vis_tokens += vision_mask.sum().item()
            self.total_tokens += input_ids.numel()
            self.step += 1
            if self.step % 100 == 0 and wandb.run:
                wandb.log({"vis_tokens_on_main_card": self.vis_tokens, "total_tokens_on_main_card": self.total_tokens})
        # assert (vision_mask.sum().item() % 1024) == 0, f"{num_vis_tokens}"

        vision_embeds = inputs_embeds[vision_mask][None]
        text_embeds = inputs_embeds[text_mask][None]

        v_pos_ids = position_ids[vision_mask][None]
        t_pos_ids = position_ids[text_mask][None]

        ori_attn_mask = attention_mask.clone()
        assert (attention_mask > 0).all()
        vision_attn_mask = ori_attn_mask[vision_mask][None]
        text_attn_mask = ori_attn_mask[text_mask][None]

        # use attn mask get flash attn kwargs
        cumulative_seqlens_q, max_length_q = get_flash_attention_args_tensorized(ori_attn_mask)
        flash_attn_kwargs["cumulative_seqlens_q"] = flash_attn_kwargs["cumulative_seqlens_k"] = cumulative_seqlens_q
        flash_attn_kwargs["max_length_q"] = flash_attn_kwargs["max_length_k"] = max_length_q

        v_flash_attn_kwargs = {}
        v_cumulative_seqlens_q, v_max_length_q = get_flash_attention_args_tensorized(vision_attn_mask)
        v_flash_attn_kwargs["cumulative_seqlens_q"] = v_flash_attn_kwargs["cumulative_seqlens_k"] = v_cumulative_seqlens_q
        v_flash_attn_kwargs["max_length_q"] = v_flash_attn_kwargs["max_length_k"] = v_max_length_q

        t_flash_attn_kwargs = {}
        t_cumulative_seqlens_q, t_max_length_q = get_flash_attention_args_tensorized(text_attn_mask)
        t_flash_attn_kwargs["cumulative_seqlens_q"] = t_flash_attn_kwargs["cumulative_seqlens_k"] = t_cumulative_seqlens_q
        t_flash_attn_kwargs["max_length_q"] = t_flash_attn_kwargs["max_length_k"] = t_max_length_q

        hidden_states = inputs_embeds
        text_states = text_embeds
        vision_states = vision_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        t_pos_emb = self.rotary_emb(text_states, t_pos_ids)
        v_pos_emb = self.rotary_emb(vision_states, v_pos_ids)

        # decoder layers
        all_hidden_states = None
        all_self_attns = None
        skip_res_states = None

        for layer_idx in range(self.config.num_vision_encode_layers):
            main_logger.debug(f"{local_rank} || in layer {layer_idx}")
            decoder_layer = self.layers[layer_idx]
            vision_layer = self.vision_encode_layers[layer_idx]
            # text layer
            text_outputs = decoder_layer(
                text_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=t_pos_emb,
                **t_flash_attn_kwargs,
            )
            # vision layer
            vision_outputs = vision_layer(
                vision_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=v_pos_emb,
                **v_flash_attn_kwargs,
            )

            if self.skip_loc == layer_idx:
                skip_res_states = vision_outputs[0]

            text_states = text_outputs[0]
            vision_states = vision_outputs[0]

        # merge states
        hidden_states[text_mask] = text_states[0]
        hidden_states[vision_mask] = vision_states[0]

        for layer_idx in range(self.config.num_vision_encode_layers,
                               self.config.num_hidden_layers - self.config.num_vision_decode_layers):
            # main_logger.debug(f" in layer {layer_idx}")
            decoder_layer = self.layers[layer_idx]
            # shared layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

        # split states
        text_states = hidden_states[text_mask][None]
        vision_states = hidden_states[vision_mask][None]

        for layer_idx in range(self.config.num_hidden_layers - self.config.num_vision_decode_layers,
                               self.config.num_hidden_layers):
            main_logger.debug(f"{local_rank} || in layer {layer_idx}")
            decoder_layer = self.layers[layer_idx]
            vision_layer = self.vision_decode_layers[layer_idx - self.config.num_hidden_layers]
            # text layer
            text_outputs = decoder_layer(
                text_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=t_pos_emb,
                **t_flash_attn_kwargs,
            )

            # skip connection
            if layer_idx - (self.config.num_hidden_layers - self.config.num_vision_decode_layers) == self.skip_loc:
                vision_states = vision_states + skip_res_states

            # vision layer
            vision_outputs = vision_layer(
                vision_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=v_pos_emb,
                **v_flash_attn_kwargs,
            )

            text_states = text_outputs[0]
            vision_states = vision_outputs[0]

        # merge states
        hidden_states[text_mask] = text_states[0]
        hidden_states[vision_mask] = vision_states[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def init_vision_weights(self):
        # Initialize vision encoder layers from corresponding text layers
        for i, vision_encode_layer in enumerate(self.vision_encode_layers):
            text_layer = self.layers[i]
            vision_encode_layer.load_state_dict(text_layer.state_dict())

        # Initialize vision decoder layers from corresponding text layers
        decode_start_idx = self.config.num_hidden_layers - self.config.num_vision_decode_layers
        for i, vision_decode_layer in enumerate(self.vision_decode_layers):
            text_layer_idx = decode_start_idx + i
            text_layer = self.layers[text_layer_idx]
            vision_decode_layer.load_state_dict(text_layer.state_dict())

        main_logger.info("Vision layers weights initialized from text layers.")


class UniQwen3ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.model = UniQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        # with torch.no_grad():
        #     # 统计 vis loss
        #     assert input_ids.shape[0] == 1
        #     if labels is not None and accelerator.is_main_process and accelerator.step % 100 == 0:
        #         vision_mask = shared_func_module.get_vision_mask(input_ids[0], self.config.real_text_vocab_size, self.config.vision_start_id)
        #         vis_logits = outputs.logits[:, vision_mask]
        #         labels = labels[:, vision_mask]
        #         vis_loss = ForCausalLMLoss(vis_logits, labels, self.config.vocab_size)
        #         wandb.log({"train-vis-loss": vis_loss.item()})

        return outputs


# =====================================================================
# ==               START OF INFERENCE-OPTIMIZED CODE                 ==
# =====================================================================

class UniQwen3Cache(DynamicCache):
    """
    A specialized KV cache for UniQwen3.

    This version is refactored for clarity and maintainability. It centralizes the
    cache-selection logic to avoid code duplication, directly addressing the inconvenience
    of handling separate list-based caches for different modalities.
    """

    def __init__(self, config: "UniQwen3Config", _distributed_cache_data: Optional[Iterable] = None):
        # The parent __init__ creates `self.key_cache` and `self.value_cache`.
        # We'll rename them to be explicit about their purpose.
        super().__init__(_distributed_cache_data)
        self.config = config

        # Explicitly name the default cache for text/shared layers
        self.text_key_cache = []
        self.text_value_cache = []

        self.key_cache, self.value_cache = self.text_key_cache, self.text_value_cache

        # Create the separate cache for vision-specific layers
        self.vision_key_cache: List[torch.Tensor] = []
        self.vision_value_cache: List[torch.Tensor] = []

        self.in_vision_context = False

    def set_context(self, in_vision: bool):
        """Sets the current token context (True for vision, False for text)."""
        self.in_vision_context = in_vision

    def _is_split_layer(self, layer_idx: int) -> bool:
        """Checks if a layer is a split-modality layer (vs. a shared one)."""
        is_encode = layer_idx < self.config.num_vision_encode_layers
        is_decode = layer_idx >= self.config.num_hidden_layers - self.config.num_vision_decode_layers
        return is_encode or is_decode

    def _get_active_cache(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Centralized logic to select the cache based on context."""
        if self.in_vision_context:
            return self.vision_key_cache, self.vision_value_cache
        return self.text_key_cache, self.text_value_cache

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Updates the correct cache by temporarily pointing `self.key_cache` to the active one."""
        # Only use the vision cache for split layers in a vision context
        if not self._is_split_layer(layer_idx) or not self.in_vision_context:
            # 理论上不需要再在这里设置，但是好像确实不设置不行。可能是 grandfather 的类又重新设置了？不懂。。
            self.key_cache, self.value_cache = self.text_key_cache, self.text_value_cache
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # The "pointer swap" trick: Temporarily replace the parent's cache pointers
        # with our vision cache pointers. This allows us to call super().update()
        # and leverage its existing, robust logic without re-implementing it.
        self.key_cache, self.value_cache = self.vision_key_cache, self.vision_value_cache

        try:
            # Let the parent method do all the heavy lifting (padding, concat, etc.)
            result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        finally:
            # Always restore the pointers to the default text cache
            self.key_cache, self.value_cache = self.text_key_cache, self.text_value_cache

        return result

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the cache for a layer, respecting the current modality context."""
        if self._is_split_layer(layer_idx):
            self.key_cache, self.value_cache = self._get_active_cache()
            try:
                # Let the parent method work on the selected cache
                item = super().__getitem__(layer_idx)
            finally:
                self.key_cache, self.value_cache = self.text_key_cache, self.text_value_cache
        else:
            item = super().__getitem__(layer_idx)
        return item

    @staticmethod
    def from_dynamic_cache(raw_cache: DynamicCache, config: UniQwen3Config):
        n_cache = UniQwen3Cache(config)
        n_cache.key_cache = raw_cache.key_cache
        n_cache.value_cache = raw_cache.value_cache
        n_cache._seen_tokens = raw_cache._seen_tokens
        return n_cache


class SdpaQwen3Attention(Qwen3Attention):
    """
    Qwen3Attention module adapted to use PyTorch's scaled_dot_product_attention (SDPA).
    It accepts a 4D attention mask for standard, causally-masked attention.
    """
    def __init__(self, config: UniQwen3Config, layer_idx: int, modal="text"):
        super().__init__(config, layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.modal = modal

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[UniQwen3Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        past_key_value.set_context(self.modal == "vision")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention heads
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply QK norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply rotary position embedding
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_position)

        # Use PyTorch's SDPA. The attention_mask is a boolean tensor where True indicates a masked position.
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask.bool(),
            is_causal=attention_mask is None,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # SDPA does not return attention weights
        return attn_output, None, past_key_value


class SdpaQwen3DecoderLayer(Qwen3DecoderLayer):
    """
    Qwen3DecoderLayer that uses the SdpaQwen3Attention module.
    It's modified to accept and pass the 4D attention_mask to the attention layer.
    """

    def __init__(self, config: UniQwen3Config, layer_idx: int, modal="text"):
        super().__init__(config, layer_idx)
        # Replace the standard attention with our SDPA variant
        self.self_attn = SdpaQwen3Attention(config, layer_idx, modal)
        # MLP 会产生一些无用计算，但是没关系，后面会被筛掉
        self.modal = modal

    def forward( # 修改了 forward 参数
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass the 4D attention_mask to the attention module
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class UniQwen3ModelInference(Qwen3Model):
    """
    The main inference model. It orchestrates the creation of 4D SDPA masks
    and calls the appropriate layers (shared vs. split) with the correct mask.
    """

    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.config = config
        # Use original layers for text and shared pathways
        self.layers = nn.ModuleList(
            [SdpaQwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # Create separate vision-specific layers
        self.vision_encode_layers = nn.ModuleList(
            [SdpaQwen3DecoderLayer(config, i, modal="vision") for i in range(config.num_vision_encode_layers)]
        )
        self.vision_decode_layers = nn.ModuleList(
            [SdpaQwen3DecoderLayer(config, i + (config.num_hidden_layers - config.num_vision_decode_layers), modal="vision")
             for i in range(config.num_vision_decode_layers)]
        )
        if getattr(config, 'add_share_ffn', 0):
            for l_idx in range(config.num_vision_encode_layers, config.num_hidden_layers - config.num_vision_decode_layers):
                self.layers[l_idx].mlp = UniShareQwen3MLP(config)

        if getattr(config, "skip_connection_loc", -1) != -1:
            raise NotImplementedError("目前测试下来 skip connection 没有带来增益")

        # Cache for vision mask and the generated 4D SDPA masks
        self.vision_mask_cache = None
        self.sdpa_causal_mask = None
        self.sdpa_text_mask = None
        self.sdpa_vision_mask = None

        self.post_init()

    def _prepare_sdpa_masks(self, vision_mask: torch.Tensor, pad_mask: torch.Tensor, q_len: int, kv_len: int):
        """Creates the three required 4D masks for SDPA. Masks are True for positions to be ignored."""
        device = vision_mask.device

        # Causal mask: prevents attending to future tokens.
        causal_mask = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=device), diagonal=kv_len - q_len)
        # Add dimensions for broadcasting: (1, 1, q_len, kv_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask: (bsz, kv_len) -> (bsz, 1, 1, kv_len)
        expanded_pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        # 1. Shared layer mask (causal + padding)
        self.sdpa_causal_mask = causal_mask & expanded_pad_mask

        # 2. Modality-specific masks
        q_is_vision = vision_mask[:, -q_len:]   # (bsz, q_len)
        kv_is_vision = vision_mask              # (bsz, kv_len)

        text_modality_mask = (~q_is_vision).unsqueeze(2) & (~kv_is_vision).unsqueeze(1)
        self.sdpa_text_mask = self.sdpa_causal_mask & text_modality_mask.unsqueeze(1)

        vision_modality_mask = q_is_vision.unsqueeze(2) & kv_is_vision.unsqueeze(1)
        self.sdpa_vision_mask = self.sdpa_causal_mask & vision_modality_mask.unsqueeze(1)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask=None,
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> BaseModelOutputWithPast:
        bsz, q_len = input_ids.shape
        use_cache = True

        _tmp_ffn_vis_mask = get_vision_mask(input_ids, self.config.real_text_vocab_size, self.config.vision_start_id)
        # set vis_mask buffer
        for n, mod in self.named_modules():
            if isinstance(mod, UniShareQwen3MLP):
                mod.set_buffer_vis_mask(_tmp_ffn_vis_mask)

        if past_key_values is None:
            past_key_values = DynamicCache()

        assert isinstance(past_key_values, DynamicCache)
        if not isinstance(past_key_values, UniQwen3Cache):
            past_key_values = UniQwen3Cache.from_dynamic_cache(past_key_values, self.config)

        # Manage vision mask and position caches across prefill and decode steps
        if q_len > 1: # Prefill
            self.vision_mask_cache = shared_func_module.get_vision_mask(input_ids, self.config.real_text_vocab_size, self.config.vision_start_id)
            cache_position = torch.arange(q_len, device=input_ids.device)
            kv_attention_mask = attention_mask
        else: # Decode
            is_vision_token = shared_func_module.get_vision_mask(input_ids, self.config.real_text_vocab_size, self.config.vision_start_id)
            self.vision_mask_cache = torch.cat([self.vision_mask_cache, is_vision_token], dim=1)
            past_seq_len = self.vision_mask_cache.shape[1] - 1
            cache_position = torch.tensor([past_seq_len], device=input_ids.device)
            # For decode, assume past tokens are valid and only check the current token's padding
            kv_attention_mask = attention_mask

        kv_len = self.vision_mask_cache.shape[1]
        self._prepare_sdpa_masks(self.vision_mask_cache, kv_attention_mask, q_len, kv_len)

        # Embed inputs and prepare for layers
        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds

        for layer_idx in range(self.config.num_hidden_layers):
            is_encode_layer = layer_idx < self.config.num_vision_encode_layers
            is_decode_layer = layer_idx >= (self.config.num_hidden_layers - self.config.num_vision_decode_layers)

            if is_encode_layer or is_decode_layer:
                # --- Split Layer Logic ---
                text_layer = self.layers[layer_idx]
                vision_layer = self.vision_encode_layers[layer_idx] if is_encode_layer else \
                    self.vision_decode_layers[layer_idx - self.config.num_hidden_layers]

                t_outputs, _ = text_layer(
                    hidden_states, position_embeddings, self.sdpa_text_mask, past_key_values, cache_position
                )
                v_outputs, _ = vision_layer(
                    hidden_states, position_embeddings, self.sdpa_vision_mask, past_key_values, cache_position
                )

                # Merge outputs based on the modality of each token position
                current_vision_mask = self.vision_mask_cache[:, -q_len:]
                hidden_states = torch.where(current_vision_mask.unsqueeze(-1), v_outputs, t_outputs)
            else:
                # --- Shared Layer Logic ---
                decoder_layer = self.layers[layer_idx]
                hidden_states, _ = decoder_layer(
                    hidden_states, position_embeddings, self.sdpa_causal_mask, past_key_values, cache_position
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class UniQwen3ForCausalLMInference(Qwen3ForCausalLM):
    """
    The final Causal LM model for inference, which wraps the UniQwen3ModelInference.
    It's designed for generation tasks.
    """

    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.model = UniQwen3ModelInference(config)
        self.post_init()


if __name__ == "__main__":
    # 写一个测试，输入随机，注意一定要同时包含 text 和 vision token。验证输出的 logits 是否相同
    model_path = "../checkpoint-80000"

    # Use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set inference mode
    torch.set_grad_enabled(False)

    # 1. Load the original training model
    print("Loading original training model...")
    try:
        # The training model requires flash-attn, so we use bfloat16 and move to GPU
        original_model = UniQwen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2"
        )
        original_model.eval()
        config = original_model.config
    except Exception as e:
        print(f"Failed to load original model from {model_path}. Error: {e}")
        print("Skipping test.")
        exit()


    # 2. Create the inference-optimized model and load weights
    print("Creating inference model and loading weights...")
    inference_model = UniQwen3ForCausalLMInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    inference_model.eval()

    # 3. Prepare a mixed-modality input
    # Ensure there are both text (< real_text_vocab_size) and vision tokens (>= real_text_vocab_size)
    text_token_1 = 100
    text_token_2 = 200
    vision_start_id = config.vision_start_id
    vision_token_1 = config.real_text_vocab_size + 10
    vision_token_2 = config.real_text_vocab_size + 20

    # input_ids = torch.tensor([[text_token_1, text_token_2, vision_start_id, vision_token_1, vision_token_2, text_token_2]], device=device)
    input_ids = torch.cat([torch.randint(config.real_text_vocab_size, (1, 321), device=device),
                           torch.randint(config.real_text_vocab_size, config.real_text_vocab_size + 8191, (1, 400), device=device),
                           torch.randint(config.real_text_vocab_size, (1, 321), device=device),], dim=-1)
    attention_mask = torch.ones_like(input_ids)

    print(f"\nInput IDs: {input_ids.tolist()}")

    # 4. Run both models and get logits
    print("Running original model...")
    # The original model requires a batch size of 1 and a specific attention mask format for its custom FA2 logic
    original_output = original_model(input_ids=input_ids, attention_mask=attention_mask)
    original_logits = original_output.logits

    print("Running inference model...")
    inference_output = inference_model(input_ids=input_ids, attention_mask=attention_mask)
    inference_logits = inference_output.logits

    # 5. Compare the outputs
    print("\nComparing logits...")
    print(f"Original model logits shape: {original_logits.shape}")
    print(f"Inference model logits shape: {inference_logits.shape}")

    # Check if logits are close enough to account for minor floating point differences
    are_close = torch.allclose(original_logits, inference_logits, atol=1e-2, rtol=1e-2)

    if are_close:
        print("\n✅ Test Passed: Logits from both models are identical.")
    else:
        print("\n❌ Test Failed: Logits do not match.")
        # Print differences for debugging
        diff = torch.abs(original_logits - inference_logits)
        print(f"   - Max absolute difference: {diff.max().item()}")
        print(f"   - Mean absolute difference: {diff.mean().item()}")

    # Optional: Print a slice of the logits for manual inspection
    print("\nSample logits (last token):")
    print("Original model :", original_logits[0, -1, :10])
    print("Inference model:", inference_logits[0, -1, :10])
    print("=" * 10)
    print("Original model :", original_logits[0, -1, -10:])
    print("Inference model:", inference_logits[0, -1, -10:])\

    ## 上面测试了 prefill，现在测试decode，你不用真的 decode，只需要依次随机输入 token 来模拟 decode
    ## 用来训练的模型不支持 use_cache，所以每次还是传入全部的 token，因此我们只对比最后一个 token 的 logits
    # =====================================================================
    # ==                 START OF DECODING TEST                          ==
    # =====================================================================
    print("\n" + "="*20 + " Starting Decode Test " + "="*20)

    # Initialize KV cache for the inference model
    past_key_values = None
    all_decode_steps_passed = True
    is_prefill = True

    # Loop through the sequence, one token at a time to simulate decoding
    for i in range(400, input_ids.shape[1] + 1):
        current_step = i - 1
        print(f"\n--- Step {current_step}: Processing token {input_ids[0, current_step].item()} ---")

        # --- Original Model (no cache) ---
        # It gets the full sequence up to the current token every time
        current_input_ids_orig = input_ids[:, :i]
        current_attn_mask_orig = attention_mask[:, :i]
        original_output_step = original_model(input_ids=current_input_ids_orig, attention_mask=current_attn_mask_orig)
        # We only care about the logits for the very last token
        original_logits_step = original_output_step.logits[:, -1, :]

        # --- Inference Model (with cache) ---
        if is_prefill:  # Prefill step
            current_input_ids_infer = input_ids[:, :i]
            is_prefill = False
        else:  # Decode step
            current_input_ids_infer = input_ids[:, i-1:i]
        current_attn_mask_infer = attention_mask[:, :i]

        inference_output_step = inference_model(
            input_ids=current_input_ids_infer,
            attention_mask=current_attn_mask_infer,
            past_key_values=past_key_values,
            use_cache=True
        )
        inference_logits_step = inference_output_step.logits[:, -1, :]
        # Update the cache for the next step
        past_key_values = inference_output_step.past_key_values

        # --- Compare logits for the current token ---
        step_are_close = torch.allclose(original_logits_step, inference_logits_step, atol=1e-1, rtol=1e-2)

        if step_are_close:
            print(f"✅ Step {current_step} Passed: Logits match.")
        else:
            all_decode_steps_passed = False
            print(f"❌ Step {current_step} Failed: Logits DO NOT match.")
            diff = torch.abs(original_logits_step - inference_logits_step)
            print(f"   - Max absolute difference: {diff.max().item()}")

    # Final summary of the decode test
    print("\n" + "="*20 + " Decode Test Summary " + "="*20)
    if all_decode_steps_passed:
        print("✅✅✅ All decode steps passed successfully!")
    else:
        print("❌❌❌ At least one decode step failed.")