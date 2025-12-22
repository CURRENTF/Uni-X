import torch

from torch import nn
# from transformers.models.qwen2.modeling_qwen2 import *
from transformers.models.qwen3.modeling_qwen3 import *
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Optional, Cache, Unpack, FlashAttentionKwargs, Qwen3Attention, apply_rotary_pos_emb,  # noqa
    Qwen3MLP, BaseModelOutputWithPast, Qwen3RMSNorm, eager_attention_forward, Callable, ALL_ATTENTION_FUNCTIONS,  # noqa
    Union, KwargsForCausalLM,  # noqa
)
from modeling.shared_func_module import UniQwen3Config
from modeling import shared_func_module
from accelerate import Accelerator

accelerator = Accelerator()


class MoTQwen3Attention(Qwen3Attention):
    def __init__(self, config: UniQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.config = config

        self.vis_q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.vis_k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.vis_v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.vis_o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.vis_q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.vis_k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.buffer_vis_mask = None

    def set_buffer_vis_mask(self, vis_mask):
        self.buffer_vis_mask = vis_mask

    def get_modality_states(self, x, wq, wk, wv, q_norm, k_norm, seq_len):
        assert x.dim() == 2 and x.shape[0] == seq_len
        query_states = q_norm(wq(x).view(seq_len, -1, self.head_dim))
        key_states = k_norm(wk(x).view(seq_len, -1, self.head_dim))
        value_states = wv(x).view(seq_len, -1, self.head_dim)
        return query_states, key_states, value_states

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # 想兼容推理，需要考虑什么？1. 没有 vis 或 text tokens 2. bs > 1
        vis_mask: torch.Tensor = self.buffer_vis_mask
        assert vis_mask is not None and vis_mask.dim() == 2
        if vis_mask.shape[0] != 1:
            assert (vis_mask[0] == vis_mask[1]).all()
        text_mask = ~vis_mask

        bs, seq_len, d = hidden_states.shape
        q = torch.empty(bs, seq_len, self.config.num_attention_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        k = torch.empty(bs, seq_len, self.config.num_key_value_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        v = torch.empty(bs, seq_len, self.config.num_key_value_heads, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        text_tokens = text_mask.sum().item()
        if text_tokens > 0:
            tx = hidden_states[text_mask]
            tq, tk, tv = self.get_modality_states(tx, self.q_proj, self.k_proj, self.v_proj, self.q_norm, self.k_norm, text_tokens)
            q[text_mask], k[text_mask], v[text_mask] = tq, tk, tv

        vis_tokens = vis_mask.sum().item()
        if vis_tokens > 0:
            vx = hidden_states[vis_mask]
            vq, vk, vv = self.get_modality_states(vx, self.vis_q_proj, self.vis_k_proj, self.vis_v_proj, self.vis_q_norm, self.vis_k_norm, vis_tokens)
            q[vis_mask], k[vis_mask], v[vis_mask] = vq, vk, vv

        q = q.view(bs, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            q, k, v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(bs, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MoTQwen3MLP(Qwen3MLP):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.config = config
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

        if has_text_token:
            text_hidden_states = x[text_mask, :]
            text_expert_output = self.down_proj(self.act_fn(self.gate_proj(text_hidden_states)) * self.up_proj(text_hidden_states))
            final_output[text_mask, :] = text_expert_output

        if has_vis_token:
            vision_hidden_states = x[vision_mask, :]
            vision_expert_output = self.vision_down_proj(self.act_fn(self.vision_gate_proj(vision_hidden_states)) * self.vision_up_proj(vision_hidden_states))
            final_output[vision_mask, :] = vision_expert_output

        return final_output


class MoTQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: UniQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MoTQwen3Attention(config, layer_idx)
        self.mlp = MoTQwen3MLP(config)
        self.config = config


class MoTQwen3Model(Qwen3Model):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList([MoTQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.config = config
        self.attn_heads = config.num_attention_heads
        self.post_init()


class MoTQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.model = MoTQwen3Model(config)
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
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
    ):
        vis_mask = shared_func_module.get_vision_mask(input_ids, self.config.real_text_vocab_size, self.config.vision_start_id)
        for n, mod in self.named_modules():
            if hasattr(mod, "set_buffer_vis_mask"):
                mod.set_buffer_vis_mask(vis_mask)

        # Pass all arguments to the parent class's forward method
        return super().forward(
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


if __name__ == "__main__":
    # 写一个测试，输入随机，注意一定要同时包含 text 和 vision token。验证输出的 logits 是否相同
    model_path = "../mock/ckpts/Qwen2.5-1.5B-uni-X/data-v4-1.5B-mot-t2i_80-ignore_ins-lr5e-5-ablation-7card-rerun/20250924_1921"

    # Use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set inference mode
    torch.set_grad_enabled(False)

    # 1. Create the inference-optimized model and load weights
    print("Creating inference model and loading weights...")
    inference_model = MoTQwen3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    config = inference_model.config
    inference_model.eval()

    # 2. Prepare a mixed-modality input
    # Ensure there are both text (< real_text_vocab_size) and vision tokens (>= real_text_vocab_size)
    text_token_1 = 100
    text_token_2 = 200
    vision_start_id = config.vision_start_id
    vision_token_1 = config.real_text_vocab_size + 10
    vision_token_2 = config.real_text_vocab_size + 20

    # input_ids = torch.tensor([[text_token_1, text_token_2, vision_start_id, vision_token_1, vision_token_2, text_token_2]], device=device)
    input_ids = torch.cat([torch.randint(config.real_text_vocab_size, (1, 321), device=device),
                           torch.randint(config.real_text_vocab_size, config.real_text_vocab_size + 8191, (1, 400), device=device),
                           torch.randint(config.real_text_vocab_size, (1, 321), device=device), ], dim=-1)
    attention_mask = torch.ones_like(input_ids)

    print(f"\nInput IDs shape: {input_ids.shape}")
    # TODO 写一个不用cache（prefill）和用cache（decode）的logits对比，来保证推理时是正确的
    print("\n--- Testing consistency between prefill and decode ---")

    # 1. Prefill pass: 一次性处理整个序列，不使用预先存在的cache
    print("Running prefill pass for the whole sequence...")
    with torch.no_grad():
        prefill_outputs = inference_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    # 获取最后一个token的logits
    last_logit_prefill = prefill_outputs.logits[:, -1, :]
    print("Prefill pass complete.")

    # 2. Decode pass: 先处理n-1个token生成cache，然后用cache处理最后一个token
    print("\nRunning decode pass (prefill n-1 tokens, then decode 1 token)...")
    input_ids_pre_cache = input_ids[:, :-1]
    attention_mask_pre_cache = attention_mask[:, :-1]
    last_token_input = input_ids[:, -1:]

    # 使用前n-1个token生成cache
    with torch.no_grad():
        pre_cache_outputs = inference_model(input_ids=input_ids_pre_cache, attention_mask=attention_mask_pre_cache, use_cache=True)
        past_key_values = pre_cache_outputs.past_key_values

    # 使用生成的cache和最后一个token进行decode
    # 注意此时的attention_mask需要覆盖包括cache在内的全部序列长度
    attention_mask_for_decode = torch.cat([attention_mask_pre_cache, torch.ones_like(last_token_input)], dim=-1)
    with torch.no_grad():
        decode_outputs = inference_model(
            input_ids=last_token_input,
            past_key_values=past_key_values,
            attention_mask=attention_mask_for_decode,
            use_cache=True
        )
    # decode模式输出的序列长度为1
    last_logit_decode = decode_outputs.logits[:, 0, :]
    print("Decode pass complete.")

    # 3. 对比两次得到的logits
    print("\nComparing logits...")
    are_close = torch.allclose(last_logit_prefill, last_logit_decode, atol=1e-1)  # 设置一个合理的容忍度
    max_diff = torch.max(torch.abs(last_logit_prefill - last_logit_decode))

    print(f"Logits are close: {are_close}")
    print(f"Maximum absolute difference: {max_diff.item()}")

    if are_close:
        print("✅ Test Passed: Prefill and decode logits are consistent.")
    else:
        print("❌ Test Failed: Prefill and decode logits are NOT consistent.")