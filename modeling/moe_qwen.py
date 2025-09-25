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
    DynamicCache, Qwen3Attention, Qwen3MLP, ACT2FN, apply_rotary_pos_emb,  # noqa
    Union, KwargsForCausalLM, CausalLMOutputWithPast  # noqa
)
from transformers.loss.loss_utils import ForCausalLMLoss

from modeling.shared_func_module import get_flash_attention_args_tensorized, get_vision_mask, UniQwen3Config
from accelerate import Accelerator
from tools.log import main_logger

accelerator = Accelerator()


class HardMoeLayer(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        self.vis_gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.vis_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.vis_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.buffer_vis_mask = None

    def set_buffer_vis_mask(self, vis_mask):
        self.buffer_vis_mask = vis_mask

    def forward(self, x):
        # assert x.shape[0] == 1, "only data packing"
        assert self.buffer_vis_mask.dim() == 2
        vis_mask = self.buffer_vis_mask
        text_mask = ~vis_mask
        down_proj = torch.empty_like(x)
        if vis_mask.any():
            vx = x[vis_mask]
            out_v = self.vis_down_proj(self.act_fn(self.vis_gate_proj(vx)) * self.vis_up_proj(vx))
            down_proj[vis_mask] = out_v
        if text_mask.any():
            tx = x[text_mask]
            out_t = self.down_proj(self.act_fn(self.gate_proj(tx)) * self.up_proj(tx))
            down_proj[text_mask] = out_t
        
        return down_proj
    

class DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: UniQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        if (layer_idx + 1) % config.decoder_sparse_step == 0:
            self.mlp = HardMoeLayer(config)
        else:
            self.mlp = Qwen3MLP(config)
    

class HardMoeModel(Qwen3Model):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()


class HardMoeQwen3CausalLM(Qwen3ForCausalLM):
    def __init__(self, config: UniQwen3Config):
        super().__init__(config)
        self.model = HardMoeModel(config)

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
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        vis_mask = get_vision_mask(input_ids, self.config.real_text_vocab_size, self.config.vision_start_id)
        for n, mod in self.named_modules():
            if isinstance(mod, HardMoeLayer):
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