import torch
from transformers import Qwen3Config, Qwen2Config


# !important: 这个函数可能会被 monkey patch
def get_vision_mask(input_ids, ori_text_vocab_size, vision_start_id) -> torch.Tensor:
    """根据 input_ids 确定哪些 token 属于视觉模态。"""
    return (input_ids >= ori_text_vocab_size) | (input_ids == vision_start_id)


def reset_get_vis_mask(vis_sep_tokens_id: torch.Tensor):
    assert vis_sep_tokens_id.dim() == 1
    vis_sep_tokens_id = vis_sep_tokens_id[:, None]

    def new_get(input_ids: torch.Tensor, ori_text_vocab_size, vision_start_id) -> torch.Tensor:
        """根据 input_ids 确定哪些 token 属于视觉模态。"""
        assert input_ids.dim() <= 2
        raw_shape = input_ids.shape
        input_ids = input_ids.view(-1)
        return ((input_ids >= ori_text_vocab_size) | (input_ids == vision_start_id) | (input_ids == vis_sep_tokens_id.to(input_ids.device)).sum(0).bool()).view(raw_shape)

    return new_get

def get_flash_attention_args_tensorized(attention_mask: torch.Tensor):
    """
    根据输入的 attention mask (prefill 阶段) 计算 cumulative_seqlens_q 和 max_length_q。
    此版本使用纯张量操作，效率更高。

    Args:
      attention_mask: shape 为 (1, seq_len) 的 PyTorch 张量。

    Returns:
      一个元组，包含:
        - cumulative_seqlens_q: 一个 PyTorch 张量，表示每个序列在批处理中的累积长度。
        - max_length_q: 一个整数，表示批处理中最长序列的长度。
    """
    # 确保输入是二维的
    if attention_mask.dim() != 2 or attention_mask.shape[0] != 1:
        raise ValueError("attention_mask 的 shape 必须是 (1, seq_len)")

    # 如果 mask 为空，直接返回
    if attention_mask.numel() == 0:
        return torch.tensor([0], device=attention_mask.device, dtype=torch.int32), 0

    # torch.unique 在 attention_mask 中寻找唯一的序列 ID，并返回它们的计数（即每个序列的长度）
    # 在 prefill 阶段，attention_mask 通常是 [1, 1, 1, 2, 2, 3, 3, 3, 3] 的形式
    _, seqlens = torch.unique(attention_mask, return_counts=True)

    # 最长序列长度
    max_length_q = torch.max(seqlens).item()

    # 计算序列长度的累积和，并在开头添加 0，以符合 flash_attn 的格式要求
    cumulative_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
    zero_tensor = torch.zeros(1, device=cumulative_seqlens.device, dtype=torch.int32)
    cumulative_seqlens_q = torch.cat((zero_tensor, cumulative_seqlens))

    return cumulative_seqlens_q, max_length_q


class CustomConfigMixin:
    def set_extra_args(self, **kwargs):
        self.num_vision_encode_layers = kwargs.pop("vision_encode_layers")
        self.num_vision_decode_layers = kwargs.pop("vision_decode_layers")
        self.vision_vocab_size = kwargs.pop("vision_vocab_size")
        self.all_modal_visible = kwargs.pop("all_modal_visible")
        self.real_text_vocab_size = kwargs.pop("real_text_vocab_size")
        self.vision_start_id = kwargs.pop("vision_start_id")
        self.add_sep_for_vis = kwargs.pop("add_sep_for_vis")
        self.vis_sep_tokens = kwargs.pop("vis_sep_tokens")
        self.vis_sep_lens = kwargs.pop("vis_sep_lens")
        self.skip_connection_loc = kwargs.pop("skip_connection_loc")
        # ---- uni share
        self.add_share_ffn = kwargs.pop("add_share_ffn")
        self.ffn_share_size = kwargs.pop("ffn_share_size")
        self.ffn_vision_size = kwargs.pop("ffn_vision_size")
        self.use_share_attn = kwargs.pop("use_share_attn")
        self.use_share_ffn = kwargs.pop("use_share_ffn")
        self.attn_v_q_heads = kwargs.pop("attn_v_q_heads")
        # ---- baseline moe
        self.decoder_sparse_step = kwargs.pop("decoder_sparse_step")


class UniQwen3Config(CustomConfigMixin, Qwen3Config):
    pass


class UniQwen2Config(CustomConfigMixin, Qwen2Config):
    pass