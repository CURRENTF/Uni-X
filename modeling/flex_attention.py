import torch
from typing import Optional

# 导入 flex_attention 和 create_block_mask 函数
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    torch._dynamo.config.cache_size_limit = 1000
    torch._dynamo.config.accumulated_cache_size_limit = 1000
    # flex_attention = torch.compile(flex_attention, dynamic=False)
    create_block_mask = torch.compile(create_block_mask, dynamic=False)
except ImportError:
    raise ImportError("Please ensure you have a PyTorch version that supports flex_attention.")


def _is_power_of_two(n: int) -> bool:
    """检查一个数是否是2的整数次幂。"""
    return (n > 0) and (n & (n - 1) == 0)

def _next_power_of_two(n: int) -> int:
    """找到大于或等于n的最小的2的整数次幂。"""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# compilation args taken from https://github.com/pytorch/pytorch/issues/142817
@torch.compile(fullgraph=True, dynamic=False)
def _flex_attention(
    query: torch.Tensor,  # (B, Hq, L, E)
    key: torch.Tensor,    # (B, Hkv, S, E)
    value: torch.Tensor,  # (B, Hkv, S, E)
    score_mod: Optional = None,
    block_mask: Optional = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional = None,
):

    batch_size, n_q_heads, q_len, head_dims = query.shape
    _, n_kv_heads, kv_len, _ = key.shape

    min_len = min(q_len, kv_len)
    group_size = n_q_heads // n_kv_heads

    if enable_gqa and min_len < 128 and not _is_power_of_two(group_size):
        new_group_size = _next_power_of_two(group_size)
        extra_heads_per_group = new_group_size - group_size

        n_groups = n_q_heads // group_size

        new_n_q_heads = n_q_heads + n_groups * extra_heads_per_group

        # for each group, append extra_heads_per_group of fake heads
        query = torch.concat([
            query.view(batch_size, n_groups, group_size, q_len, head_dims),
            query.new_zeros(batch_size, n_groups, extra_heads_per_group, q_len, head_dims),
        ], dim=2).view(batch_size, new_n_q_heads, q_len, head_dims).contiguous()

        result = flex_attention(
            query, key, value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse,
            kernel_options=kernel_options,
        )

        attn_out = result if not return_lse else result[0]

        attn_out = attn_out.view(batch_size, n_groups, new_group_size, q_len, head_dims)
        attn_out = attn_out[:, :, :-extra_heads_per_group, :, :]
        attn_out = attn_out.reshape(batch_size, n_q_heads, q_len, head_dims)

        return attn_out if not return_lse else (attn_out, result[1])

    # If no padding is needed, just run flex_attention directly
    return flex_attention(
        query, key, value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
    )
