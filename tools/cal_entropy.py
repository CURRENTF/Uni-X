from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer, Qwen2Tokenizer
from tqdm import *

import torch
import collections
import math
import json
from typing import List
import time # 导入 time 模块

# 只单卡
def calculate_ngram_conditional_entropy(token_sequences: List[torch.LongTensor], n: int = 2, device: str = 'cuda:0') -> float:
    # TODO 给这个函数加一些执行进度的print，和一些过程的计时 (DONE)
    """
    根据给定的token序列，计算 n-gram 条件信息熵。(已优化)

    此方法遵循您提供的PDF文档中的计算方式，衡量在给定前 n-1 个字符的条件下，
    下一个字符出现的不确定性。利用 PyTorch 进行了向量化和 GPU 加速。

    计算公式:
     - n=1: H = - Σ p(w_1) * log2(p(w_1))
     - n>1: H = - Σ p(w_1,...,w_n) * log2(p(w_n|w_1,...,w_{n-1}))
    """
    start_time = time.time()
    print(f"\nCalculating for n={n}...")

    if not isinstance(n, int) or n < 1:
        raise ValueError("n 必须是大于或等于1的整数。")

    # 如果输入为空，熵为0
    if not token_sequences:
        return 0.0

    # --- Case n=1: Standard Entropy (向量化实现) ---
    if n == 1:
        # 将所有 token 序列拼接成一个长张量
        all_tokens = torch.cat(token_sequences)
        total_tokens = all_tokens.numel()

        if total_tokens == 0:
            return 0.0

        # 高效计算每个 token 的频率
        _, counts = torch.unique(all_tokens, return_counts=True)

        # 计算概率 p(x)
        p = counts.float() / total_tokens

        # 计算信息熵: H = - Σ p(x) * log2(p(x))
        entropy = -torch.sum(p * torch.log2(p))
        print(f"n=1 calculation finished in {time.time() - start_time:.2f}s.")
        return entropy.item()

    # --- Case n > 1: Conditional Entropy (向量化实现) ---
    torch.cuda.synchronize()
    # 使用 unfold 高效地从每个序列中提取 n-grams 和 (n-1)-grams (即前缀)
    # unfold 创建了一个滑动窗口视图，避免了 Python 循环
    # 稍微有一些误差感觉问题不大，所以先全部cat起来
    long_seq = torch.cat(token_sequences, dim=-1).to(device)
    all_ngrams = long_seq.unfold(0, n, 1)
    all_prefixes = long_seq.unfold(0, n - 1, 1)
    print(f"[{time.time() - start_time:.2f}s] Step 1: Prepared {all_ngrams.shape[0]} n-grams.")

    total_ngrams = all_ngrams.shape[0]
    if total_ngrams == 0:
        return 0.0

    # 高效计算 n-gram 的联合频率 count(w_1, ..., w_n)
    unique_ngrams, ngram_counts = torch.unique(all_ngrams, dim=0, return_counts=True)

    # 高效计算前缀的频率 count(w_1, ..., w_{n-1})
    unique_prefixes, prefix_counts = torch.unique(all_prefixes, dim=0, return_counts=True)
    print(f"[{time.time() - start_time:.2f}s] Step 2: Counted {unique_ngrams.shape[0]} unique n-grams and {unique_prefixes.shape[0]} unique prefixes.")

    # 为了计算条件概率，我们需要为每个 unique_ngram 找到其对应前缀的计数值。
    # 我们创建一个从前缀到其计数的查找表 (哈希表) 来加速这个过程。
    prefix_counts_map = {tuple(p.tolist()): c.item() for p, c in zip(unique_prefixes.cpu(), prefix_counts.cpu())}

    # 提取每个 unique_ngram 的前缀
    ngram_prefixes = unique_ngrams[:, :-1].cpu()

    # 使用查找表，为每个 unique_ngram 找到其前缀的计数值
    corresponding_prefix_counts = torch.tensor(
        [prefix_counts_map[tuple(p.tolist())] for p in ngram_prefixes],
        device=device,
        dtype=torch.float
    )
    print(f"[{time.time() - start_time:.2f}s] Step 3: Built and used prefix count map.")

    # 转换为 float 以进行后续计算
    ngram_counts = ngram_counts.float()

    # 计算联合概率 p(w_1, ..., w_n)
    p_ngram = ngram_counts / total_ngrams

    # 计算条件概率 p(w_n | w_1, ..., w_{n-1}) = count(ngram) / count(prefix)
    p_conditional = ngram_counts / corresponding_prefix_counts

    # 过滤掉概率为0的情况，避免 log(0) 导致 nan
    valid_indices = p_conditional > 0

    # 计算条件熵: H = - Σ p(ngram) * log2(p_cond)
    conditional_entropy = -torch.sum(
        p_ngram[valid_indices] * torch.log2(p_conditional[valid_indices])
    )

    print(f"[{time.time() - start_time:.2f}s] Step 4: Final entropy calculation complete.")
    return conditional_entropy.item()


def calculate_ngram_conditional_entropy_optimized(token_sequences: List[torch.LongTensor], n: int = 2, device: str = 'cuda') -> float:
    """
    根据给定的token序列，计算 n-gram 条件信息熵。(已优化)

    此优化版本将所有计算保留在 GPU 上，避免了低效的 GPU-CPU 数据传输
    和 Python 字典操作，显著提升了 n > 1 时的计算性能。
    """
    start_time = time.time()
    print(f"\nCalculating for n={n} (Optimized)...")

    if not isinstance(n, int) or n < 1:
        raise ValueError("n 必须是大于或等于1的整数。")
    if not token_sequences:
        return 0.0

    # --- Case n=1: 与原版相同 ---
    if n == 1:
        all_tokens = torch.cat(token_sequences).to(device)
        total_tokens = all_tokens.numel()
        if total_tokens == 0:
            return 0.0
        _, counts = torch.unique(all_tokens, return_counts=True)
        p = counts.float() / total_tokens
        entropy = -torch.sum(p * torch.log2(p))
        print(f"n=1 calculation finished in {time.time() - start_time:.2f}s.")
        return entropy.item()

    # --- Case n > 1: Conditional Entropy (全程 GPU 优化) ---
    long_seq = torch.cat(token_sequences, dim=-1).to(device)

    # 从长序列中高效地提取所有 n-grams
    all_ngrams = long_seq.unfold(0, n, 1)
    print(f"[{time.time() - start_time:.2f}s] Step 1: Prepared {all_ngrams.shape[0]} n-grams.")

    total_ngrams = all_ngrams.shape[0]
    if total_ngrams == 0:
        return 0.0

    # 检查可用GPU数量，若多于1个，则启用并行计算
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"[{time.time() - start_time:.2f}s] Found {num_gpus} GPUs. Using multi-GPU processing.")

        # ==================== 多 GPU 并行计算 ====================
        # 1. 字典序排序：确保相同前缀的 n-gram 在分块时能聚集在一起
        print(f"[{time.time() - start_time:.2f}s] Step 2.1: Starting lexicographical sort on main GPU...")
        indices = torch.arange(all_ngrams.shape[0], device=device)
        for col in range(n - 1, -1, -1):
            # 稳定排序是保证字典序正确的关键
            indices = indices[torch.argsort(all_ngrams[indices, col], stable=True)]
        all_ngrams = all_ngrams[indices]
        del indices
        torch.cuda.empty_cache()
        print(f"[{time.time() - start_time:.2f}s] Step 2.2: Sorted all n-grams.")

        # 2. 分块并将任务分发到各个GPU
        chunks = torch.tensor_split(all_ngrams, num_gpus, dim=0)
        total_entropy_sum = 0.0

        for i, chunk in enumerate(chunks):
            gpu_device = f'cuda:{i}'
            chunk = chunk.to(gpu_device)
            print(f"[{time.time() - start_time:.2f}s] Processing chunk {i+1}/{num_gpus} on {gpu_device} ({chunk.shape[0]} n-grams)...")

            # --- 在单个GPU上执行与原版类似的计算 ---
            chunk_prefixes = chunk[:, :-1]
            unique_ngrams, ngram_inverse, ngram_counts = torch.unique(
                chunk, dim=0, return_inverse=True, return_counts=True
            )
            unique_prefixes, prefix_inverse, prefix_counts = torch.unique(
                chunk_prefixes, dim=0, return_inverse=True, return_counts=True
            )

            map_ngram_id_to_prefix_id = torch.empty(unique_ngrams.shape[0], dtype=torch.long, device=gpu_device)
            map_ngram_id_to_prefix_id.scatter_(0, ngram_inverse, prefix_inverse)
            corresponding_prefix_counts = prefix_counts[map_ngram_id_to_prefix_id]

            ngram_counts = ngram_counts.float()
            # 联合概率 p(ngram) 必须基于全局的总数进行归一化
            p_ngram = ngram_counts / total_ngrams
            # 条件概率 p(w_n|prefix) 在块内计算，由于排序，这是一个很好的近似
            p_conditional = ngram_counts / corresponding_prefix_counts.float()

            valid_indices = p_conditional > 0
            # 计算当前块对总熵的贡献值
            chunk_entropy_sum = -torch.sum(
                p_ngram[valid_indices] * torch.log2(p_conditional[valid_indices])
            )
            total_entropy_sum += chunk_entropy_sum.item()

            # 清理当前GPU的显存
            del chunk, chunk_prefixes, unique_ngrams, ngram_inverse, ngram_counts, unique_prefixes, prefix_inverse, prefix_counts, map_ngram_id_to_prefix_id, corresponding_prefix_counts
            torch.cuda.empty_cache()

        print(f"[{time.time() - start_time:.2f}s] Step 3: Finished processing all chunks.")
        final_entropy = total_entropy_sum
        # ========================================================
    else:
        # ==================== 单 GPU 计算 (原逻辑) ====================
        print(f"[{time.time() - start_time:.2f}s] Using single-GPU processing.")
        all_prefixes = all_ngrams[:, :-1]  # 直接获取每个 n-gram 的前缀

        # 高效计算 n-gram 的唯一值、ID映射 和 频率
        unique_ngrams, ngram_inverse, ngram_counts = torch.unique(
            all_ngrams, dim=0, return_inverse=True, return_counts=True
        )

        # 同样地，计算前缀的唯一值、ID映射 和 频率
        unique_prefixes, prefix_inverse, prefix_counts = torch.unique(
            all_prefixes, dim=0, return_inverse=True, return_counts=True
        )
        print(f"[{time.time() - start_time:.2f}s] Step 2: Counted {unique_ngrams.shape[0]} unique n-grams and {unique_prefixes.shape[0]} unique prefixes using torch.unique.")

        # 优化的核心步骤: 创建从 unique_ngram ID 到 unique_prefix ID 的映射
        map_ngram_id_to_prefix_id = torch.empty(unique_ngrams.shape[0], dtype=torch.long, device=device)
        map_ngram_id_to_prefix_id.scatter_(0, ngram_inverse, prefix_inverse)
        corresponding_prefix_counts = prefix_counts[map_ngram_id_to_prefix_id]
        print(f"[{time.time() - start_time:.2f}s] Step 3: Built and used prefix count map entirely on GPU.")

        ngram_counts = ngram_counts.float()
        p_ngram = ngram_counts / total_ngrams
        p_conditional = ngram_counts / corresponding_prefix_counts.float()
        valid_indices = p_conditional > 0

        # 计算条件熵
        final_entropy = -torch.sum(
            p_ngram[valid_indices] * torch.log2(p_conditional[valid_indices])
        ).item()
        # ==========================================================

    print(f"[{time.time() - start_time:.2f}s] Step 4: Final entropy calculation complete.")
    return final_entropy


def get_tensor_list(data, tkn: Qwen2Tokenizer):
    lst = []
    total_token_num = 0
    for sample in tqdm(data, desc="tokenize ..."):
        # 对每个文本样本进行分词，不添加特殊token，并返回PyTorch张量
        tensor = tkn(sample['text'], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        lst.append(tensor)
        assert tensor.dim() == 1
        total_token_num += tensor.numel()
    return lst


def main():
    # 数据集路径，请根据您的环境修改
    wiki_path = "../mock/datasets/wikitext-en-de"
    # 模型路径，请根据您的环境修改
    model_path = "../models/Qwen2.5-1.5B-AddTokens"
    wiki_zh_path = '../mock/datasets/wiki_zh_2'

    # wiki_en = load_dataset(wiki_path, "featured_en", split='train')
    # wiki_de = load_dataset(wiki_path, "exzellent_de", split='train')
    # wiki_zh = load_dataset(wiki_zh_path, split='train')
    img_data = load_dataset("../new_data/ugen-pro/pretrain/converted_data/shijuezhongguo_gaozhiliang", split='train', streaming=True)

    # tkn = AutoTokenizer.from_pretrained(model_path)

    # 下面 en, de 已经计算完毕
    # en_token_list = get_tensor_list(wiki_en, tkn)
    # for n in trange(1, 10, desc="English Entropy"):
    #     entropy = calculate_ngram_conditional_entropy(en_token_list, n)
    #     print(f"English Wikipedia, n={n} entropy = {entropy:.4f} bits")
    #
    # de_token_list = get_tensor_list(wiki_de, tkn)
    # for n in trange(1, 10, desc="German Entropy"):
    #     entropy = calculate_ngram_conditional_entropy(de_token_list, n)
    #     print(f"German Wikipedia, n={n} entropy = {entropy:.4f} bits")
    #
    # de_token_list = get_tensor_list(wiki_zh, tkn)
    # for n in trange(1, 10, desc="Chinese Entropy"):
    #     entropy = calculate_ngram_conditional_entropy(de_token_list, n)
    #     print(f"Chinese Wikipedia, n={n} entropy = {entropy:.4f} bits")

    max_limit = 10_000_000
    lst = []
    for sample in tqdm(img_data, desc='collect img vqcode'):
        if sample['vqcode_512'] is None:
            continue

        vqcode = json.loads(sample['vqcode_512'])
        tensor = torch.tensor(vqcode)
        assert tensor.dim() == 1
        lst.append(tensor)
        if len(lst) >= max_limit:
            break

    for n in range(3, 5):
        entropy = calculate_ngram_conditional_entropy_optimized(lst, n)
        print(f"Image, n={n} entropy = {entropy:.4f} bits")


if __name__ == "__main__":
    main()