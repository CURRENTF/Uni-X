import copy
import os
from dataclasses import asdict
from datetime import datetime
import wandb
from typing import Dict, Tuple, List
import random
import glob
import numpy as np
from collections import defaultdict
import csv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import transformers

from configs.args import ModelArguments, DataArguments, TrainingArguments
from uni_arch.train.train import make_supervised_data_module, load_model_and_tokenizer
from accelerate import Accelerator
from tqdm import *

local_rank = None
accelerator = Accelerator()


def collect_gradients(
        model: torch.nn.Module,
        data_subset: torch.utils.data.Dataset,
        collator,
        num_batches: int,
        batch_size: int,
        device: str
) -> Dict[str, torch.Tensor]:
    """
    为指定的层（FFN 和 Attention 权重）收集和累积梯度。
    """
    # 确定我们感兴趣的目标参数
    target_param_substrings = [
        "mlp.up_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    ]
    target_params = {
        name: p for name, p in model.named_parameters()
        if any(sub in name for sub in target_param_substrings) and p.requires_grad
    }

    # 初始化一个字典来存储累积的梯度
    accumulated_grads = {name: torch.zeros_like(p.data) for name, p in target_params.items()}

    model.train()  # 确保模型处于训练模式

    for i in trange(num_batches):
        subset = data_subset[i * batch_size: (i + 1) * batch_size]
        data_dict = collator(subset)

        # 将数据移动到正确的设备
        for k, v in data_dict.items():
            if hasattr(v, 'to'):
                data_dict[k] = v.to(device)

        # 前向传播计算损失
        outputs = model(**data_dict)
        loss = outputs.loss

        # 反向传播计算梯度
        loss.backward()

        # 累积目标参数的梯度
        with torch.no_grad():
            for name, p in target_params.items():
                if p.grad is not None:
                    # 平均梯度
                    accumulated_grads[name] += p.grad / num_batches

        # 清除梯度，为下一次迭代做准备
        model.zero_grad()

    return accumulated_grads


def analyze(model_args, data_args, training_args, attn_implementation="flash_attention_2") -> Dict[str, Tuple[List[int], List[float]]]:
    # Prepare run name and output directory
    print(training_args.extra_tags, type(training_args.extra_tags))
    if isinstance(training_args.extra_tags, str):
        training_args.extra_tags = training_args.extra_tags.split(',')

    assert isinstance(training_args.extra_tags, list)
    tags_str = None
    if training_args.extra_tags is not None:
        tags_str = "-".join(training_args.extra_tags)
        training_args.run_name = f"{training_args.run_name}__{tags_str}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        # Note: output_dir is now handled in main for per-checkpoint saving
        # training_args.output_dir = os.path.join(training_args.output_dir, tags_str, timestamp)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args, attn_implementation)

    # Initialize vision weights if specified
    if training_args.init_vision_with_text and hasattr(model.model, "init_vision_weights"):
        model.model.init_vision_weights()

    # Freeze layers if specified
    if training_args.unfreeze_keys != "train-all":
        freeze_tag_exist = any("freeze" in t for t in training_args.extra_tags)
        assert freeze_tag_exist, "Freeze tag is required when unfreezing specific keys."

        unfreeze_keys = set(training_args.unfreeze_keys.split(','))
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n for k in unfreeze_keys)

    # Disable cache for training
    model.config.use_cache = False

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Prepare data module
    data_modules = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_data = data_modules['train_dataset']
    data_collator = data_modules['data_collator']

    train_subset = []
    for i, s in enumerate(train_data):
        if i == training_args.for_analysis_samples_num:
            break
        train_subset.append(s)
    # train_subset = train_data[:training_args.for_analysis_samples_num]
    bs = training_args.per_device_train_batch_size
    n = len(train_subset) // bs
    device = 'cuda'

    # 为纯文本数据收集梯度
    print("Collecting gradients for text-only data...")
    data_collator.set_filter_data_type([None, 'text'])
    text_grads = collect_gradients(model, train_subset, data_collator, n - 1, bs, device)

    # 为多模态数据收集梯度
    print("Collecting gradients for multimodal data...")
    data_collator.set_filter_data_type(['image_text', 't2i', 'i2t'])
    multimodal_grads = collect_gradients(model, train_subset, data_collator, n - 1, bs, device)

    # 为基线计算准备数据和梯度
    print("Collecting gradients for baseline calculation...")
    # 使用所有模态的数据
    data_collator.set_filter_data_type(None)
    # 随机打乱并拆分数据集
    shuffled_subset = random.sample(train_subset, len(train_subset))
    mid_idx = len(shuffled_subset) // 2
    subset_a, subset_b = shuffled_subset[:mid_idx], shuffled_subset[mid_idx:]
    # 为两个随机子集分别收集梯度
    grads_a = collect_gradients(model, subset_a, data_collator, len(subset_a) // bs, bs, device)
    grads_b = collect_gradients(model, subset_b, data_collator, len(subset_b) // bs, bs, device)

    # 按参数类型分组，为每组参数单独计算和绘图
    param_groups = {
        "up_proj": "mlp.up_proj.weight",
        "gate_proj": "mlp.gate_proj.weight",
        "down_proj": "mlp.down_proj.weight",
        "q_proj": "self_attn.q_proj.weight",
        "k_proj": "self_attn.k_proj.weight",
        "v_proj": "self_attn.v_proj.weight",
        "o_proj": "self_attn.o_proj.weight",
    }

    analysis_results = {}  # Initialize dict to store results

    for group_name, substring in param_groups.items():
        print(f"\n--- Processing parameter group: {group_name} ---")

        # 1. 筛选出当前组的梯度
        text_grads_group = {k: v for k, v in text_grads.items() if substring in k}

        # 如果模型中没有该类型的参数，则跳过
        if not text_grads_group:
            print(f"No parameters found for group: {group_name}. Skipping.")
            continue

        multimodal_grads_group = {k: v for k, v in multimodal_grads.items() if substring in k}
        grads_a_group = {k: v for k, v in grads_a.items() if substring in k}
        grads_b_group = {k: v for k, v in grads_b.items() if substring in k}

        # 按层号对参数名进行排序
        sorted_layer_names = sorted(text_grads_group.keys(), key=lambda x: int(x.split('.')[2]))
        # 当为 uni-x 时，过滤掉非共享layers
        total_layers = model.config.num_hidden_layers
        vis_encode_layers, vis_decode_layers = getattr(model_args, 'vision_encode_layers', 0), getattr(model_args, 'vision_decode_layers', 0)
        sorted_layer_names = [x for x in sorted_layer_names if vis_encode_layers <= int(x.split('.')[2]) < total_layers - vis_decode_layers]
        assert len(sorted_layer_names) == total_layers - vis_encode_layers - vis_decode_layers, f"{total_layers}, {vis_encode_layers}, {vis_decode_layers}"

        # 2. 计算基线相似度（不同数据子集之间的梯度相似度）
        # 这个即使是 uni-X，也一定每套参数都有梯度，所以不需要特殊处理
        baseline_similarities = {}
        for name in sorted_layer_names:
            grad_a_flat = grads_a_group[name].flatten()
            grad_b_flat = grads_b_group[name].flatten()
            sim = F.cosine_similarity(grad_a_flat, grad_b_flat, dim=0).item()
            baseline_similarities[name] = sim

        # 3. 计算文本 vs. 多模态的调整后相似度
        similarities = []
        layer_indices = []
        for name in sorted_layer_names:
            grad_text = text_grads_group[name]  # 这俩如果对应参数没梯度，就是0tensor
            grad_multimodal = multimodal_grads_group[name]

            # 当处理纯文本数据时，vision-only 的层没有梯度，其累积梯度为0.
            # 经过计算时，也会自动出现相似度为0
            grad_text_flat = grad_text.flatten()
            grad_multimodal_flat = grad_multimodal.flatten()
            similarity_raw = F.cosine_similarity(grad_text_flat, grad_multimodal_flat, dim=0, eps=1e-6).item()

            # 减去基线，得到调整后相似度，用于衡量模态冲突
            baseline_sim = baseline_similarities[name]
            adjusted_similarity = similarity_raw - baseline_sim
            similarities.append(adjusted_similarity)

            # 提取层索引用于绘图
            layer_idx = int(name.split('.')[2])
            layer_indices.append(layer_idx)
            print(f"Layer {layer_idx} ({group_name}): Raw Sim={similarity_raw:.4f}, Baseline={baseline_sim:.4f}, Adjusted={adjusted_similarity:.4f}")

        # 4. 绘制并保存结果
        print(f"Plotting results for {group_name}...")
        plt.figure(figsize=(12, 7))
        plt.plot(layer_indices, similarities, marker='o', linestyle='-')

        # 格式化组名用于图表标题
        title = f'Gradient Conflict (Text vs. Multimodal) for {group_name.replace("_", " ").title()} Layers'
        plt.title(title)
        plt.xlabel('Layer Index')
        plt.ylabel('Adjusted Cosine Similarity (Raw - Baseline)')
        plt.grid(True)
        plt.xticks(sorted(list(set(layer_indices))))
        plt.tight_layout()

        # 确保输出目录存在并保存图像
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_path = os.path.join(training_args.output_dir, f"gradient_similarity_{group_name}.pdf")
        plt.savefig(save_path)
        plt.close()  # 释放内存
        print(f"Plot saved to {save_path}")

        # Store results for this group
        analysis_results[group_name] = (layer_indices, similarities)

    return analysis_results


def main():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Base paths for models and outputs
    base_model_path = model_args.model_name_or_path
    base_output_dir = training_args.output_dir

    # Find all checkpoint folders that match the pattern "checkpoint-*"
    if not os.path.isdir(base_model_path):
        raise ValueError(f"model_name_or_path '{base_model_path}' must be a directory.")

    checkpoint_paths = glob.glob(os.path.join(base_model_path, "checkpoint-*"))
    checkpoint_paths = [p for p in checkpoint_paths if os.path.isdir(p)]

    # If no checkpoints are found, analyze the base model path itself
    if not checkpoint_paths:
        print(f"No checkpoint folders found in {base_model_path}. Analyzing the base path directly.")
        checkpoint_paths = [base_model_path]

    all_results = {}  # {ckpt_name: {group_name: (indices, similarities)}}
    summary_lines = []  # For the final txt summary

    # 1. Loop through each checkpoint and run analysis
    for ckpt_path in sorted(checkpoint_paths):
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\n{'=' * 20} Analyzing checkpoint: {ckpt_name} {'=' * 20}")

        _model_args, _data_args, _training_args = copy.deepcopy((model_args, data_args, training_args))

        # Update paths for the current checkpoint
        _model_args.model_name_or_path = ckpt_path
        # Save per-checkpoint plots in a dedicated subfolder
        _training_args.output_dir = os.path.join(base_output_dir, ckpt_name)

        results = analyze(_model_args, _data_args, _training_args)
        all_results[ckpt_name] = results
        # Add results to summary
        summary_lines.append(f"--- Results for {ckpt_name} ---")
        for group, (indices, sims) in results.items():
            summary_lines.append(f"  {group}: Average Adjusted Similarity = {np.mean(sims):.4f}")
        summary_lines.append("")

    if not all_results:
        print("No analysis could be completed. Exiting.")
        return

    # --- Aggregation and Plotting ---
    os.makedirs(base_output_dir, exist_ok=True)

    # 保存一份最完整的记录，保存为csv格式，columns 有 checkpoint_id, group name, layer_idx, similarities
    # --- Start: Save full results to CSV ---
    csv_path = os.path.join(base_output_dir, "baseline_full_analysis_results.csv")
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            # 定义CSV文件的列名
            fieldnames = ['checkpoint_id', 'group_name', 'layer_idx', 'similarity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 遍历所有结果并逐行写入
            for ckpt_name, results in all_results.items():
                for group_name, (layer_indices, similarities) in results.items():
                    for layer_idx, similarity in zip(layer_indices, similarities):
                        writer.writerow({
                            'checkpoint_id': ckpt_name,
                            'group_name': group_name,
                            'layer_idx': layer_idx,
                            'similarity': similarity
                        })
        print(f"Full analysis results saved to {csv_path}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")
    # --- End: Save full results to CSV ---

    sample_ckpt = next(iter(all_results))
    param_groups = list(all_results[sample_ckpt].keys())
    layer_indices = all_results[sample_ckpt][param_groups[0]][0] if param_groups else []

    # 2. Average by checkpoint ID (one curve per param group, averaged over all checkpoints)
    print("\n--- Aggregating results by parameter group (averaging over checkpoints) ---")
    summary_lines.append("--- Aggregation: Averaged over Checkpoints ---\n")
    plt.figure(figsize=(15, 10))
    avg_results_by_group = defaultdict(list)
    for ckpt_name, results in all_results.items():
        for group_name, (indices, sims) in results.items():
            assert len(sims) == len(layer_indices)
            avg_results_by_group[group_name].append(sims)

    for group_name, sim_lists in avg_results_by_group.items():
        if sim_lists:
            avg_sims = np.mean(sim_lists, axis=0)
            plt.plot(layer_indices, avg_sims, marker='o', linestyle='-', label=group_name)
            summary_lines.append(f"  {group_name}: {list(np.round(avg_sims, 4))}")

    plt.title('Average Gradient Conflict Across All Checkpoints')
    plt.xlabel('Layer Index')
    plt.ylabel('Adjusted Cosine Similarity (Raw - Baseline)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(sorted(list(set(layer_indices))))
    save_path = os.path.join(base_output_dir, "summary_avg_by_param_group.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved average-by-group plot to {save_path}")

    # 3. Average by parameter group (one curve per checkpoint, averaged over all groups)
    print("\n--- Aggregating results by checkpoint (averaging over parameter groups) ---")
    summary_lines.append("\n--- Aggregation: Averaged over Parameter Groups ---\n")
    plt.figure(figsize=(15, 10))
    for ckpt_name, results in all_results.items():
        sims_for_ckpt = [sims for _, (indices, sims) in results.items() if len(sims) == len(layer_indices)]
        if sims_for_ckpt:
            avg_sims = np.mean(sims_for_ckpt, axis=0)
            plt.plot(layer_indices, avg_sims, marker='o', linestyle='-', label=ckpt_name)
            summary_lines.append(f"  {ckpt_name}: {list(np.round(avg_sims, 4))}")

    plt.title('Average Gradient Conflict for Each Checkpoint')
    plt.xlabel('Layer Index')
    plt.ylabel('Adjusted Cosine Similarity (Raw - Baseline)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(sorted(list(set(layer_indices))))
    save_path = os.path.join(base_output_dir, "summary_avg_by_checkpoint.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved average-by-checkpoint plot to {save_path}")

    # 4. Grand average (one curve, averaged over all checkpoints and groups)
    print("\n--- Calculating grand average over all results ---")
    summary_lines.append("\n--- Aggregation: Grand Average ---\n")
    all_sims = []
    for ckpt_name, results in all_results.items():
        all_sims.extend([sims for _, (indices, sims) in results.items() if len(sims) == len(layer_indices)])

    if all_sims:
        grand_avg_sims = np.mean(all_sims, axis=0)
        plt.figure(figsize=(12, 7))
        plt.plot(layer_indices, grand_avg_sims, marker='o', linestyle='-')
        plt.title('Grand Average Gradient Conflict (All Checkpoints & Groups)')
        plt.xlabel('Layer Index')
        plt.ylabel('Adjusted Cosine Similarity (Raw - Baseline)')
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(sorted(list(set(layer_indices))))
        save_path = os.path.join(base_output_dir, "summary_grand_average.pdf")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved grand average plot to {save_path}")
        summary_lines.append(f"  Grand Average: {list(np.round(grand_avg_sims, 4))}")

    # 5. Save summary text file
    summary_path = os.path.join(base_output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"Analysis summary saved to {summary_path}")


if __name__ == "__main__":
    main()