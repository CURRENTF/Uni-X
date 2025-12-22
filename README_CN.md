## 📂 项目结构

```
.
├── configs/                # 训练参数、模型配置、分布式配置 (YAML) 和对话模板
├── data_process/           # 数据预处理脚本，特别是 VQGAN 图像编码
├── draw_pics/              # 用于从分析结果生成图表的脚本
├── evaluation/             # 自动化评估流水线
│   ├── T2I_Eval/           # 文生图 (Text-to-Image) 评估模块
│   ├── api_server.py       # VQA 评估用的 API 服务器
│   ├── eval_template.py    # 评估任务的主入口
│   └── eval_vqa.py         # VQA 评估客户端
├── uni_arch/                 # 核心训练逻辑
│   ├── train/              # 训练器、数据加载器和训练主脚本
│   └── ...
├── modeling/               # 模型架构定义 (Uni-X, MoE, MoT 等)
├── tools/                  # 各种辅助工具 (梯度分析、数据翻译、日志等)
└── ...
```

## ⚙️ 环境设置

1.  克隆本仓库：

    ```bash
    git clone ...
    cd ...
    ```

2.  安装所需的依赖包：

    ```bash
    conda create -n uni python=3.10 -y
    conda activate uni

    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --resume-retries 10
    pip install torch==2.5.1 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==3.6.0
    pip install transformers==4.53.3 fire matplotlib seaborn wandb loguru
    MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
    ```

## 📦 数据准备

模型的训练需要将图像数据预处理为 VQGAN Token。

1.  **图像编码**: 使用 `data_process/encode_vq_finevision.py` 或 `data_process/convert_imagepair_cc512_batch_version.py` 脚本将您的图文对数据转换为 VQGAN 编码。这些脚本会将图像转换为离散的Token序列并与文本配对。

2.  **数据格式**: 确保最终的训练数据每行包含文本、VQGAN编码以及其他元数据。具体格式请参考 `uni_arch/train/data_collator.py` 中的处理逻辑。

## 🚀 模型训练

训练过程通过一个集中的 Shell 脚本进行管理，您只需在脚本顶部配置参数即可。

下面是一个多机多卡（7台机器，每台8卡）使用 DeepSpeed 进行SFT的训练示例脚本：

```bash
#!/bin/bash
# 脚本将在遇到任何错误时立即退出
set -e
# =================================================================
# 1. 参数配置 (Parameter Configuration)
# =================================================================

# -- 任务与日志 --
run_name='Qwen2.5-3B-uni-X'
output_dir="../ckpts/${run_name}"
extra_tags="3B,unix,x12_6,try_sota,big-SFT,ignore_ins,sft_v4_more" # 用于W&B分类和路径命名

# -- 分布式训练配置 --
main_port=16374
main_ip='10.54.107.215'         # 主节点IP
hostfile='./host_file7'         # DeepSpeed hostfile
config_file="configs/accel_ds_7machine.yaml" # Accelerate 配置文件

# -- 模型与数据路径 --
model_path="../mock/ckpts/Qwen2.5-3B-uni-X/..." # 预训练模型或断点路径
data_path="../datasets/uni_sft_v4"             # 训练数据路径
streaming_data=0                               # 是否使用流式加载数据
data_percentage="1.0"                          # 使用数据的百分比
t2i_ratio=0.5                                  # 文生图/图生文数据的构造比例
shuffle_seed=218                               # 数据集打乱种子
vq_resolution=512                              # VQGAN 分辨率

# -- 模型结构 --
model_version="gemma"
custom_cls="uni_qwen"             # 使用自定义模型类
model_spec_module="x"             # 指定为 Uni-X 架构
vision_encode_layers=12           # 视觉编码器层数
vision_decode_layers=6            # 视觉解码器层数
all_modal_visible=0
unfreeze_keys="train-all"         # 训练所有参数
ffn_vision_size=0              # 视觉FFN大小
ffn_share_size=0               # 共享FFN大小

# -- 训练超参数 --
bf16="true"
learning_rate=1e-5
max_steps=10000
train_batch_size=20
model_max_length=20480
use_data_packing=2                # 0:不打包, 1:预训练打包, 2:SFT打包
grad_accum_steps=1
weight_decay=0.0
warmup_ratio=0.1
lr_scheduler="linear"
ignore_instruction=1              # 是否在计算loss时忽略指令部分

# -- 保存与评估 --
save_steps=0.05                   # 按训练步数比例保存 (0.05 = 每5%保存一次)
save_total_limit=1                # 最多保存的checkpoint数量
eval_strategy="no"
logging_steps=10

# -- 性能与其他 --
gradient_checkpointing=1
dataloader_workers=16
resume_from_checkpoint=0          # 是否从断点恢复

# =================================================================
# 2. 执行命令 (通常无需修改)
# =================================================================
echo "--- Starting Training: ${run_name} ---"

nohup accelerate launch --main_process_port ${main_port} --main_process_ip "${main_ip}" \
--deepspeed_hostfile "${hostfile}" --config_file "${config_file}" \
uni_arch/train/hf_trainer.py \
--model_name_or_path "${model_path}" \
--data_path "${data_path}" \
--percentage "${data_percentage}" \
# ... (此处省略了所有参数以保持简洁, 实际脚本中会完整列出)
> train.log 2>&1 &

echo "--- Training launched in background. Check train.log for output. ---"
```

**如何使用:**

1.  复制上面的模板到一个新的 `train.sh` 文件。
2.  根据您的需求修改 **参数配置** 部分的内容。
3.  执行脚本: `bash train.sh`。

## 📊 模型评估

本框架提供了一个强大的模板化评估流水线，可以一键执行多种类型的评估。

**1. 配置实验**

在 `evaluation/exp.py` 文件中定义您要运行的评估实验列表。每个实验是一个字典，指定了模型路径和要执行的评估类型。

  * **`eval_type`**: 一个列表，用于指定评估任务，可选值包括：
      * `"text"`: 文本能力评估 (MMLU, ARC 等)。
      * `"vis_und"`: 视觉理解评估 (MME, POPE, MMBench 等)。
      * `"dpg_bench"`: DPG-Bench 文生图评估。
      * `"geneval"`: GenEval 文生图评估。

**示例 `evaluation/exp.py`:**

```python
EXPERIMENTS = [
    {
        "name": "sft_v4-ckpt5k",
        "model_path": "../mock/ckpts/.../checkpoint-5000",
        
        # --- 文生图评估配置 ---
        "dpg_bench_prompts_path": "evaluation/T2I_Eval/dpg_bench/dpg_prompts_zh_fixed.jsonl",
        "geneval_prompts_path": "evaluation/T2I_Eval/geneval/geneval_prompts_zh.txt",
        "cfg": 2.0,

        # --- 视觉理解评估配置 ---
        "vis_und_server_gpus": 8,
        "vis_und_max_batch_size": 40,
        "vis_und_api_nproc": 1600,
        "vis_und_max_tokens": 10,
        
        # --- 指定要执行的评估任务 ---
        # 您可以组合多个任务，或只执行一个
        "eval_type": ["geneval", "dpg_bench", "vis_und", "text"],
    },
]
```

**2. 运行评估**

配置好 `exp.py`后，在项目根目录运行以下命令即可启动全自动评估：

```bash
# 设置 PYTHONPATH 以确保能找到项目模块
export PYTHONPATH=. 

# 启动评估
python evaluation/eval_template.py
```

脚本会自动解析 `exp.py` 中的配置，依次执行每个实验：

  * 对于 **文生图** 任务，它会先并行生成所有图片，然后调用相应的评估脚本计算分数。
  * 对于 **视觉理解** 任务，它会自动在后台启动一个多GPU的API服务器，然后运行VQA客户端进行评估，并在结束后自动关闭服务器。
  * 对于 **文本** 任务，它会调用 `lm-eval-harness` 进行评估。

所有日志和结果都将保存在 `outputs/` 目录下。

## 💡 推理

  * **API 服务**: 您可以独立运行 `evaluation/api_server.py` 来部署一个与 OpenAI API 兼容的推理服务，方便与其他应用集成。
  * **脚本推理**: `evaluation/uni_infer.py` 包含了核心的图文生成逻辑 (`any_modal_chat_api` 函数)，可以作为编写自定义推理脚本的参考。

## 🛠️ 工具与分析

项目中包含多种有用的分析工具：

  * `tools/analyze_model_grad.py`: 计算并可视化不同模态（纯文本 vs. 多模态）数据在训练过程中的梯度余弦相似度，用于分析**梯度冲突**问题。
  * `draw_pics/grad_conflict.py`: 将梯度分析的结果绘制成图表。
  * `tools/cal_entropy.py`: 计算不同数据源（如英文维基、中文维基、图像Token）的N-gram条件信息熵，以衡量数据的复杂性和可预测性。
  * `tools/data_translator.py`: 调用API对数据集进行批量翻译。

## 📄 许可证

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。