## 📂 Project Structure

```
.
├── configs/                # Training args, model configs, distributed configs (YAML), and conversation templates
├── data_process/           # Data preprocessing scripts, especially for VQGAN image encoding
├── draw_pics/              # Scripts to generate plots from analysis results
├── evaluation/             # Automated evaluation pipeline
│   ├── T2I_Eval/           # Text-to-Image evaluation module
│   ├── api_server.py       # API server for VQA evaluation
│   ├── eval_template.py    # Main entry point for evaluation tasks
│   └── eval_vqa.py         # VQA evaluation client
├── uni_arch/                 # Core training logic
│   ├── train/              # Trainer, data collator, and main training script
│   └── ...
├── modeling/               # Model architecture definitions (Uni-X, MoE, MoT, etc.)
├── tools/                  # Various utilities (gradient analysis, data translation, logging, etc.)
└── ...
```

## ⚙️ Setup and Installation

1.  Clone this repository:

    ```bash
    git clone ...
    cd ...
    ```

2.  Install the required dependencies:

    ```bash
    conda create -n uni python=3.10 -y
    conda activate uni

    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --resume-retries 10
    pip install torch==2.5.1 transformers[torch]==4.53.3 accelerate deepspeed==0.15.4 torchvision datasets==3.6.0
    pip install transformers==4.53.3 fire matplotlib seaborn wandb loguru
    MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
    ```

## 📦 Data Preparation

The model requires image data to be preprocessed into VQGAN tokens.

1.  **Image Encoding**: Use scripts like `data_process/encode_vq_finevision.py` or `data_process/convert_imagepair_cc512_batch_version.py` to convert your image-text pair data into VQGAN encodings. These scripts will transform images into discrete token sequences and pair them with their corresponding text.

2.  **Data Format**: Ensure the final training data is in `jsonl` format, where each line contains text, VQGAN codes, and other metadata. For details on the format, please refer to the logic in `uni_arch/train/data_collator.py`.

## 🚀 Model Training

The training process is managed through a centralized shell script. You only need to configure the parameters at the top of the script.

Below is an example script for multi-node, multi-GPU (7 machines, 8 GPUs each) SFT using DeepSpeed:

```bash
#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
# =================================================================
# 1. Parameter Configuration
# =================================================================

# -- Task & Logging --
run_name='Qwen2.5-3B-uni-X'
output_dir="../ckpts/${run_name}"
extra_tags="3B,unix,x12_6,try_sota,big-SFT,ignore_ins,sft_v4_more" # For W&B and path naming

# -- Distributed Training Config --
main_port=16374
main_ip='10.54.107.215'         # IP address of the main node
hostfile='./host_file7'         # DeepSpeed hostfile
config_file="configs/accel_ds_7machine.yaml" # Accelerate config file

# -- Model & Data Paths --
model_path="../mock/ckpts/Qwen2.5-3B-uni-X/..." # Path to a pretrained model or checkpoint
data_path="../datasets/uni_sft_v4"             # Path to the training data
streaming_data=0                               # Whether to stream data
data_percentage="1.0"                          # Percentage of data to use
t2i_ratio=0.5                                  # Ratio for constructing T2I/I2T data
shuffle_seed=218                               # Seed for dataset shuffling
vq_resolution=512                              # VQGAN resolution

# -- Model Architecture --
model_version="gemma"
custom_cls="uni_qwen"             # Use custom model class
model_spec_module="x"             # Specify the Uni-X architecture
vision_encode_layers=12           # Number of vision encoder layers
vision_decode_layers=6            # Number of vision decoder layers
all_modal_visible=0
unfreeze_keys="train-all"         # Train all parameters
ffn_vision_size=4096              # Vision FFN size
ffn_share_size=4096               # Shared FFN size

# -- Training Hyperparameters --
bf16="true"
learning_rate=1e-5
max_steps=10000
train_batch_size=20
model_max_length=20480
use_data_packing=2                # 0:No packing, 1:Pretrain packing, 2:SFT packing
grad_accum_steps=1
weight_decay=0.0
warmup_ratio=0.1
lr_scheduler="linear"
ignore_instruction=1              # Whether to ignore the instruction part when calculating loss

# -- Saving & Evaluation --
save_steps=0.05                   # Save checkpoint every 5% of total steps
save_total_limit=1                # Maximum number of checkpoints to keep
eval_strategy="no"
logging_steps=10

# -- Performance & Others --
gradient_checkpointing=1
dataloader_workers=16
resume_from_checkpoint=0          # Whether to resume from a checkpoint

# =================================================================
# 2. Execute Command (Usually no changes needed below)
# =================================================================
echo "--- Starting Training: ${run_name} ---"

nohup accelerate launch --main_process_port ${main_port} --main_process_ip "${main_ip}" \
--deepspeed_hostfile "${hostfile}" --config_file "${config_file}" \
uni_arch/train/hf_trainer.py \
--model_name_or_path "${model_path}" \
--data_path "${data_path}" \
--percentage "${data_percentage}" \
# ... (all other parameters are passed here)
> train.log 2>&1 &

echo "--- Training launched in background. Check train.log for output. ---"
```

**How to Use:**

1.  Copy the template above into a new `train.sh` file.
2.  Modify the parameters in the **Parameter Configuration** section to fit your needs.
3.  Execute the script: `bash train.sh`.

## 📊 Model Evaluation

This framework provides a powerful, template-driven evaluation pipeline that can run multiple types of evaluations with a single command.

**1. Configure Experiments**

Define the list of evaluations you want to run in the `evaluation/exp.py` file. Each experiment is a dictionary specifying the model path and the evaluation types.

  * **`eval_type`**: A list that specifies the evaluation tasks. Possible values include:
      * `"text"`: Text capability evaluation (MMLU, ARC, etc.).
      * `"vis_und"`: Visual understanding evaluation (MME, POPE, MMBench, etc.).
      * `"dpg_bench"`: DPG-Bench text-to-image evaluation.
      * `"geneval"`: GenEval text-to-image evaluation.

**Example `evaluation/exp.py`:**

```python
EXPERIMENTS = [
    {
        "name": "sft_v4-ckpt5k",
        "model_path": "../mock/ckpts/.../checkpoint-5000",
        
        # --- Text-to-Image Eval Config ---
        "dpg_bench_prompts_path": "evaluation/T2I_Eval/dpg_bench/dpg_prompts_zh_fixed.jsonl",
        "geneval_prompts_path": "evaluation/T2I_Eval/geneval/geneval_prompts_zh.txt",
        "cfg": 2.0,

        # --- Visual Understanding Eval Config ---
        "vis_und_server_gpus": 8,
        "vis_und_max_batch_size": 40,
        "vis_und_api_nproc": 1600,
        "vis_und_max_tokens": 10,
        
        # --- Specify Evaluation Tasks to Run ---
        # You can combine multiple tasks or run just one.
        "eval_type": ["geneval", "dpg_bench", "vis_und", "text"],
    },
]
```

**2. Run Evaluation**

After configuring `exp.py`, run the following command from the project root to start the automated evaluation:

```bash
# Set PYTHONPATH to ensure project modules can be found
export PYTHONPATH=. 

# Launch the evaluation
python evaluation/eval_template.py
```

The script will automatically parse the configuration in `exp.py` and execute each experiment in sequence:

  * For **Text-to-Image** tasks, it will first generate all images in parallel and then invoke the corresponding evaluation scripts to compute scores.
  * For **Visual Understanding** tasks, it will automatically start a multi-GPU API server in the background, run the VQA client for evaluation, and shut down the server upon completion.
  * For **Text** tasks, it will invoke `lm-eval-harness` for evaluation.

All logs and results will be saved in the `outputs/` directory.

## 💡 Inference

  * **API Service**: You can run `evaluation/api_server.py` independently to deploy a persistent, OpenAI-compatible API endpoint for easy integration with other applications.
  * **Script-based Inference**: The file `evaluation/uni_infer.py` contains the core text and image generation logic (the `any_modal_chat_api` function) and can be used as a reference for writing custom inference scripts.

## 🛠️ Tools & Analysis

The project includes several useful tools for analysis:

  * `tools/analyze_model_grad.py`: Calculates and visualizes the cosine similarity of gradients between different modalities (text-only vs. multi-modal) during training to analyze the **gradient conflict** problem.
  * `draw_pics/grad_conflict.py`: Plots the results from the gradient analysis script.
  * `tools/cal_entropy.py`: Computes the N-gram conditional entropy for different data sources (e.g., English Wikipedia, Chinese Wikipedia, image tokens) to measure data complexity and predictability.
  * `tools/data_translator.py`: A utility to batch-translate datasets using an API.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.