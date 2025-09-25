from dataclasses import dataclass, field
from typing import Optional, Union

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="gemma")
    image_processor: Optional[str] = field(default=None)
    model_custom_cls: str = "auto"
    model_spec_module: str = "x"
    vision_encode_layers: int = 0
    vision_decode_layers: int = 0
    all_modal_visible: int = 0
    add_share_ffn: int = 0
    ffn_vision_size: int = 2048
    ffn_share_size: int = 2048
    attn_v_q_heads: int = 4
    use_share_ffn: int = 1
    use_share_attn: int = 1
    decoder_sparse_step: int = 2
    skip_connection_loc: int = -1


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    vq_resolution: str = '512',
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    cfg_ratio: Optional[float] = 0.9,
    percentage: Optional[str] = field(default='1.0', metadata={"help": "how many data to use"})
    T2I_ratio: Optional[float] = field(default=0.5, metadata={"help": "the ratio to construct T2I or I2T pair"})
    shuffleseed: Optional[int] = field(default=42)
    use_data_packing: int = 1  # 0 raw 1 pretrain 2 sft
    packing_bug_fix: int = 1
    ignore_instruction: int = 0
    add_sep_for_vis: int = 0
    vis_sep_tokens: str = "<|vision_pad|>"
    vis_sep_lens: str = "8"
    streaming_data: int = 0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    # freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    group_by_modality_length: bool = field(default=False)
    # lr_multi: Optional[str] = field(default=None)
    vision_lr: Optional[float] = None
    label_smoothing_factor: float = 0.0
    project_name: str = "UniArch-Pilot"
    run_name: str = "default"
    extra_tags: str = None
    unfreeze_keys: str = "train-all"
    init_vision_with_text: int = 0

    for_analysis_samples_num: int = 1000
