import logging
import deepspeed
import torch
import os
import time

from torch import nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_last_checkpoint
from transformers.integrations.deepspeed import deepspeed_config
from tools.log import main_logger


class CustomTrainer(Trainer):

    def _save_checkpoint(self, model, trial):
        """
        重写 _save_checkpoint 方法，增加对不稳定存储的重试逻辑。
        """
        max_retries = 5
        wait_seconds = 5  # 每次重试前等待的时间

        for attempt in range(1, max_retries + 1):
            try:
                # 直接调用父类的方法来执行实际的保存操作。
                # 父类已经处理了所有细节，包括 deepspeed 的保存、清理旧 checkpoints 等。
                # 这种方式最简单且不易出错。
                # 在非NFS情况下，理论上不应该报错，应该一次性保存成功。
                return super()._save_checkpoint(model, trial)
            except Exception as e:
                main_logger.warning(
                    f"保存 Checkpoint 失败，尝试次数: {attempt}/{max_retries}。"
                    f"错误: {e}。将在 {wait_seconds} 秒后重试..."
                )
                if attempt == max_retries:
                    main_logger.error(f"已达到最大重试次数，保存 Checkpoint 彻底失败。")
                    raise  # 在最后一次尝试失败后，重新抛出异常

                time.sleep(wait_seconds)