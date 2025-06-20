from transformers import HfArgumentParser
import datetime
from typing import List
from dataclasses import field
from dataclasses import dataclass
import os
from typing import Union
from transformers import EarlyStoppingCallback
import sys

from pathlib import Path

'''
PROJECT_ROOT = Path(r"D:\projects\gitlab\easy_bge")
sys.path.insert(0, str(PROJECT_ROOT))  # 保证优先从项目根目录导入

# 将 internal_Flagembedding 伪装成 FlagEmbedding
try:
    from internal_flagembedding import FlagEmbedding
    sys.modules["FlagEmbedding"] = FlagEmbedding  # 覆盖模块引用
except ImportError as e:
    raise RuntimeError("项目目录结构错误，请检查 internal_Flagembedding 是否存在") from e

# 验证路径（可选）
print(f"FlagEmbedding模块路径: {FlagEmbedding.__file__}")

'''
# 禁用所有分布式相关环境变量
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from FlagEmbedding.finetune.embedder.encoder_only.base import (
    EncoderOnlyEmbedderDataArguments,
    EncoderOnlyEmbedderTrainingArguments,
    EncoderOnlyEmbedderModelArguments,
    EncoderOnlyEmbedderRunner,
)
@dataclass
class HardcodedModelArgs(EncoderOnlyEmbedderModelArguments):
    model_name_or_path: str = "/Users/guoxing.lan/projects/models/bge-small-zh-v1.5"
    cache_dir: str = "/Users/guoxing.lan/projects/models/cache/huggingface/hub"

@dataclass
class HardcodedDataArgs(EncoderOnlyEmbedderDataArguments):
    train_data: List[str] = field(
        default_factory=lambda: [
            "/Users/guoxing.lan/projects/github/bge/FlagEmbedding/dataset/train/zh1.jsonl"
        ]
    )
    eval_data: List[str] = field(
        default_factory=lambda: [
            "/Users/guoxing.lan/projects/github/bge/FlagEmbedding/dataset/dev/zh1.jsonl"
        ]
    )
    cache_path: str =  "/Users/guoxing.lan/projects/models/cache"

    # pos_num: int = -1 #multi_pos_loss才需要用到
    train_group_size: int = 4
    eval_group_size: int = 4
    query_max_len: int = 512
    passage_max_len: int = 512
    pad_to_multiple_of: int = 8
    query_instruction_for_retrieval: str = "为这个句子生成表示以用于检索相关文章： "
    query_instruction_format: str = "{}{}"
    knowledge_distillation: bool = False


@dataclass
class HardcodedTrainingArgs(EncoderOnlyEmbedderTrainingArguments):
    # Early stopping parameters
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Number of evaluation calls with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold for measuring the new optimum, to only focus on significant improvements."}
    )

    output_dir: str = "/Users/guoxing.lan/projects/models/outputs/bge-small-zh-v1.5-finetuned"
    logging_dir: str = "/Users/guoxing.lan/projects/models/outputs/bge-small-zh-v1.5-finetuned/logs"
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: int = 1
    report_to: Union[None, str, list[str]] = field(
        default='tensorboard', metadata={"help": "The list of integrations to report the results and logs to."}
    )
    include_for_metrics: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of strings to specify additional data to include in the `compute_metrics` function."
                    "Options: 'inputs', 'loss'."
        },
    )
    save_steps: int = 1

    # distributed training
    local_rank: int = -1
    ddp_backend: str = None
    ddp_find_unused_parameters: bool = False
    use_ipex: bool = False  # 禁用Intel优化
    _n_gpu: int=0
    # 确保以下参数已设置
    # no_cuda: bool = True
    use_cpu: bool = True  # 显式使用CPU
    fp16: bool = False
    deepspeed: str = None
    overwrite_output_dir: bool = False
    # resume_from_checkpoint: str=""
    # resume_from_checkpoint: bool = True
    # resume_from_checkpoint: str = "/Users/guoxing.lan/projects/models/outputs/bge-small-zh-v1.5-finetuned/20250617/checkpoint-4"
    # training parameters
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int =4
    dataloader_drop_last: bool = True
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = True
    negatives_cross_device: bool = False
    temperature: float = 0.02
    sentence_pooling_method: str = "cls"
    normalize_embeddings: bool = True

    # evaluation during training
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    eval_strategy: Union[str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: int = 1  # 每50步评估一次
    eval_accumulation_steps: int = 1  # 梯度累积步数
    load_best_model_at_end: bool = True  # 训练完成后加载最佳模型
    metric_for_best_model: str = "eval_loss"  # 以验证集loss作为指标
# ====================================================================
def main():
    '''
    parser = HfArgumentParser((
        EncoderOnlyEmbedderModelArguments,
        EncoderOnlyEmbedderDataArguments,
        EncoderOnlyEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    '''

    # 直接实例化参数对象
    model_args = HardcodedModelArgs()
    data_args = HardcodedDataArgs()
    training_args = HardcodedTrainingArgs()
    # 添加early stopping callback
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold
        )
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # timestamp='20250617'
    training_args.output_dir=os.path.join(training_args.output_dir,timestamp)
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    training_args.include_for_metrics=['loss']
    model_args: EncoderOnlyEmbedderModelArguments
    data_args: EncoderOnlyEmbedderDataArguments
    training_args: EncoderOnlyEmbedderTrainingArguments

    runner = EncoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        callbacks=callbacks  # 添加callbacks参数
    )
    runner.run()


if __name__ == "__main__":
    import numpy as np
    a=np.array([1,2,3])
    import torch
    b=torch.tensor(a)
    print(b)
    main()