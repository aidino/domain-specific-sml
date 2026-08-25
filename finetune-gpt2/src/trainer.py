"""
Training Orchestration Module cho GPT-2 Fine-tuning.
Xử lý logic thiết lập Trainer, Callbacks, Metrics và Hyperparameter Search.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.callbacks import (
    GenerationSampleCallback,
    RichLoggingCallback,
    TensorBoardMetricsCallback,
)
from src.config import PipelineConfig
from src.hp_search import run_hp_search
from src.utils import setup_logger

logger = setup_logger("trainer")


def _build_training_args(config: PipelineConfig, best_hp: Optional[Dict[str, Any]] = None) -> TrainingArguments:
    """
    Xây dựng cấu hình TrainingArguments từ PipelineConfig và tham số từ HP Search.
    
    Args:
        config (PipelineConfig): Cấu hình pipeline.
        best_hp (dict, optional): Tham số tối ưu từ HP search. Defaults to None.
        
    Returns:
        TrainingArguments: Đối tượng args cho HF Trainer.
    """
    # Tự động phát hiện chế độ mixed precision (bf16/fp16)
    bf16 = False
    fp16 = False
    
    if config.training.mixed_precision == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                bf16 = True
            else:
                fp16 = True
    elif config.training.mixed_precision == "bf16":
        bf16 = True
    elif config.training.mixed_precision == "fp16":
        fp16 = True

    # Đường dẫn lưu log TensorBoard
    logging_dir = f"{config.logging.tensorboard_dir}/{config.logging.experiment_name}"
    
    # Thiết lập cơ bản
    args_dict = {
        "output_dir": config.output.checkpoint_dir,
        "num_train_epochs": config.training.num_epochs,
        "per_device_train_batch_size": config.training.batch_size,
        "per_device_eval_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_steps": max(1, int(config.training.warmup_ratio * config.training.num_epochs * 100)),  # Chuyển warmup_ratio → warmup_steps (ước lượng)
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "seed": config.training.seed,
        "bf16": bf16,
        "fp16": fp16,
        "logging_steps": config.logging.logging_steps,
        "report_to": ["tensorboard"],
        "eval_strategy": config.output.save_strategy, # Bắt buộc giống save_strategy khi load_best_model_at_end=True
        "save_strategy": config.output.save_strategy,
        "save_steps": config.output.save_steps,
        "save_total_limit": config.output.save_total_limit,
        "load_best_model_at_end": config.early_stopping.enabled,
        "metric_for_best_model": "eval_loss", # Thường dùng eval_loss làm gốc
        "greater_is_better": False,
        "remove_unused_columns": False, # Tránh lỗi với data collators tùy chỉnh
    }

    # Ghi đè các tham số nếu có best_hp
    if best_hp:
        for k, v in best_hp.items():
            if k in args_dict:
                args_dict[k] = v
                logger.info(f"Đã ghi đè {k} = {v} từ kết quả HP search.")

    return TrainingArguments(**args_dict)


def _build_callbacks(config: PipelineConfig, tokenizer: PreTrainedTokenizerBase) -> List[TrainerCallback]:
    """
    Khởi tạo danh sách các callbacks cần thiết cho quá trình huấn luyện.
    
    Args:
        config (PipelineConfig): Cấu hình pipeline.
        tokenizer (PreTrainedTokenizerBase): Tokenizer sử dụng.
        
    Returns:
        List[TrainerCallback]: Danh sách callbacks.
    """
    callbacks = []
    
    # Bắt buộc: Log đẹp ra màn hình console
    callbacks.append(RichLoggingCallback())
    
    # Bắt buộc: Ghi nhận thêm metric ra TensorBoard
    callbacks.append(TensorBoardMetricsCallback())
    
    # Early Stopping Callback nếu được bật
    if config.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping.patience,
                early_stopping_threshold=config.early_stopping.threshold,
            )
        )
        logger.info(f"Đã bật EarlyStopping: patience={config.early_stopping.patience}")
        
    # Callback sinh mẫu văn bản (chỉ cho causal_lm hoặc completion)
    if config.model.task_type in ["causal_lm", "completion"]:
        callbacks.append(GenerationSampleCallback(tokenizer=tokenizer))
        logger.info("Đã bật GenerationSampleCallback để sinh text cuối mỗi epoch.")
        
    return callbacks


def _build_compute_metrics(config: PipelineConfig) -> Optional[Callable]:
    """
    Trả về hàm tính toán metrics (compute_metrics) tùy theo từng task_type.
    
    Args:
        config (PipelineConfig): Cấu hình pipeline.
        
    Returns:
        Callable hoặc None: Hàm tính metrics hoặc None.
    """
    if config.model.task_type == "classification":
        # Load các metric chuẩn để đánh giá phân loại văn bản
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        def compute_metrics(eval_pred):
            """Hàm tính accuracy và f1 cho đánh giá."""
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            acc = accuracy_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
            
            return {
                "accuracy": acc["accuracy"],
                "f1": f1["f1"],
            }
        
        return compute_metrics
        
    # Với Causal LM và Completion, không tính ở đây (perplexity được tính riêng)
    return None


def run_training(
    config: PipelineConfig,
    datasets: DatasetDict,
    data_collator: Any,
    tokenizer: PreTrainedTokenizerBase,
    model_init: Callable,
    best_hp: Optional[Dict[str, Any]] = None,
) -> Tuple[Trainer, Dict[str, Any]]:
    """
    Hàm điều phối toàn bộ quá trình huấn luyện model.
    
    Args:
        config (PipelineConfig): Cấu hình chung của pipeline.
        datasets (DatasetDict): Bộ dữ liệu đã chuẩn bị.
        data_collator (Any): Data collator cho Trainer.
        tokenizer (PreTrainedTokenizerBase): Tokenizer xử lý văn bản.
        model_init (Callable): Hàm khởi tạo mô hình.
        best_hp (dict, optional): Tham số HP search (nếu có).
        
    Returns:
        Tuple[Trainer, Dict]: Đối tượng Trainer và metrics kết quả train.
    """
    logger.info("Bắt đầu quá trình huấn luyện...")
    
    # 1. Chạy HP Search nếu có yêu cầu và chưa có kết quả best_hp
    if config.hp_search.enabled and best_hp is None:
        logger.info("Khởi động quá trình tìm kiếm siêu tham số (HP Search)...")
        # Khởi tạo trainer tạm thời cho HP search
        temp_args = _build_training_args(config)
        temp_trainer = Trainer(
            model_init=model_init,
            args=temp_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            data_collator=data_collator,
            compute_metrics=_build_compute_metrics(config),
        )
        best_hp = run_hp_search(config, temp_trainer)
        logger.info(f"Kết thúc HP Search. Tham số tốt nhất: {best_hp}")
        
    # 2. Xây dựng các thành phần cho Trainer chính
    training_args = _build_training_args(config, best_hp)
    callbacks = _build_callbacks(config, tokenizer)
    compute_metrics_fn = _build_compute_metrics(config)
    
    # 3. Tạo Trainer instance
    # Lưu ý: truyền model_init thay vì model để đảm bảo seed và initialization độc lập
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    
    # 4. Tiến hành huấn luyện
    logger.info("Bắt đầu gọi trainer.train()...")
    train_result = trainer.train()
    
    # 5. Lưu lại thông số huấn luyện cuối cùng
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Quá trình huấn luyện hoàn tất thành công.")
    
    return trainer, metrics
