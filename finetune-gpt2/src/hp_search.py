import optuna
from typing import Callable, Dict, Any
from transformers import Trainer
from src.config import PipelineConfig
from src.utils import setup_logger, format_metrics_table

# Khởi tạo logger cho module
logger = setup_logger(__name__)

def build_hp_space(config: PipelineConfig) -> Callable[[optuna.Trial], Dict[str, Any]]:
    """
    Xây dựng không gian tìm kiếm hyperparameter dựa trên cấu hình.
    Trả về một hàm callable nhận optuna.Trial và trả về dict các hyperparameters.
    """
    search_space = config.hp_search.search_space or {}
    
    def hp_space(trial: optuna.Trial) -> Dict[str, Any]:
        # Khởi tạo dict chứa các hyperparameters được chọn
        params = {}
        
        # Tìm kiếm learning rate trên thang log (ví dụ: 1e-5 đến 1e-3)
        if "learning_rate" in search_space:
            min_lr, max_lr = search_space["learning_rate"]
            params["learning_rate"] = trial.suggest_float("learning_rate", min_lr, max_lr, log=True)
            
        # Lựa chọn batch size từ một danh sách cố định
        if "batch_size" in search_space:
            choices = search_space["batch_size"]
            params["per_device_train_batch_size"] = trial.suggest_categorical("per_device_train_batch_size", choices)
            
        # Lựa chọn gradient accumulation steps từ một danh sách cố định
        if "gradient_accumulation_steps" in search_space:
            choices = search_space["gradient_accumulation_steps"]
            params["gradient_accumulation_steps"] = trial.suggest_categorical("gradient_accumulation_steps", choices)
            
        # Tìm kiếm weight decay
        if "weight_decay" in search_space:
            min_wd, max_wd = search_space["weight_decay"]
            params["weight_decay"] = trial.suggest_float("weight_decay", min_wd, max_wd)
            
        # Tìm kiếm warmup ratio
        if "warmup_ratio" in search_space:
            min_wr, max_wr = search_space["warmup_ratio"]
            params["warmup_ratio"] = trial.suggest_float("warmup_ratio", min_wr, max_wr)
            
        # Tìm kiếm số epoch huấn luyện
        if "num_epochs" in search_space:
            min_ep, max_ep = search_space["num_epochs"]
            params["num_train_epochs"] = trial.suggest_int("num_train_epochs", min_ep, max_ep)
            
        return params
        
    return hp_space

def run_hp_search(config: PipelineConfig, trainer: Trainer) -> Dict[str, Any]:
    """
    Thực hiện quá trình tìm kiếm hyperparameter bằng Optuna.
    """
    logger.info("Bắt đầu quá trình tìm kiếm hyperparameters với Optuna.")
    
    # Cấu hình sampler và pruner cho Optuna
    # Sử dụng TPESampler với seed cố định để đảm bảo tính lặp lại
    sampler = optuna.samplers.TPESampler(seed=config.training.seed)
    
    # Sử dụng MedianPruner để dừng sớm các trial có hiệu suất kém
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    
    # Lấy hàm khởi tạo không gian tìm kiếm
    hp_space_fn = build_hp_space(config)
    
    # Chạy quá trình tìm kiếm thông qua HuggingFace Trainer
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space_fn,
        backend="optuna",
        n_trials=config.hp_search.n_trials,
        direction=config.hp_search.direction,
        sampler=sampler,
        pruner=pruner
    )
    
    # Ghi log kết quả tốt nhất
    logger.info("Hoàn thành tìm kiếm hyperparameters. Kết quả tốt nhất:")
    best_params = best_run.hyperparameters
    
    # Sử dụng tiện ích format_metrics_table để hiển thị đẹp mắt
    # Chuyển đổi dict params sang string để hiển thị
    formatted_table = format_metrics_table(best_params)
    logger.info("\n" + formatted_table)
    
    return best_params

