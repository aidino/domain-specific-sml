"""
Config System — Hệ thống cấu hình Pipeline
=============================================
Đọc file YAML config → parse vào dataclasses → validate types.
Hỗ trợ CLI override cho bất kỳ field nào.
"""

# === Standard Library ===
import os               # Thao tác hệ thống (đường dẫn, biến môi trường)
from dataclasses import dataclass, field, asdict  # Tạo data classes tự động
from typing import Any, Dict, List, Optional      # Type hints cho Python 3.10+

# === Third-Party ===
import yaml             # Đọc/ghi file YAML config

# === Local ===
from src.utils import setup_logger  # Logger từ module utils

# Tạo logger cho module config — tên "gpt2_finetune.config" để dễ trace
logger = setup_logger("gpt2_finetune.config")


# ============================================================
# DATACLASS DEFINITIONS — Định nghĩa cấu trúc config
# ============================================================

@dataclass
class ModelConfig:
    """
    Cấu hình Model — chọn model và loại task.

    name: Tên model HuggingFace hoặc local path
        "gpt2"        → 124M params, GPU nhỏ (4-8GB)
        "gpt2-medium" → 355M params (8-16GB)
        "gpt2-large"  → 774M params (16-24GB)
        "gpt2-xl"     → 1.5B params (24GB+)
    task_type: Loại task fine-tuning
        "causal_lm"      → Sinh văn bản tự do
        "classification" → Phân loại văn bản
        "completion"     → Instruction fine-tuning
    num_labels: Số labels (chỉ cho classification)
    """
    name: str = "gpt2"                    # Tên model trên HuggingFace Hub
    task_type: str = "causal_lm"          # Loại task: causal_lm | classification | completion
    num_labels: int = 2                   # Số labels cho classification


@dataclass
class PeftConfig:
    """
    Cấu hình PEFT/LoRA — Fine-tuning hiệu quả.

    enabled: true → chỉ train ~0.2-1.5% params, tiết kiệm VRAM
    r: Rank LoRA (4/8/16/32/64), khuyến nghị 16
    lora_alpha: Scaling = 2 × r
    lora_dropout: Dropout chống overfit (0.0-0.2)
    target_modules: GPT-2 dùng Conv1D: "c_attn", "c_proj", "c_fc"
    """
    enabled: bool = False                 # Bật/tắt LoRA fine-tuning
    r: int = 16                           # Rank ma trận LoRA
    lora_alpha: int = 32                  # Hệ số scaling = 2 × r
    lora_dropout: float = 0.05            # Dropout cho LoRA layers
    target_modules: List[str] = field(    # Layers áp dụng LoRA
        default_factory=lambda: ["c_attn"]
    )


@dataclass
class AugmentationConfig:
    """
    Cấu hình Data Augmentation — tăng cường dữ liệu training.

    techniques: "synonym_replace", "random_delete", "random_swap", "random_insert"
    augment_ratio: Tỷ lệ samples được augment (0.0-1.0)
    """
    enabled: bool = False                 # Bật/tắt augmentation
    techniques: List[str] = field(        # Danh sách kỹ thuật augmentation
        default_factory=lambda: ["synonym_replace", "random_delete"]
    )
    augment_ratio: float = 0.2            # Tỷ lệ augment (20% mặc định)


@dataclass
class DataConfig:
    """
    Cấu hình Data — nguồn dữ liệu và cách xử lý.

    source: "huggingface" (Hub) hoặc "local" (CSV/JSON/TXT)
    max_length: 128 (classification), 512 (text gen), 1024 (max GPT-2)
    """
    source: str = "huggingface"           # Nguồn: huggingface | local
    dataset_name: str = "wikitext"        # Tên dataset trên HuggingFace Hub
    dataset_config: Optional[str] = "wikitext-2-raw-v1"  # Config/subset
    local_path: Optional[str] = None      # Đường dẫn file local
    text_column: str = "text"             # Tên cột chứa text
    label_column: str = "label"           # Tên cột chứa label
    max_length: int = 512                 # Độ dài tối đa sequence
    split_ratios: List[float] = field(    # Tỷ lệ chia [train, val, test]
        default_factory=lambda: [0.8, 0.1, 0.1]
    )
    augmentation: AugmentationConfig = field(  # Config augmentation
        default_factory=AugmentationConfig
    )


@dataclass
class TrainingConfig:
    """
    Cấu hình Training — hyperparameters.

    learning_rate: 1e-5 (bảo thủ), 5e-5 (phổ biến), 1e-4 (LoRA only)
    lr_scheduler_type: "linear", "cosine" (khuyến nghị), "constant"
    mixed_precision: "auto" (khuyến nghị), "bf16", "fp16", "none"
    """
    num_epochs: int = 5                   # Số epoch training
    batch_size: int = 8                   # Batch size trên mỗi device
    gradient_accumulation_steps: int = 2  # Tích lũy gradient (effective BS = BS × này)
    learning_rate: float = 5e-5           # Tốc độ học
    weight_decay: float = 0.01            # Regularization chống overfit
    warmup_ratio: float = 0.05            # % steps warmup (0.0-0.1)
    lr_scheduler_type: str = "cosine"     # LR scheduler: linear | cosine | constant
    seed: int = 42                        # Seed cho reproducibility
    mixed_precision: str = "auto"         # auto | bf16 | fp16 | none


@dataclass
class EarlyStoppingConfig:
    """
    Cấu hình Early Stopping — dừng khi không cải thiện.

    patience: 3 (dataset nhỏ), 5 (dataset lớn)
    threshold: Ngưỡng cải thiện tối thiểu
    """
    enabled: bool = True                  # Bật/tắt early stopping
    patience: int = 3                     # Số eval không cải thiện → dừng
    threshold: float = 0.001              # Ngưỡng cải thiện tối thiểu


@dataclass
class HPSearchConfig:
    """
    Cấu hình Optuna Hyperparameter Search.

    direction: "minimize" (loss), "maximize" (accuracy)
    n_trials: 10-20 (exploration), 30-50 (kỹ)
    """
    enabled: bool = False                 # Bật/tắt HP search
    n_trials: int = 15                    # Số trials Optuna chạy
    direction: str = "minimize"           # minimize (loss) | maximize (accuracy)
    metric: str = "eval_loss"             # Metric objective
    search_space: Dict[str, Any] = field( # Khoảng search
        default_factory=lambda: {
            "learning_rate": [1e-5, 5e-4],
            "batch_size": [4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4],
            "weight_decay": [0.0, 0.2],
            "warmup_ratio": [0.0, 0.1],
            "num_epochs": [3, 6],
        }
    )


@dataclass
class LoggingConfig:
    """
    Cấu hình Logging & TensorBoard.

    level: "DEBUG" | "INFO" (khuyến nghị) | "WARNING" | "ERROR"
    """
    level: str = "INFO"                   # Mức log
    logging_steps: int = 50               # Log mỗi N steps
    tensorboard_dir: str = "./runs"       # Thư mục TensorBoard
    experiment_name: str = "gpt2-finetune"  # Tên experiment


@dataclass
class OutputConfig:
    """
    Cấu hình Output — lưu checkpoints và model.

    save_strategy: "epoch" (dataset nhỏ) | "steps" (dataset lớn)
    save_total_limit: 2-3 để tiết kiệm disk
    """
    checkpoint_dir: str = "./checkpoints"  # Thư mục checkpoints
    model_dir: str = "./final_model"       # Thư mục model cuối
    save_total_limit: int = 3              # Giới hạn checkpoints
    save_strategy: str = "epoch"           # epoch | steps
    save_steps: int = 500                  # Steps giữa mỗi lần save


@dataclass
class PipelineConfig:
    """Config tổng hợp — chứa TẤT CẢ config sections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    hp_search: HPSearchConfig = field(default_factory=HPSearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ============================================================
# CONFIG LOADING & SAVING — Đọc/Ghi config
# ============================================================

def _dict_to_dataclass(data: dict, dc_class: type) -> Any:
    """
    Chuyển dict thành dataclass instance, hỗ trợ nested dataclasses.

    Args:
        data: Dict chứa giá trị config
        dc_class: Dataclass class cần tạo instance

    Returns:
        Instance của dc_class đã populate giá trị
    """
    # Import field metadata để duyệt fields
    from dataclasses import fields as dc_fields

    # Dict chứa giá trị đã xử lý cho constructor
    kwargs = {}

    # Duyệt qua từng field trong dataclass
    for f in dc_fields(dc_class):
        # Chỉ xử lý field có trong data dict
        if f.name in data:
            value = data[f.name]
            # Nếu field type là dataclass VÀ value là dict → recursive parse
            if hasattr(f.type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[f.name] = _dict_to_dataclass(value, f.type)
            else:
                kwargs[f.name] = value
    # Tạo instance — fields không có trong data dùng default value
    return dc_class(**kwargs)


def load_config(yaml_path: str, cli_overrides: dict = None) -> PipelineConfig:
    """
    Load config từ file YAML, merge CLI overrides, validate.

    Args:
        yaml_path: Đường dẫn tới file YAML
        cli_overrides: Dict override từ CLI, VD: {"training.batch_size": 4}

    Returns:
        PipelineConfig đã validate

    Raises:
        FileNotFoundError: File YAML không tồn tại
        ValueError: Giá trị config không hợp lệ
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file không tìm thấy: {yaml_path}")

    # Đọc YAML → dict
    logger.info(f"Đọc config từ: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # Áp dụng CLI overrides: "training.batch_size" → navigate dict → set value
    if cli_overrides:
        logger.info(f"Áp dụng {len(cli_overrides)} CLI overrides")
        for dotted_key, value in cli_overrides.items():
            keys = dotted_key.split(".")        # Tách dotted key thành list
            target = raw_config
            for key in keys[:-1]:               # Navigate vào nested dict
                target = target.setdefault(key, {})
            target[keys[-1]] = value            # Gán giá trị mới
            logger.debug(f"  Override: {dotted_key} = {value}")

    # Parse dict → PipelineConfig dataclass
    config = _dict_to_dataclass(raw_config, PipelineConfig)

    # Validate tất cả giá trị
    _validate_config(config)

    logger.info("✅ Config loaded và validated thành công")
    return config


def _validate_config(config: PipelineConfig) -> None:
    """
    Validate config — kiểm tra giá trị hợp lệ.

    Raises:
        ValueError: Nếu bất kỳ giá trị nào không hợp lệ
    """
    # Kiểm tra task_type
    valid_tasks = {"causal_lm", "classification", "completion"}
    if config.model.task_type not in valid_tasks:
        raise ValueError(f"task_type '{config.model.task_type}' không hợp lệ. Chọn: {valid_tasks}")

    # Kiểm tra mixed_precision
    valid_precisions = {"auto", "bf16", "fp16", "none"}
    if config.training.mixed_precision not in valid_precisions:
        raise ValueError(f"mixed_precision '{config.training.mixed_precision}' không hợp lệ. Chọn: {valid_precisions}")

    # Kiểm tra save_strategy
    valid_strategies = {"epoch", "steps"}
    if config.output.save_strategy not in valid_strategies:
        raise ValueError(f"save_strategy '{config.output.save_strategy}' không hợp lệ. Chọn: {valid_strategies}")

    # Kiểm tra split_ratios tổng ≈ 1.0
    ratio_sum = sum(config.data.split_ratios)
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(f"split_ratios tổng = {ratio_sum}, phải = 1.0")

    # Kiểm tra giá trị dương
    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate phải > 0, hiện tại: {config.training.learning_rate}")
    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size phải > 0, hiện tại: {config.training.batch_size}")
    if config.training.num_epochs <= 0:
        raise ValueError(f"num_epochs phải > 0, hiện tại: {config.training.num_epochs}")

    # Kiểm tra source
    valid_sources = {"huggingface", "local"}
    if config.data.source not in valid_sources:
        raise ValueError(f"data.source '{config.data.source}' không hợp lệ. Chọn: {valid_sources}")

    # Kiểm tra local_path khi source="local"
    if config.data.source == "local" and not config.data.local_path:
        raise ValueError("data.local_path phải được chỉ định khi source='local'")

    # Kiểm tra HP search direction
    valid_directions = {"minimize", "maximize"}
    if config.hp_search.direction not in valid_directions:
        raise ValueError(f"hp_search.direction '{config.hp_search.direction}' không hợp lệ. Chọn: {valid_directions}")

    logger.debug("Config validation passed")


def save_config(config: PipelineConfig, path: str) -> None:
    """
    Lưu config thành file YAML — dùng cho reproducibility.

    Args:
        config: PipelineConfig cần lưu
        path: Đường dẫn file YAML đích
    """
    config_dict = asdict(config)          # Dataclass → dict
    # Tạo thư mục cha nếu chưa có
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    # Ghi file YAML (Unicode cho tiếng Việt, block style dễ đọc)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info(f"Config đã lưu tại: {path}")


def config_to_dict(config: PipelineConfig) -> dict:
    """
    Chuyển PipelineConfig thành dict phẳng — cho banner, logging.

    Returns:
        Dict với keys: model_name, task_type, peft_enabled, peft_r, peft_target
    """
    return {
        "model_name": config.model.name,
        "task_type": config.model.task_type,
        "peft_enabled": config.peft.enabled,
        "peft_r": config.peft.r,
        "peft_target": ", ".join(config.peft.target_modules),
    }
