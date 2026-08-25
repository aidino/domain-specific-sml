# GPT-2 Fine-Tuning Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, production-ready GPT-2 fine-tuning pipeline with CLI interface, YAML+dataclass config, LoRA/PEFT support, Optuna HP search, TensorBoard logging, and comprehensive Vietnamese-commented code.

**Architecture:** Modular pipeline in `finetune-gpt2/` with 10 Python modules, each responsible for one pipeline step. Steps can run independently or chained via CLI. Config system uses YAML files parsed into validated Python dataclasses.

**Tech Stack:** Python 3.14, PyTorch, HuggingFace Transformers/Datasets/Evaluate/PEFT, Optuna, TensorBoard, Rich (console UI), PyYAML

## Global Constraints

- All code in `finetune-gpt2/` subdirectory
- Vietnamese comments on every line explaining meaning
- Config options with multiple choices must list all options with explanations and suggestions
- Python 3.10+ type hints throughout
- All imports at top of file, grouped: stdlib → third-party → local
- Follow HuggingFace Trainer API best practices (model_init pattern, dynamic padding, eval_strategy=save_strategy)
- GPT-2 specific: `pad_token = eos_token`, LoRA targets `c_attn`/`c_proj` (Conv1D not nn.Linear)
- Classification: `tokenizer.padding_side = "left"`
- Spec: `docs/superpowers/specs/2026-08-25-gpt2-finetune-pipeline-design.md`

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `finetune-gpt2/requirements.txt`
- Create: `finetune-gpt2/src/__init__.py`

**Interfaces:**
- Consumes: Nothing (first task)
- Produces: `requirements.txt` with all pinned dependencies, empty `src/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
# ============================================================
# DEPENDENCIES CHO GPT-2 FINE-TUNING PIPELINE
# ============================================================

# === Core Framework ===
# HuggingFace Transformers: Thư viện chính cho model, tokenizer, Trainer API
transformers>=4.40.0
# HuggingFace Datasets: Thư viện load và xử lý dataset từ Hub hoặc local
datasets>=2.18.0
# PyTorch: Backend deep learning framework
torch>=2.0.0
# PEFT: Parameter-Efficient Fine-Tuning (LoRA, Adapter)
peft>=0.10.0
# Optuna: Hyperparameter optimization framework
optuna>=3.6.0
# Accelerate: Hỗ trợ multi-GPU, mixed precision, distributed training
accelerate>=0.28.0

# === Evaluation ===
# HuggingFace Evaluate: Tính metrics (accuracy, f1, rouge, bleu)
evaluate>=0.4.0
# Scikit-learn: Confusion matrix, classification report
scikit-learn>=1.4.0

# === Logging & Visualization ===
# TensorBoard: Visualization training history
tensorboard>=2.16.0
# Rich: Beautiful console output với colors, tables, progress bars
rich>=13.7.0

# === Data Processing ===
# NumPy: Numerical operations
numpy>=1.26.0
# Pandas: Data manipulation cho local files (CSV, JSON)
pandas>=2.2.0

# === Utilities ===
# PyYAML: Đọc/ghi file YAML config
pyyaml>=6.0
# Safetensors: Lưu model weights an toàn, nhanh
safetensors>=0.4.0

# === Optional ===
# TRL: DataCollatorForCompletionOnlyLM cho instruction fine-tuning
trl>=0.8.0
# NLPAug: Text augmentation (synonym replace, random delete...)
# nlpaug>=1.1.11
```

- [ ] **Step 2: Create src/__init__.py**

```python
"""
GPT-2 Fine-Tuning Pipeline — Source Package
============================================
Pipeline chuyên nghiệp để fine-tune GPT-2 model.
Hỗ trợ: causal LM, classification, instruction completion.
"""
```

- [ ] **Step 3: Verify directory structure exists**

Run:
```bash
mkdir -p finetune-gpt2/src finetune-gpt2/configs
ls -la finetune-gpt2/
ls -la finetune-gpt2/src/
```
Expected: directories exist, `requirements.txt` and `src/__init__.py` present

- [ ] **Step 4: Commit**

```bash
git add finetune-gpt2/
git commit -m "feat: scaffold project structure and dependencies"
```

---

### Task 2: Utilities Module (`src/utils.py`)

**Files:**
- Create: `finetune-gpt2/src/utils.py`

**Interfaces:**
- Consumes: Nothing (foundation module)
- Produces:
  - `setup_logger(name: str, level: str = "INFO") -> logging.Logger`
  - `detect_device() -> dict`
  - `set_seed(seed: int) -> None`
  - `class StepTimer` (context manager)
  - `format_banner(config_dict: dict, device_info: dict) -> str`
  - `format_metrics_table(metrics: dict, title: str = "") -> str`
  - `format_duration(seconds: float) -> str`

- [ ] **Step 1: Write src/utils.py with all utility functions**

```python
"""
Utilities Module — Tiện ích chung cho pipeline
================================================
Cung cấp: logger setup, device detection, seed, timer, formatting.
Được dùng bởi TẤT CẢ các module khác trong pipeline.
"""

# === Standard Library ===
import logging          # Thư viện logging chuẩn của Python
import os               # Thao tác hệ thống (biến môi trường, đường dẫn)
import random           # Random number generator
import time             # Đo thời gian

# === Third-Party ===
import numpy as np      # Thư viện tính toán số học
import torch            # PyTorch framework

# === Constants ===
# Emoji cho từng level log — giúp dễ phân biệt khi đọc log
LOG_EMOJIS = {
    "DEBUG": "🔍",      # Debug: chi tiết cho developer
    "INFO": "ℹ️ ",       # Info: thông tin chung
    "WARNING": "⚠️ ",    # Warning: cảnh báo cần chú ý
    "ERROR": "❌",       # Error: lỗi xảy ra
    "CRITICAL": "🔥",   # Critical: lỗi nghiêm trọng
}

# Màu ANSI cho từng level — hiển thị màu trong terminal
LOG_COLORS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[35m",   # Magenta
}
RESET_COLOR = "\033[0m"       # Reset về màu mặc định


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter — format log với màu sắc và emoji.

    Format: [YYYY-MM-DD HH:MM:SS] 🔍 DEBUG | module | Message
    Giúp dễ đọc và trace log trong terminal.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Lấy tên level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        level_name = record.levelname
        # Lấy emoji tương ứng với level, mặc định là "📝"
        emoji = LOG_EMOJIS.get(level_name, "📝")
        # Lấy mã màu ANSI tương ứng với level
        color = LOG_COLORS.get(level_name, "")
        # Lấy tên module (phần cuối của logger name, VD: "data", "model")
        module_name = record.name.split(".")[-1]
        # Định dạng thời gian theo format YYYY-MM-DD HH:MM:SS
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Ghép thành chuỗi log hoàn chỉnh với màu sắc
        formatted = (
            f"[{timestamp}] "                                         # Thời gian
            f"{emoji} {color}{level_name:8s}{RESET_COLOR} "           # Level với màu
            f"| {module_name:10s} "                                   # Tên module (căn lề 10 ký tự)
            f"| {record.getMessage()}"                                # Nội dung message
        )
        return formatted


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Tạo và cấu hình logger với format đẹp.

    Args:
        name: Tên logger (thường là tên module, VD: "gpt2_finetune.data")
        level: Mức độ log — "DEBUG", "INFO", "WARNING", "ERROR"
               - "DEBUG"   → Hiển thị tất cả, dùng khi debug
               - "INFO"    → Thông tin chung (khuyến nghị cho production)
               - "WARNING" → Chỉ cảnh báo và lỗi
               - "ERROR"   → Chỉ lỗi

    Returns:
        Logger đã cấu hình sẵn, sẵn sàng sử dụng
    """
    # Tạo logger với tên được chỉ định
    logger = logging.getLogger(name)
    # Set level cho logger (chuyển string → logging constant, VD: "INFO" → logging.INFO)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Chỉ thêm handler nếu logger chưa có — tránh duplicate log
    if not logger.handlers:
        # Tạo handler ghi log ra console (stdout)
        console_handler = logging.StreamHandler()
        # Gán formatter đẹp đã tạo ở trên
        console_handler.setFormatter(ColoredFormatter())
        # Thêm handler vào logger
        logger.addHandler(console_handler)

    return logger


def detect_device() -> dict:
    """
    Tự động phát hiện hardware (CPU / GPU / Multi-GPU).

    Returns:
        dict chứa thông tin:
        {
            "device": "cuda" hoặc "cpu",
            "n_gpus": số lượng GPU,
            "gpu_names": danh sách tên GPU,
            "total_vram_gb": tổng VRAM (GB),
            "bf16_supported": GPU có hỗ trợ bfloat16 không
        }
    """
    # Khởi tạo dict kết quả với giá trị mặc định cho CPU
    device_info = {
        "device": "cpu",           # Mặc định là CPU
        "n_gpus": 0,               # Không có GPU
        "gpu_names": [],           # Danh sách tên GPU rỗng
        "total_vram_gb": 0.0,      # Không có VRAM
        "bf16_supported": False,   # CPU không hỗ trợ bf16 (theo mặc định)
    }

    # Kiểm tra CUDA có khả dụng không (= có GPU NVIDIA + driver + CUDA toolkit)
    if torch.cuda.is_available():
        # Cập nhật device thành "cuda"
        device_info["device"] = "cuda"
        # Đếm số GPU hiện có
        device_info["n_gpus"] = torch.cuda.device_count()

        # Tổng VRAM trên tất cả GPU
        total_vram = 0.0
        # Duyệt qua từng GPU để lấy thông tin
        for i in range(device_info["n_gpus"]):
            # Lấy tên GPU (VD: "NVIDIA GeForce RTX 4090")
            gpu_name = torch.cuda.get_device_name(i)
            device_info["gpu_names"].append(gpu_name)
            # Lấy tổng VRAM của GPU này (bytes → GB)
            vram_bytes = torch.cuda.get_device_properties(i).total_mem
            total_vram += vram_bytes / (1024 ** 3)  # Chuyển bytes sang GB

        # Lưu tổng VRAM, làm tròn 1 chữ số thập phân
        device_info["total_vram_gb"] = round(total_vram, 1)
        # Kiểm tra GPU có hỗ trợ bf16 không (Ampere+ = RTX 30xx/40xx, A100, H100)
        device_info["bf16_supported"] = torch.cuda.is_bf16_supported()

    return device_info


def set_seed(seed: int) -> None:
    """
    Đặt seed cho tất cả random generators — đảm bảo reproducibility.
    Cùng seed → cùng kết quả training (trên cùng hardware).

    Args:
        seed: Số nguyên bất kỳ (phổ biến: 42, 0, 123)
    """
    # Set seed cho Python random module
    random.seed(seed)
    # Set seed cho NumPy random
    np.random.seed(seed)
    # Set seed cho PyTorch CPU
    torch.manual_seed(seed)
    # Set seed cho tất cả GPU (nếu có)
    torch.cuda.manual_seed_all(seed)
    # Đặt biến môi trường PYTHONHASHSEED — ảnh hưởng hash() của Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Bật deterministic mode cho cuDNN — chậm hơn nhưng reproducible
    torch.backends.cudnn.deterministic = True
    # Tắt benchmark mode — benchmark tìm algorithm nhanh nhất nhưng non-deterministic
    torch.backends.cudnn.benchmark = False


class StepTimer:
    """
    Context manager đo thời gian thực hiện mỗi pipeline step.

    Sử dụng:
        with StepTimer("Data Preparation", logger) as timer:
            # ... code chạy trong step ...
        # Tự động log thời gian khi thoát block

    Attributes:
        elapsed: Thời gian đã trôi qua (giây) — có giá trị sau khi thoát block
    """

    def __init__(self, step_name: str, logger: logging.Logger = None):
        """
        Args:
            step_name: Tên step (VD: "Data Preparation", "Training")
            logger: Logger để ghi log thời gian (optional)
        """
        # Lưu tên step để hiển thị trong log
        self.step_name = step_name
        # Logger để ghi thông tin (nếu không truyền thì dùng root logger)
        self.logger = logger or logging.getLogger(__name__)
        # Thời gian bắt đầu — sẽ được gán trong __enter__
        self.start_time = None
        # Thời gian đã trôi qua — sẽ được tính trong __exit__
        self.elapsed = 0.0

    def __enter__(self):
        """Bắt đầu đo thời gian khi vào block `with`."""
        # Ghi nhận thời điểm bắt đầu (dùng time.time() cho wall-clock time)
        self.start_time = time.time()
        # Log thông báo bắt đầu step
        self.logger.info(f"Bắt đầu: {self.step_name}")
        # Trả về self để có thể truy cập timer.elapsed sau này
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Kết thúc đo thời gian khi thoát block `with`."""
        # Tính thời gian đã trôi qua = thời điểm hiện tại - thời điểm bắt đầu
        self.elapsed = time.time() - self.start_time
        # Format thời gian thành chuỗi dễ đọc (VD: "1h 23m 45s")
        duration_str = format_duration(self.elapsed)

        # Log kết quả: thành công hoặc thất bại
        if exc_type is None:
            # Không có exception → step hoàn thành thành công
            self.logger.info(f"✅ {self.step_name} hoàn thành trong {duration_str}")
        else:
            # Có exception → step thất bại
            self.logger.error(f"❌ {self.step_name} thất bại sau {duration_str}: {exc_val}")

        # Return False → không suppress exception (để exception propagate bình thường)
        return False


def format_duration(seconds: float) -> str:
    """
    Chuyển đổi số giây thành chuỗi dễ đọc.

    Args:
        seconds: Số giây (VD: 3725.5)

    Returns:
        Chuỗi formatted (VD: "1h 2m 5s" hoặc "45.2s" hoặc "350ms")

    Examples:
        format_duration(0.35)   → "350ms"
        format_duration(45.2)   → "45.2s"
        format_duration(3725)   → "1h 2m 5s"
    """
    # Dưới 1 giây → hiển thị milliseconds
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    # Dưới 60 giây → hiển thị giây với 1 số thập phân
    elif seconds < 60:
        return f"{seconds:.1f}s"
    # Dưới 1 giờ → hiển thị phút và giây
    elif seconds < 3600:
        minutes = int(seconds // 60)       # Số phút (phần nguyên)
        secs = int(seconds % 60)           # Số giây còn lại
        return f"{minutes}m {secs}s"
    # Từ 1 giờ trở lên → hiển thị giờ, phút, giây
    else:
        hours = int(seconds // 3600)       # Số giờ
        minutes = int((seconds % 3600) // 60)  # Số phút còn lại
        secs = int(seconds % 60)           # Số giây còn lại
        return f"{hours}h {minutes}m {secs}s"


def format_banner(config_dict: dict, device_info: dict) -> str:
    """
    Tạo banner đẹp hiển thị đầu pipeline run.

    Args:
        config_dict: Dict chứa thông tin config chính:
            {"model_name": str, "task_type": str, "peft_enabled": bool,
             "peft_r": int, "peft_target": str}
        device_info: Dict từ detect_device()

    Returns:
        Chuỗi banner formatted với box drawing characters
    """
    # Lấy thông tin model từ config
    model_name = config_dict.get("model_name", "gpt2")
    task_type = config_dict.get("task_type", "causal_lm")
    peft_enabled = config_dict.get("peft_enabled", False)

    # Tạo chuỗi LoRA info
    if peft_enabled:
        peft_r = config_dict.get("peft_r", 16)
        peft_target = config_dict.get("peft_target", "c_attn")
        lora_str = f"enabled (r={peft_r}, target={peft_target})"
    else:
        lora_str = "disabled (full fine-tuning)"

    # Tạo chuỗi device info
    if device_info["device"] == "cuda":
        # Hiển thị tên GPU đầu tiên × số lượng GPU
        gpu_name = device_info["gpu_names"][0] if device_info["gpu_names"] else "Unknown GPU"
        vram = device_info["total_vram_gb"]
        n_gpus = device_info["n_gpus"]
        device_str = f"{gpu_name} ({vram}GB) × {n_gpus}"
    else:
        device_str = "CPU"

    # Xác định precision dựa trên device
    if device_info["bf16_supported"]:
        precision_str = "bf16 (bfloat16)"
    elif device_info["device"] == "cuda":
        precision_str = "fp16 (float16)"
    else:
        precision_str = "fp32 (float32)"

    # Độ rộng banner cố định
    width = 62

    # Tạo banner với box drawing characters
    lines = [
        "╔" + "═" * width + "╗",
        "║" + "🚀 GPT-2 Fine-Tuning Pipeline".center(width) + "║",
        "╠" + "═" * width + "╣",
        "║" + f"  Model:     {model_name}".ljust(width) + "║",
        "║" + f"  Task:      {task_type}".ljust(width) + "║",
        "║" + f"  LoRA:      {lora_str}".ljust(width) + "║",
        "║" + f"  Device:    {device_str}".ljust(width) + "║",
        "║" + f"  Precision: {precision_str}".ljust(width) + "║",
        "╚" + "═" * width + "╝",
    ]

    return "\n".join(lines)


def format_metrics_table(metrics: dict, title: str = "") -> str:
    """
    Format dict metrics thành bảng căn lề đẹp.

    Args:
        metrics: Dict chứa tên metric → giá trị
            VD: {"eval_loss": 2.39, "perplexity": 10.9, "accuracy": 0.95}
        title: Tiêu đề bảng (optional)

    Returns:
        Chuỗi bảng formatted sẵn sàng in ra console
    """
    # Nếu dict rỗng, trả về thông báo
    if not metrics:
        return "  (Không có metrics)"

    # Tìm độ dài lớn nhất của tên metric (để căn lề)
    max_key_len = max(len(str(k)) for k in metrics.keys())

    # Tạo danh sách các dòng
    lines = []
    # Thêm tiêu đề nếu có
    if title:
        lines.append(f"  📊 {title}")
        lines.append("  " + "─" * (max_key_len + 20))

    # Duyệt qua từng metric và format
    for key, value in metrics.items():
        # Format giá trị: số thực → 4 chữ số thập phân, còn lại giữ nguyên
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        # Căn lề tên metric và giá trị
        lines.append(f"  {str(key).ljust(max_key_len)}  │  {value_str}")

    return "\n".join(lines)
```

- [ ] **Step 2: Verify module imports correctly**

Run:
```bash
cd finetune-gpt2 && python -c "from src.utils import setup_logger, detect_device, set_seed, StepTimer, format_banner, format_metrics_table, format_duration; print('✅ utils module OK')"
```
Expected: `✅ utils module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/utils.py
git commit -m "feat: add utilities module with logger, device detection, seed, timer"
```

---

### Task 3: Config System (`src/config.py`)

**Files:**
- Create: `finetune-gpt2/src/config.py`

**Interfaces:**
- Consumes: `src/utils.py` → `setup_logger()`
- Produces:
  - Dataclasses: `ModelConfig`, `PeftConfig`, `DataConfig`, `TrainingConfig`, `EarlyStoppingConfig`, `HPSearchConfig`, `LoggingConfig`, `OutputConfig`, `PipelineConfig`
  - `load_config(yaml_path: str, cli_overrides: dict = None) -> PipelineConfig`
  - `save_config(config: PipelineConfig, path: str) -> None`
  - `config_to_dict(config: PipelineConfig) -> dict`

- [ ] **Step 1: Write src/config.py with all dataclasses and config loading**

```python
"""
Config System — Hệ thống cấu hình Pipeline
=============================================
Đọc file YAML config → parse vào dataclasses → validate types.
Hỗ trợ CLI override cho bất kỳ field nào.
"""

# === Standard Library ===
import copy             # Deep copy cho config
import json             # JSON serialization
import os               # Path operations
from dataclasses import dataclass, field, asdict  # Dataclass decorators
from typing import Any, Dict, List, Optional, Tuple  # Type hints

# === Third-Party ===
import yaml             # Đọc/ghi file YAML

# === Local ===
from src.utils import setup_logger  # Logger từ module utils

# Tạo logger cho module config
logger = setup_logger("gpt2_finetune.config")


# ============================================================
# DATACLASS DEFINITIONS — Định nghĩa cấu trúc config
# ============================================================

@dataclass
class ModelConfig:
    """
    Cấu hình Model — chọn model và loại task.

    Attributes:
        name: Tên model trên HuggingFace Hub hoặc đường dẫn local
            Lựa chọn:
              "gpt2"        → 124M params, nhanh, GPU nhỏ (4-8GB VRAM)
              "gpt2-medium" → 355M params, cân bằng (8-16GB VRAM)
              "gpt2-large"  → 774M params, chất lượng cao (16-24GB VRAM)
              "gpt2-xl"     → 1.5B params, tốt nhất (24GB+ VRAM)
            Gợi ý: Bắt đầu với "gpt2" để test, sau đó scale lên

        task_type: Loại task fine-tuning
            Lựa chọn:
              "causal_lm"      → Sinh văn bản tự do (mặc định GPT-2)
              "classification" → Phân loại văn bản (sentiment, topic)
              "completion"     → Instruction fine-tuning, chỉ loss trên response
            Gợi ý: "causal_lm" cho đa số, "classification" nếu có labels

        num_labels: Số lượng labels (chỉ dùng khi task_type="classification")
    """
    name: str = "gpt2"
    task_type: str = "causal_lm"
    num_labels: int = 2


@dataclass
class PeftConfig:
    """
    Cấu hình PEFT/LoRA — Fine-tuning hiệu quả với ít tài nguyên.

    Attributes:
        enabled: Bật/tắt LoRA
            true  → Chỉ train ~0.2-1.5% params, tiết kiệm VRAM, checkpoint nhỏ
            false → Full fine-tuning, cần nhiều VRAM hơn
            Gợi ý: true nếu GPU < 8GB hoặc dataset nhỏ

        r: Rank ma trận LoRA — quyết định "sức mạnh" adapter
            4  → Rất nhỏ, nhanh nhưng khả năng học hạn chế
            8  → Phù hợp task đơn giản hoặc dataset nhỏ
            16 → Cân bằng (khuyến nghị mặc định)
            32 → Phù hợp task phức tạp hoặc dataset lớn
            64 → Gần như full fine-tuning
            Gợi ý: 16, tăng nếu underfit

        lora_alpha: Hệ số scaling, thường = 2 × r
        lora_dropout: Dropout cho LoRA (0.0-0.2), chống overfitting
        target_modules: Layers áp dụng LoRA (GPT-2 dùng Conv1D)
            ["c_attn"]              → Chỉ attention (~0.2% params)
            ["c_attn", "c_proj"]    → + output projection (~0.4%)
            ["c_attn", "c_proj", "c_fc"] → + MLP (~1%)
    """
    enabled: bool = False
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])


@dataclass
class AugmentationConfig:
    """
    Cấu hình Data Augmentation — tăng cường dữ liệu training.

    Attributes:
        enabled: Bật/tắt augmentation
        techniques: Danh sách kỹ thuật augmentation
            "synonym_replace" → Thay từ đồng nghĩa
            "random_delete"   → Xóa từ ngẫu nhiên
            "random_swap"     → Đổi chỗ 2 từ
            "random_insert"   → Chèn từ ngẫu nhiên
        augment_ratio: Tỷ lệ samples được augment (0.0-1.0)
    """
    enabled: bool = False
    techniques: List[str] = field(default_factory=lambda: ["synonym_replace", "random_delete"])
    augment_ratio: float = 0.2


@dataclass
class DataConfig:
    """
    Cấu hình Data — nguồn dữ liệu và cách xử lý.

    Attributes:
        source: Nguồn dataset
            "huggingface" → Tải từ HuggingFace Hub
            "local"       → Đọc file local (CSV/JSON/TXT)
            Gợi ý: "huggingface" cho dataset chuẩn

        dataset_name: Tên dataset trên Hub (khi source="huggingface")
        dataset_config: Config/subset của dataset
        local_path: Đường dẫn file local (khi source="local")
        text_column: Tên cột chứa text
        label_column: Tên cột chứa label (cho classification)

        max_length: Độ dài tối đa sequence sau tokenize
            128  → Ngắn, cho classification
            256  → Trung bình
            512  → Dài, cho text generation (khuyến nghị)
            1024 → Tối đa GPT-2, cần nhiều VRAM

        split_ratios: Tỷ lệ chia [train, val, test]
        augmentation: Cấu hình augmentation
    """
    source: str = "huggingface"
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    local_path: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 512
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class TrainingConfig:
    """
    Cấu hình Training — hyperparameters cho quá trình huấn luyện.

    Attributes:
        num_epochs: Số epoch (3-5 dataset lớn, 5-10 dataset nhỏ)

        batch_size: Batch size trên mỗi device
            4  → GPU 4GB VRAM
            8  → GPU 8GB (khuyến nghị)
            16 → GPU 16GB+
            Gợi ý: Lớn nhất mà GPU chịu được

        gradient_accumulation_steps: Tích lũy gradient qua N steps
            Effective batch = batch_size × grad_accum × n_gpus
            Gợi ý: Tăng khi batch_size phải nhỏ do VRAM

        learning_rate: Tốc độ học
            1e-5 → Bảo thủ (dataset nhỏ)
            5e-5 → Phổ biến cho fine-tuning
            1e-4 → Tích cực (chỉ dùng với LoRA)

        weight_decay: Regularization chống overfitting (0.0-0.1)
        warmup_ratio: % steps đầu tăng dần LR (0.0-0.1)

        lr_scheduler_type: Cách giảm learning rate
            "linear"  → Giảm đều
            "cosine"  → Giảm mượt theo cosine (khuyến nghị)
            "constant" → Giữ nguyên

        mixed_precision: Chế độ precision
            "auto" → Tự chọn bf16/fp16/fp32 (khuyến nghị)
            "bf16" → Bfloat16 (Ampere+, ổn định)
            "fp16" → Float16 (cũ hơn, cần loss scaling)
            "none" → FP32 (chậm nhất)

        seed: Seed cho reproducibility
    """
    num_epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    mixed_precision: str = "auto"


@dataclass
class EarlyStoppingConfig:
    """
    Cấu hình Early Stopping — dừng training khi không cải thiện.

    Attributes:
        enabled: Bật/tắt early stopping
        patience: Số lần eval liên tiếp không cải thiện thì dừng
            3 → Dataset nhỏ (khuyến nghị)
            5 → Dataset lớn
        threshold: Ngưỡng cải thiện tối thiểu (0.0-0.01)
    """
    enabled: bool = True
    patience: int = 3
    threshold: float = 0.001


@dataclass
class HPSearchConfig:
    """
    Cấu hình Optuna Hyperparameter Search.

    Attributes:
        enabled: Bật/tắt HP search
        n_trials: Số trials Optuna chạy (10-50)
        direction: Hướng tối ưu
            "minimize" → Tối thiểu hóa (loss, perplexity)
            "maximize" → Tối đa hóa (accuracy, f1)
        metric: Metric dùng làm objective
        search_space: Khoảng search cho từng hyperparameter
    """
    enabled: bool = False
    n_trials: int = 15
    direction: str = "minimize"
    metric: str = "eval_loss"
    search_space: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": [1e-5, 5e-4],
        "batch_size": [4, 8, 16],
        "gradient_accumulation_steps": [1, 2, 4],
        "weight_decay": [0.0, 0.2],
        "warmup_ratio": [0.0, 0.1],
        "num_epochs": [3, 6],
    })


@dataclass
class LoggingConfig:
    """
    Cấu hình Logging & TensorBoard.

    Attributes:
        level: Mức log ("DEBUG", "INFO", "WARNING", "ERROR")
        logging_steps: Log mỗi N training steps
        tensorboard_dir: Thư mục TensorBoard logs
        experiment_name: Tên experiment (phân biệt trong TensorBoard)
    """
    level: str = "INFO"
    logging_steps: int = 50
    tensorboard_dir: str = "./runs"
    experiment_name: str = "gpt2-finetune"


@dataclass
class OutputConfig:
    """
    Cấu hình Output — nơi lưu checkpoints và model cuối.

    Attributes:
        checkpoint_dir: Thư mục lưu checkpoints training
        model_dir: Thư mục lưu model cuối (best model)
        save_total_limit: Giới hạn số checkpoints trên disk (2-5)

        save_strategy: Chiến lược lưu checkpoint
            "epoch" → Lưu sau mỗi epoch (dataset nhỏ/trung bình)
            "steps" → Lưu mỗi N steps (dataset lớn, training lâu)

        save_steps: Số steps giữa mỗi lần save (khi save_strategy="steps")
    """
    checkpoint_dir: str = "./checkpoints"
    model_dir: str = "./final_model"
    save_total_limit: int = 3
    save_strategy: str = "epoch"
    save_steps: int = 500


@dataclass
class PipelineConfig:
    """
    Config tổng hợp — chứa TẤT CẢ config sections.
    Đây là object chính được truyền vào tất cả các module.
    """
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
        Instance của dc_class đã được populate giá trị
    """
    # Import field metadata để xác định nested dataclasses
    from dataclasses import fields as dc_fields

    # Dict chứa giá trị đã xử lý
    kwargs = {}

    # Duyệt qua từng field trong dataclass
    for f in dc_fields(dc_class):
        # Nếu field có trong data dict
        if f.name in data:
            value = data[f.name]
            # Kiểm tra xem field type có phải là dataclass không
            if hasattr(f.type, "__dataclass_fields__") and isinstance(value, dict):
                # Recursive: chuyển nested dict → nested dataclass
                kwargs[f.name] = _dict_to_dataclass(value, f.type)
            else:
                # Gán giá trị trực tiếp
                kwargs[f.name] = value

    # Tạo instance với giá trị đã xử lý (các field không có trong data → dùng default)
    return dc_class(**kwargs)


def load_config(yaml_path: str, cli_overrides: dict = None) -> PipelineConfig:
    """
    Load config từ file YAML, merge với CLI overrides, validate.

    Args:
        yaml_path: Đường dẫn tới file YAML config
        cli_overrides: Dict các override từ CLI
            VD: {"training.batch_size": 4, "peft.enabled": True}

    Returns:
        PipelineConfig đã validate và sẵn sàng sử dụng

    Raises:
        FileNotFoundError: Nếu file YAML không tồn tại
        yaml.YAMLError: Nếu file YAML có syntax error
        ValueError: Nếu giá trị config không hợp lệ
    """
    # Kiểm tra file YAML có tồn tại không
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file không tìm thấy: {yaml_path}")

    # Đọc file YAML
    logger.info(f"Đọc config từ: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        # yaml.safe_load: parse YAML thành dict Python (an toàn, không execute code)
        raw_config = yaml.safe_load(f) or {}

    # Áp dụng CLI overrides (nếu có)
    if cli_overrides:
        logger.info(f"Áp dụng {len(cli_overrides)} CLI overrides")
        for dotted_key, value in cli_overrides.items():
            # Parse dotted key: "training.batch_size" → ["training", "batch_size"]
            keys = dotted_key.split(".")
            # Navigate vào nested dict
            target = raw_config
            for key in keys[:-1]:
                # Tạo dict nếu chưa có (auto-create nested structure)
                target = target.setdefault(key, {})
            # Gán giá trị override
            target[keys[-1]] = value
            logger.debug(f"  Override: {dotted_key} = {value}")

    # Chuyển dict → PipelineConfig dataclass (với nested dataclasses)
    config = _dict_to_dataclass(raw_config, PipelineConfig)

    # Validate config
    _validate_config(config)

    logger.info("✅ Config loaded và validated thành công")
    return config


def _validate_config(config: PipelineConfig) -> None:
    """
    Validate config — kiểm tra giá trị hợp lệ, raise ValueError nếu không.

    Checks:
        - task_type phải là một trong các giá trị cho phép
        - mixed_precision phải hợp lệ
        - save_strategy phải hợp lệ
        - split_ratios phải tổng bằng 1.0
        - learning_rate > 0
        - batch_size > 0
    """
    # Kiểm tra task_type hợp lệ
    valid_tasks = {"causal_lm", "classification", "completion"}
    if config.model.task_type not in valid_tasks:
        raise ValueError(
            f"task_type '{config.model.task_type}' không hợp lệ. "
            f"Các lựa chọn: {valid_tasks}"
        )

    # Kiểm tra mixed_precision hợp lệ
    valid_precisions = {"auto", "bf16", "fp16", "none"}
    if config.training.mixed_precision not in valid_precisions:
        raise ValueError(
            f"mixed_precision '{config.training.mixed_precision}' không hợp lệ. "
            f"Các lựa chọn: {valid_precisions}"
        )

    # Kiểm tra save_strategy hợp lệ
    valid_strategies = {"epoch", "steps"}
    if config.output.save_strategy not in valid_strategies:
        raise ValueError(
            f"save_strategy '{config.output.save_strategy}' không hợp lệ. "
            f"Các lựa chọn: {valid_strategies}"
        )

    # Kiểm tra split_ratios tổng ≈ 1.0 (cho phép sai số nhỏ do floating point)
    ratio_sum = sum(config.data.split_ratios)
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(
            f"split_ratios tổng = {ratio_sum}, phải bằng 1.0. "
            f"Giá trị hiện tại: {config.data.split_ratios}"
        )

    # Kiểm tra giá trị dương
    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate phải > 0, hiện tại: {config.training.learning_rate}")
    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size phải > 0, hiện tại: {config.training.batch_size}")
    if config.training.num_epochs <= 0:
        raise ValueError(f"num_epochs phải > 0, hiện tại: {config.training.num_epochs}")

    # Kiểm tra source hợp lệ
    valid_sources = {"huggingface", "local"}
    if config.data.source not in valid_sources:
        raise ValueError(
            f"data.source '{config.data.source}' không hợp lệ. "
            f"Các lựa chọn: {valid_sources}"
        )

    # Kiểm tra local_path khi source="local"
    if config.data.source == "local" and not config.data.local_path:
        raise ValueError("data.local_path phải được chỉ định khi data.source='local'")

    # Kiểm tra HP search direction hợp lệ
    valid_directions = {"minimize", "maximize"}
    if config.hp_search.direction not in valid_directions:
        raise ValueError(
            f"hp_search.direction '{config.hp_search.direction}' không hợp lệ. "
            f"Các lựa chọn: {valid_directions}"
        )

    logger.debug("Config validation passed")


def save_config(config: PipelineConfig, path: str) -> None:
    """
    Lưu config thành file YAML — dùng cho reproducibility.

    Args:
        config: PipelineConfig cần lưu
        path: Đường dẫn file YAML đích
    """
    # Chuyển dataclass → dict
    config_dict = asdict(config)
    # Tạo thư mục cha nếu chưa có
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    # Ghi ra file YAML
    with open(path, "w", encoding="utf-8") as f:
        # allow_unicode=True: cho phép ký tự Unicode (tiếng Việt)
        # default_flow_style=False: format YAML dạng block (dễ đọc)
        # sort_keys=False: giữ nguyên thứ tự keys
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info(f"Config đã lưu tại: {path}")


def config_to_dict(config: PipelineConfig) -> dict:
    """
    Chuyển PipelineConfig thành dict phẳng — dùng cho banner, logging.

    Returns:
        Dict với các key chính: model_name, task_type, peft_enabled, ...
    """
    return {
        "model_name": config.model.name,
        "task_type": config.model.task_type,
        "peft_enabled": config.peft.enabled,
        "peft_r": config.peft.r,
        "peft_target": ", ".join(config.peft.target_modules),
    }
```

- [ ] **Step 2: Verify config module loads**

Run:
```bash
cd finetune-gpt2 && python -c "
from src.config import PipelineConfig, load_config, save_config, config_to_dict
# Test default config
cfg = PipelineConfig()
print(f'Model: {cfg.model.name}, Task: {cfg.model.task_type}')
print(f'LoRA: {cfg.peft.enabled}, LR: {cfg.training.learning_rate}')
print('✅ config module OK')
"
```
Expected: prints config values and `✅ config module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/config.py
git commit -m "feat: add config system with YAML + dataclass validation"
```

---

### Task 4: YAML Config Files

**Files:**
- Create: `finetune-gpt2/configs/default.yaml`
- Create: `finetune-gpt2/configs/classification.yaml`
- Create: `finetune-gpt2/configs/lora.yaml`

**Interfaces:**
- Consumes: `src/config.py` → `PipelineConfig` schema
- Produces: 3 YAML config files with comprehensive Vietnamese comments

- [ ] **Step 1: Write configs/default.yaml**

Full YAML config for causal language modeling with Vietnamese comments on every line explaining all options and suggestions. Content as specified in the approved Design Section 1 of the spec (the full YAML with `model`, `peft`, `data`, `training`, `early_stopping`, `hp_search`, `logging`, `output` sections — every field commented in Vietnamese with all option values, their meanings, and recommendations).

- [ ] **Step 2: Write configs/classification.yaml**

YAML config for text classification (sentiment analysis) with:
- `model.task_type: "classification"`, `model.num_labels: 3`
- `data.source: "huggingface"`, `data.dataset_name: "financial_phrasebank"`, `data.dataset_config: "sentences_allagree"`
- `data.max_length: 256`
- `training.learning_rate: 2e-5`, `training.batch_size: 16`
- Full Vietnamese comments

- [ ] **Step 3: Write configs/lora.yaml**

YAML config for LoRA fine-tuning with:
- `peft.enabled: true`, `peft.r: 16`, `peft.lora_alpha: 32`, `peft.target_modules: ["c_attn", "c_proj"]`
- `training.learning_rate: 1e-4`
- Full Vietnamese comments

- [ ] **Step 4: Verify all configs load successfully**

Run:
```bash
cd finetune-gpt2 && python -c "
from src.config import load_config
for name in ['default', 'classification', 'lora']:
    cfg = load_config(f'configs/{name}.yaml')
    print(f'✅ {name}.yaml → model={cfg.model.name}, task={cfg.model.task_type}, lora={cfg.peft.enabled}')
"
```
Expected: all 3 configs load and print correctly

- [ ] **Step 5: Commit**

```bash
git add finetune-gpt2/configs/
git commit -m "feat: add YAML config files with Vietnamese documentation"
```

---

### Task 5: Data Pipeline (`src/data.py`)

**Files:**
- Create: `finetune-gpt2/src/data.py`

**Interfaces:**
- Consumes:
  - `src/config.py` → `PipelineConfig`, `DataConfig`, `AugmentationConfig`
  - `src/utils.py` → `setup_logger()`, `StepTimer`
- Produces:
  - `prepare_data(config: PipelineConfig) -> tuple[DatasetDict, Any, PreTrainedTokenizerBase]`
    - Returns: (datasets with "train"/"validation"/"test" keys, data_collator, tokenizer)

- [ ] **Step 1: Write src/data.py with complete data pipeline**

Complete module with Vietnamese comments implementing:
- `_load_tokenizer(config)` → load tokenizer, set pad_token = eos_token, set padding_side="left" for classification
- `_load_raw_data(config)` → load from HuggingFace Hub or local (csv/json/text)
- `_split_dataset(dataset, ratios)` → auto-split if no pre-existing splits
- `_filter_and_clean(dataset, text_column)` → remove empty/too-short samples
- `_augment_data(dataset, config)` → simple text augmentation (random_delete, random_swap)
- `_tokenize_dataset(dataset, tokenizer, config)` → tokenize with truncation=True, NO static padding
- `_get_data_collator(tokenizer, config)` → return correct collator per task_type
- `prepare_data(config)` → orchestrate all steps, return (datasets, collator, tokenizer)

Key implementation details:
- Dynamic padding via DataCollator (NOT pad in tokenize step)
- For classification: `tokenizer.padding_side = "left"` (GPT-2 classifies on last token)
- For completion: use `DataCollatorForCompletionOnlyLM` from `trl`
- Filter empty texts: `dataset.filter(lambda x: len(x[text_column].strip()) > 10)`
- Simple augmentation without nlpaug dependency: random word deletion, random word swap

- [ ] **Step 2: Verify data module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.data import prepare_data; print('✅ data module OK')"
```
Expected: `✅ data module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/data.py
git commit -m "feat: add data pipeline with HuggingFace/local loading, augmentation, tokenization"
```

---

### Task 6: Model Factory (`src/model.py`)

**Files:**
- Create: `finetune-gpt2/src/model.py`

**Interfaces:**
- Consumes:
  - `src/config.py` → `PipelineConfig`, `ModelConfig`, `PeftConfig`
  - `src/utils.py` → `setup_logger()`
- Produces:
  - `create_model_init(config: PipelineConfig) -> Callable[[Optional[Any]], PreTrainedModel]`
  - `load_model_for_inference(model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`

- [ ] **Step 1: Write src/model.py with model factory**

Complete module with Vietnamese comments implementing:
- `create_model_init(config)` → returns `model_init(trial=None)` callable
  - task_type "causal_lm"/"completion" → `GPT2LMHeadModel.from_pretrained(name)`
  - task_type "classification" → `GPT2ForSequenceClassification.from_pretrained(name, num_labels=N)`
  - Always set `model.config.pad_token_id = model.config.eos_token_id`
  - If peft.enabled: wrap with `get_peft_model(model, LoraConfig(...))`
  - If trial is not None (Optuna mode): `lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])`
  - Log trainable parameters with `model.print_trainable_parameters()` (for PEFT) or manual count
- `load_model_for_inference(model_path)` → load saved model
  - Check if `adapter_config.json` exists → LoRA model → load base + adapter + merge_and_unload()
  - Otherwise → load full model directly
  - Load tokenizer from same path

- [ ] **Step 2: Verify model module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.model import create_model_init, load_model_for_inference; print('✅ model module OK')"
```
Expected: `✅ model module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/model.py
git commit -m "feat: add model factory with full/LoRA/classification support"
```

---

### Task 7: Custom Callbacks (`src/callbacks.py`)

**Files:**
- Create: `finetune-gpt2/src/callbacks.py`

**Interfaces:**
- Consumes: `src/utils.py` → `setup_logger()`, `format_duration()`, `format_metrics_table()`
- Produces:
  - `class RichLoggingCallback(TrainerCallback)` — beautiful console output
  - `class GenerationSampleCallback(TrainerCallback)` — generate sample text each epoch
  - `class TensorBoardMetricsCallback(TrainerCallback)` — log perplexity to TensorBoard

- [ ] **Step 1: Write src/callbacks.py with 3 callback classes**

Complete module with Vietnamese comments implementing:
- `RichLoggingCallback`:
  - `on_train_begin`: log banner with total steps, epochs
  - `on_log`: format metrics in step logging (train_loss, learning_rate)
  - `on_epoch_end`: print epoch summary table (train_loss, eval_loss, perplexity)
  - `on_train_end`: print final summary (total time, best metric)
- `GenerationSampleCallback(sample_prompt, tokenizer)`:
  - `on_epoch_end`: generate text from sample_prompt using `model.generate()` with `temperature=0.7, max_new_tokens=50, do_sample=True, pad_token_id=tokenizer.eos_token_id`
  - Only active when task_type in ["causal_lm", "completion"]
- `TensorBoardMetricsCallback(log_dir)`:
  - `on_evaluate`: compute perplexity = `math.exp(eval_loss)`, write to TensorBoard via `SummaryWriter`
  - `on_train_end`: close writer

- [ ] **Step 2: Verify callbacks module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.callbacks import RichLoggingCallback, GenerationSampleCallback, TensorBoardMetricsCallback; print('✅ callbacks module OK')"
```
Expected: `✅ callbacks module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/callbacks.py
git commit -m "feat: add custom callbacks for logging, generation sampling, TensorBoard"
```

---

### Task 8: HP Search (`src/hp_search.py`)

**Files:**
- Create: `finetune-gpt2/src/hp_search.py`

**Interfaces:**
- Consumes:
  - `src/config.py` → `PipelineConfig`, `HPSearchConfig`
  - `src/utils.py` → `setup_logger()`, `format_metrics_table()`
- Produces:
  - `build_hp_space(config: PipelineConfig) -> Callable[[optuna.Trial], dict]`
  - `run_hp_search(config: PipelineConfig, trainer: Trainer) -> dict`
    - Returns: dict of best hyperparameters

- [ ] **Step 1: Write src/hp_search.py with Optuna integration**

Complete module with Vietnamese comments implementing:
- `build_hp_space(config)` → returns function `hp_space(trial) -> dict`
  - Build search ranges from `config.hp_search.search_space`
  - `learning_rate` → `trial.suggest_float("learning_rate", min, max, log=True)`
  - `batch_size` → `trial.suggest_categorical("per_device_train_batch_size", choices)`
  - `gradient_accumulation_steps` → `trial.suggest_categorical(...)`
  - `weight_decay` → `trial.suggest_float("weight_decay", min, max)`
  - `warmup_ratio` → `trial.suggest_float("warmup_ratio", min, max)`
  - `num_epochs` → `trial.suggest_int("num_train_epochs", min, max)`
- `run_hp_search(config, trainer)` → run Optuna via `trainer.hyperparameter_search()`
  - sampler: `TPESampler(seed=config.training.seed)`
  - pruner: `MedianPruner(n_startup_trials=3, n_warmup_steps=1)`
  - Log each trial result
  - Log best trial with formatted table
  - Return `best_run.hyperparameters`

- [ ] **Step 2: Verify hp_search module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.hp_search import build_hp_space, run_hp_search; print('✅ hp_search module OK')"
```
Expected: `✅ hp_search module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/hp_search.py
git commit -m "feat: add Optuna hyperparameter search integration"
```

---

### Task 9: Training Orchestration (`src/trainer.py`)

**Files:**
- Create: `finetune-gpt2/src/trainer.py`

**Interfaces:**
- Consumes:
  - `src/config.py` → `PipelineConfig`
  - `src/utils.py` → `setup_logger()`, `detect_device()`
  - `src/callbacks.py` → `RichLoggingCallback`, `GenerationSampleCallback`, `TensorBoardMetricsCallback`
  - `src/hp_search.py` → `run_hp_search()`, `build_hp_space()`
- Produces:
  - `run_training(config: PipelineConfig, datasets: DatasetDict, data_collator: Any, tokenizer: PreTrainedTokenizerBase, model_init: Callable, best_hp: dict = None) -> tuple[Trainer, dict]`
    - Returns: (trained Trainer instance, final training metrics dict)

- [ ] **Step 1: Write src/trainer.py with training orchestration**

Complete module with Vietnamese comments implementing:
- `_build_training_args(config, best_hp)` → construct `TrainingArguments` from config
  - Mixed precision auto-detect: `bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()`
  - `eval_strategy = save_strategy` (mandatory when `load_best_model_at_end=True`)
  - `report_to=["tensorboard"]`
  - `logging_dir = f"{config.logging.tensorboard_dir}/{config.logging.experiment_name}"`
  - Apply `best_hp` overrides if provided
- `_build_callbacks(config, tokenizer)` → list of callbacks
  - Always: `RichLoggingCallback`
  - If early_stopping.enabled: `EarlyStoppingCallback(patience, threshold)`
  - If task_type in ["causal_lm", "completion"]: `GenerationSampleCallback`
  - Always: `TensorBoardMetricsCallback`
- `_build_compute_metrics(config)` → return compute_metrics function or None
  - For classification: return function using `evaluate.load("accuracy")` and `evaluate.load("f1")`
  - For causal_lm/completion: return None (perplexity computed separately)
- `run_training(config, datasets, data_collator, tokenizer, model_init, best_hp)`:
  - Build TrainingArguments, callbacks, compute_metrics
  - If hp_search enabled AND best_hp is None: run HP search first, get best_hp
  - Create Trainer with model_init (NOT model instance)
  - Call `trainer.train()`
  - Return (trainer, train_result.metrics)

- [ ] **Step 2: Verify trainer module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.trainer import run_training; print('✅ trainer module OK')"
```
Expected: `✅ trainer module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/trainer.py
git commit -m "feat: add training orchestration with auto-config, callbacks, HP search"
```

---

### Task 10: Evaluation Module (`src/evaluate.py`)

**Files:**
- Create: `finetune-gpt2/src/evaluate.py`

**Interfaces:**
- Consumes:
  - `src/config.py` → `PipelineConfig`
  - `src/utils.py` → `setup_logger()`, `format_metrics_table()`
- Produces:
  - `run_evaluation(trainer: Trainer, test_dataset: Dataset, config: PipelineConfig) -> dict`
  - `generate_evaluation_report(metrics: dict, config: PipelineConfig) -> str`

- [ ] **Step 1: Write src/evaluate.py with evaluation logic**

Complete module with Vietnamese comments implementing:
- `run_evaluation(trainer, test_dataset, config)`:
  - Call `trainer.evaluate(eval_dataset=test_dataset)`
  - Compute perplexity: `math.exp(metrics["eval_loss"])`
  - For classification: metrics already include accuracy/f1 from compute_metrics
  - For causal_lm/completion: add perplexity to metrics
  - Return metrics dict
- `generate_evaluation_report(metrics, config)`:
  - Format metrics into beautiful console output using `format_metrics_table()`
  - Include step header: `📊 [Step N/N] Evaluation (Test Set)`
  - Show all metrics with aligned formatting
  - Return formatted string

- [ ] **Step 2: Verify evaluate module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.evaluate import run_evaluation, generate_evaluation_report; print('✅ evaluate module OK')"
```
Expected: `✅ evaluate module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/evaluate.py
git commit -m "feat: add evaluation module with perplexity, accuracy, report generation"
```

---

### Task 11: Inference Module (`src/inference.py`)

**Files:**
- Create: `finetune-gpt2/src/inference.py`

**Interfaces:**
- Consumes:
  - `src/config.py` → `PipelineConfig`, `save_config()`
  - `src/utils.py` → `setup_logger()`
- Produces:
  - `save_model(trainer: Trainer, tokenizer: PreTrainedTokenizerBase, config: PipelineConfig, metrics: dict) -> str`
  - `load_model(model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`
  - `generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str], **kwargs) -> list[str]`

- [ ] **Step 1: Write src/inference.py with save/load/generate**

Complete module with Vietnamese comments implementing:
- `save_model(trainer, tokenizer, config, metrics)`:
  - Create output directory `config.output.model_dir`
  - `trainer.save_model(model_dir)` → save model weights
  - `tokenizer.save_pretrained(model_dir)` → save tokenizer
  - Save `GenerationConfig` with sensible defaults (temperature=0.7, top_p=0.9, repetition_penalty=1.1)
  - `save_config(config, os.path.join(model_dir, "training_config.yaml"))` → reproducibility
  - Save metrics to `training_metrics.json`
  - Log saved files list and total size
  - Return model_dir path
- `load_model(model_path)`:
  - Check for `adapter_config.json` → LoRA adapter
  - If LoRA: load base model from adapter_config, load adapter, `merge_and_unload()`
  - If full: `AutoModelForCausalLM.from_pretrained(model_path)` or `AutoModelForSequenceClassification`
  - Detect model type from `config.json`
  - Load tokenizer
  - Return (model, tokenizer)
- `generate_text(model, tokenizer, prompts, **kwargs)`:
  - Default kwargs: `max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id`
  - Merge user kwargs
  - Tokenize prompts, generate, decode
  - Return list of generated texts

- [ ] **Step 2: Verify inference module imports**

Run:
```bash
cd finetune-gpt2 && python -c "from src.inference import save_model, load_model, generate_text; print('✅ inference module OK')"
```
Expected: `✅ inference module OK`

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/src/inference.py
git commit -m "feat: add inference module with save/load/generate"
```

---

### Task 12: CLI Entry Point (`main.py`)

**Files:**
- Create: `finetune-gpt2/main.py`

**Interfaces:**
- Consumes: ALL modules (config, utils, data, model, trainer, evaluate, inference, hp_search)
- Produces: CLI entry point with `--config`, `--step`, `--prompts`, and dotted config overrides

- [ ] **Step 1: Write main.py with CLI argument parsing and pipeline orchestration**

Complete module with Vietnamese comments implementing:
- `parse_args()` → argparse with:
  - `--config` (required): path to YAML config
  - `--step` (optional): run specific step only ("data", "hp_search", "train", "evaluate", "generate")
  - `--prompts` (optional): text prompts for generation step
  - All remaining args collected as config overrides (dotted notation)
- `run_pipeline(config, step)`:
  - Full pipeline sequence with StepTimer for each step
  - Step 1: `prepare_data(config)` → datasets, collator, tokenizer
  - Step 2: `create_model_init(config)` → model_init
  - Step 3 (optional): `run_hp_search()` → best_hp
  - Step 4: `run_training()` → trainer, metrics
  - Step 5: `run_evaluation()` → test_metrics
  - Step 6: `save_model()` → saved path
  - Print final summary with TensorBoard command
- `if __name__ == "__main__":` block:
  - Parse args, load config with overrides
  - Setup seed, detect device, print banner
  - Run pipeline or specific step
  - Exception handling with helpful error messages

- [ ] **Step 2: Verify CLI help works**

Run:
```bash
cd finetune-gpt2 && python main.py --help
```
Expected: prints usage with all arguments

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/main.py
git commit -m "feat: add CLI entry point with step selection and config overrides"
```

---

### Task 13: Experiment Comparison (`compare.py`)

**Files:**
- Create: `finetune-gpt2/compare.py`

**Interfaces:**
- Consumes: `src/utils.py` → `setup_logger()`, `format_metrics_table()`
- Produces: CLI tool `python compare.py <dir1> <dir2> [dir3...]`

- [ ] **Step 1: Write compare.py**

Complete module with Vietnamese comments implementing:
- Load `training_metrics.json` and `training_config.yaml` from each experiment directory
- Build comparison table: model name, task, LoRA status, train_loss, eval_loss, perplexity, accuracy (if available), training time, model size
- Print formatted table with aligned columns and highlighting for best values
- Support 2+ experiments

- [ ] **Step 2: Verify compare.py runs**

Run:
```bash
cd finetune-gpt2 && python compare.py --help
```
Expected: prints usage

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/compare.py
git commit -m "feat: add experiment comparison tool"
```

---

### Task 14: README.md Tutorial with 8 Production Use Cases

**Files:**
- Create: `finetune-gpt2/README.md`

**Interfaces:**
- Consumes: All config files, CLI interface
- Produces: Comprehensive tutorial document

- [ ] **Step 1: Write README.md**

Complete README in Vietnamese with:
- **Giới thiệu & Tính năng**: Overview, feature list
- **Cài đặt**: pip install, environment setup
- **Quick Start**: 5-minute guide with `configs/default.yaml`
- **8 Production Use Cases** (each with: mô tả, dataset, config YAML mẫu, lệnh chạy, metrics kỳ vọng, lỗi thường gặp & cách fix, tips):
  - UC1: Customer Service Chatbot (DialoGPT, bitext dataset)
  - UC2: Marketing Content Generation (Amazon reviews, completion)
  - UC3: Code Generation (CodeGPT, code_search_net)
  - UC4: Sentiment Analysis (financial_phrasebank, classification, padding_side="left")
  - UC5: Domain-Specific Legal/Medical (pile-of-law, LoRA, low LR)
  - UC6: Vietnamese Language Tasks (NlpHUST/gpt2-vietnamese, BKAINewsCorpus)
  - UC7: Creative Writing (TinyStories, decoding params)
  - UC8: Data Extraction/JSON (web_nlg, constrained decoding)
- **Troubleshooting Guide**: Universal error table (7 common errors)
- **Bảng tham khảo Hyperparameters**: By model size
- **CLI Reference**: Full command reference
- **FAQ**: Common questions

- [ ] **Step 2: Verify README renders correctly**

Run:
```bash
wc -l finetune-gpt2/README.md
head -50 finetune-gpt2/README.md
```
Expected: README exists with substantial content (500+ lines)

- [ ] **Step 3: Commit**

```bash
git add finetune-gpt2/README.md
git commit -m "docs: add comprehensive tutorial README with 8 production use cases"
```

---

### Task 15: Integration Verification

**Files:**
- No new files (verification only)

**Interfaces:**
- Consumes: All modules
- Produces: Verified working pipeline

- [ ] **Step 1: Verify all module imports work together**

Run:
```bash
cd finetune-gpt2 && python -c "
from src.config import load_config, PipelineConfig
from src.utils import setup_logger, detect_device, set_seed, StepTimer, format_banner
from src.data import prepare_data
from src.model import create_model_init, load_model_for_inference
from src.hp_search import build_hp_space, run_hp_search
from src.trainer import run_training
from src.evaluate import run_evaluation, generate_evaluation_report
from src.inference import save_model, load_model, generate_text
from src.callbacks import RichLoggingCallback, GenerationSampleCallback, TensorBoardMetricsCallback
print('✅ All modules import successfully')

# Test config loading
for name in ['default', 'classification', 'lora']:
    cfg = load_config(f'configs/{name}.yaml')
    print(f'  ✅ {name}.yaml loaded: {cfg.model.name}/{cfg.model.task_type}')

# Test device detection
info = detect_device()
print(f'  ✅ Device: {info[\"device\"]} ({info[\"n_gpus\"]} GPUs)')

# Test banner
banner = format_banner({\"model_name\": \"gpt2\", \"task_type\": \"causal_lm\", \"peft_enabled\": False}, info)
print(banner)

print('\\n✅ Integration verification PASSED')
"
```
Expected: all imports succeed, configs load, banner displays

- [ ] **Step 2: Verify CLI help**

Run:
```bash
cd finetune-gpt2 && python main.py --help
```
Expected: Shows all CLI options

- [ ] **Step 3: Final commit with updated .gitignore**

Create/update `.gitignore` in `finetune-gpt2/` to ignore:
```
# Checkpoints & models
checkpoints/
final_model/
*.safetensors
*.bin

# TensorBoard logs
runs/

# Python cache
__pycache__/
*.pyc

# Optuna
*.db
```

```bash
git add finetune-gpt2/.gitignore
git commit -m "chore: add .gitignore for checkpoints, models, TensorBoard logs"
```
