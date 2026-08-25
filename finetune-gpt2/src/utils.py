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
