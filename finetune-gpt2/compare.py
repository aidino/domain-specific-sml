"""
Compare Module — Công cụ so sánh kết quả giữa nhiều experiments
================================================================
Đọc kết quả huấn luyện (training_metrics.json, training_config.yaml) từ các
thư mục experiments, xây dựng bảng so sánh chi tiết và làm nổi bật kết quả tốt nhất.
"""

# === Standard Library ===
import argparse  # Thư viện phân tích tham số dòng lệnh CLI
import json  # Thư viện đọc và phân tích dữ liệu định dạng JSON
import math  # Thư viện toán học phục vụ tính toán perplexity và kiểm tra số thực
import os  # Thư viện thao tác với hệ điều hành và đường dẫn tập tin
import re  # Thư viện regular expression hỗ trợ xử lý chuỗi ANSI
import sys  # Thư viện hệ thống quản lý sys.path và luồng thực thi
from dataclasses import dataclass, field  # Cung cấp cấu trúc dataclass lưu trữ dữ liệu
from typing import Any, Dict, List, Optional, Tuple  # Type hints cho kiểu dữ liệu Python 3.10+

# Thêm thư mục hiện tại vào sys.path để đảm bảo import được package src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# === Third-Party ===
import yaml  # Thư viện đọc và phân tích tập tin YAML cấu hình

# === Local ===
from src.utils import format_duration, format_metrics_table, setup_logger  # Tiện ích định dạng và ghi log

# Khởi tạo logger riêng cho module compare
logger = setup_logger("gpt2_finetune.compare")


def visible_len(text: str) -> int:
    """
    Tính độ dài hiển thị thực tế của chuỗi ký tự bằng cách loại bỏ ANSI escape codes.
    
    Args:
        text (str): Chuỗi ký tự có thể chứa mã màu ANSI.
        
    Returns:
        int: Số lượng ký tự hiển thị thực tế trên màn hình terminal.
    """
    # Sử dụng biểu thức chính quy xóa toàn bộ mã màu ANSI dạng \033[...m
    clean_text = re.sub(r"\033\[[0-9;]*m", "", text)
    # Trả về độ dài chuỗi ký tự sau khi đã lọc mã màu
    return len(clean_text)


def pad_visible(text: str, width: int, align: str = "left") -> str:
    """
    Căn lề chuỗi ký tự theo độ rộng hiển thị mong muốn, bỏ qua độ dài mã ANSI.
    
    Args:
        text (str): Chuỗi ký tự nguồn.
        width (int): Độ rộng mục tiêu sau khi căn lề.
        align (str): Hướng căn lề: 'left', 'right', hoặc 'center'.
        
    Returns:
        str: Chuỗi ký tự đã được chèn khoảng trắng căn lề chuẩn xác.
    """
    # Lấy độ dài hiển thị thực tế của chuỗi
    vlen = visible_len(text)
    # Tính số khoảng trắng cần bổ sung để đạt độ rộng yêu cầu
    padding_needed = max(0, width - vlen)
    # Xử lý trường hợp căn phải
    if align == "right":
        # Thêm khoảng trắng vào phía trước chuỗi
        return " " * padding_needed + text
    # Xử lý trường hợp căn giữa
    elif align == "center":
        # Chia đôi khoảng trắng cho bên trái
        left_pad = padding_needed // 2
        # Phần khoảng trắng còn lại cho bên phải
        right_pad = padding_needed - left_pad
        # Ghép chuỗi hoàn chỉnh căn giữa
        return " " * left_pad + text + " " * right_pad
    # Mặc định xử lý trường hợp căn trái
    else:
        # Thêm khoảng trắng vào phía sau chuỗi
        return text + " " * padding_needed


def format_size(size_bytes: int) -> str:
    """
    Chuyển đổi kích thước dung lượng từ đơn vị bytes sang chuỗi định dạng dễ đọc.
    
    Args:
        size_bytes (int): Số lượng bytes cần chuyển đổi.
        
    Returns:
        str: Chuỗi dung lượng đã định dạng (VD: '487 MB', '1.42 GB').
    """
    # Nếu dung lượng nhỏ hơn hoặc bằng 0 trả về 0 B
    if size_bytes <= 0:
        # Trả về chuỗi mặc định 0 B
        return "0 B"
    # Danh sách các tiền tố đơn vị đo dung lượng
    units = ["B", "KB", "MB", "GB", "TB"]
    # Tính bậc logarithm cơ số 1024 để tìm đơn vị phù hợp
    order = min(int(math.log(size_bytes, 1024)), len(units) - 1)
    # Quy đổi số bytes sang đơn vị tương ứng
    scaled_size = size_bytes / (1024 ** order)
    # Nếu đơn vị là Bytes thì hiển thị số nguyên
    if order == 0:
        # Định dạng số nguyên cho đơn vị B
        return f"{int(scaled_size)} {units[order]}"
    # Nếu dung lượng lớn hơn thì hiển thị 2 chữ số thập phân
    elif scaled_size >= 100 or order == 1:
        # Định dạng làm tròn 1 chữ số thập phân cho MB/KB lớn
        return f"{scaled_size:.1f} {units[order]}"
    # Định dạng 2 chữ số thập phân cho đơn vị GB hoặc giá trị nhỏ
    else:
        # Định dạng làm tròn 2 chữ số thập phân
        return f"{scaled_size:.2f} {units[order]}"


def get_directory_model_size(dir_path: str) -> Tuple[int, bool]:
    """
    Quét thư mục để tính tổng dung lượng trọng số mô hình và nhận diện LoRA adapter.
    
    Args:
        dir_path (str): Đường dẫn tới thư mục experiment hoặc checkpoint.
        
    Returns:
        Tuple[int, bool]: (Tổng số bytes của các file trọng số, True nếu là LoRA adapter).
    """
    # Khởi tạo tổng dung lượng ban đầu bằng 0
    total_bytes = 0
    # Cờ đánh dấu có phải là LoRA adapter hay không
    is_adapter = False
    # Kiểm tra sự tồn tại của thư mục
    if not os.path.exists(dir_path):
        # Trả về dung lượng 0 và False nếu thư mục không tồn tại
        return 0, False
    
    # Tập hợp các đuôi file chứa trọng số mô hình
    weight_extensions = {".safetensors", ".bin", ".pt", ".h5", ".pth"}
    # Tập hợp các tên file đặc trưng cho LoRA adapter
    adapter_filenames = {"adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"}
    
    # Duyệt đệ quy qua toàn bộ cây thư mục
    for root, _, files in os.walk(dir_path):
        # Duyệt qua từng file trong thư mục hiện tại
        for filename in files:
            # Đường dẫn tuyệt đối tới file
            file_path = os.path.join(root, filename)
            # Kiểm tra nếu tên file là file đặc trưng của adapter
            if filename in adapter_filenames:
                # Đánh dấu đây là LoRA adapter
                is_adapter = True
            # Tách phần mở rộng của file
            _, ext = os.path.splitext(filename)
            # Kiểm tra nếu file là file trọng số hoặc adapter
            if ext.lower() in weight_extensions or filename in adapter_filenames:
                # Thử lấy dung lượng file an toàn
                try:
                    # Cộng dồn dung lượng file vào tổng dung lượng
                    total_bytes += os.path.getsize(file_path)
                # Bắt lỗi nếu file bị khóa hoặc không truy cập được
                except OSError:
                    # Bỏ qua lỗi truy cập file
                    pass
                    
    # Nếu không tìm thấy file trọng số cụ thể nào, quét toàn bộ file trong thư mục
    if total_bytes == 0:
        # Duyệt lại toàn bộ thư mục
        for root, _, files in os.walk(dir_path):
            # Duyệt qua từng file
            for filename in files:
                # Lấy đường dẫn file
                file_path = os.path.join(root, filename)
                # Thử lấy kích thước file
                try:
                    # Cộng dồn kích thước file
                    total_bytes += os.path.getsize(file_path)
                # Bỏ qua lỗi nếu có
                except OSError:
                    # Bỏ qua ngoại lệ
                    pass

    # Trả về cặp giá trị tổng dung lượng và cờ adapter
    return total_bytes, is_adapter


@dataclass
class ExperimentData:
    """
    Dataclass lưu trữ toàn bộ thông tin và chỉ số đo lường của một experiment.
    """
    # Tên định danh của experiment
    name: str
    # Đường dẫn thư mục của experiment
    dir_path: str
    # Tên mô hình gốc (VD: gpt2, gpt2-medium)
    model_name: str = "gpt2"
    # Chuỗi hiển thị tên model hoàn chỉnh (VD: gpt2 + LoRA r=16)
    model_display_name: str = "gpt2"
    # Loại tác vụ fine-tuning (causal_lm, classification, completion)
    task_type: str = "causal_lm"
    # Trạng thái LoRA (enabled (r=16) hoặc disabled)
    lora_status: str = "disabled"
    # Rank LoRA r nếu có
    lora_r: int = 16
    # Giá trị train loss
    train_loss: Optional[float] = None
    # Giá trị eval loss
    eval_loss: Optional[float] = None
    # Giá trị perplexity
    perplexity: Optional[float] = None
    # Giá trị accuracy (nếu có cho bài toán phân loại)
    accuracy: Optional[float] = None
    # Giá trị f1 score (nếu có)
    f1: Optional[float] = None
    # Thời gian huấn luyện tính bằng giây
    training_time: Optional[float] = None
    # Chuỗi hiển thị thời gian huấn luyện định dạng đẹp
    training_time_str: str = "N/A"
    # Chuỗi hiển thị kích thước mô hình
    model_size_str: str = "N/A"
    # Chuỗi hiển thị epoch tốt nhất (VD: 4/5)
    best_epoch_str: str = "N/A"
    # Toàn bộ metrics thô đọc từ JSON
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    # Toàn bộ config thô đọc từ YAML
    raw_config: Dict[str, Any] = field(default_factory=dict)


def load_experiment_data(exp_dir: str) -> Optional[ExperimentData]:
    """
    Đọc dữ liệu metrics và cấu hình từ thư mục experiment.
    
    Args:
        exp_dir (str): Đường dẫn tới thư mục experiment.
        
    Returns:
        Optional[ExperimentData]: Đối tượng dữ liệu experiment hoặc None nếu lỗi.
    """
    # Chuẩn hóa đường dẫn thư mục
    normalized_path = os.path.normpath(exp_dir)
    # Kiểm tra thư mục có tồn tại hay không
    if not os.path.exists(normalized_path):
        # Ghi cảnh báo thư mục không tồn tại
        logger.warning(f"Không tìm thấy thư mục experiment: {exp_dir}")
        # Trả về None báo hiệu không thể đọc
        return None
        
    # Lấy tên định danh thư mục mặc định
    exp_name = os.path.basename(normalized_path)
    # Khởi tạo dictionary chứa metrics thô
    raw_metrics: Dict[str, Any] = {}
    # Khởi tạo dictionary chứa config thô
    raw_config: Dict[str, Any] = {}
    
    # Danh sách các tên file metrics khả dĩ theo thứ tự ưu tiên
    metrics_files = [
        "training_metrics.json",
        "eval_results.json",
        "metrics.json",
        "all_results.json",
    ]
    
    # Tìm kiếm và đọc file metrics trong thư mục
    for filename in metrics_files:
        # Tạo đường dẫn đầy đủ tới file metrics
        filepath = os.path.join(normalized_path, filename)
        # Kiểm tra file có tồn tại
        if os.path.isfile(filepath):
            # Thử mở và đọc dữ liệu JSON
            try:
                # Mở file với encoding utf-8
                with open(filepath, "r", encoding="utf-8") as f:
                    # Parse dữ liệu JSON vào dictionary
                    raw_metrics = json.load(f)
                # Ghi nhận log đọc thành công
                logger.debug(f"Đã đọc metrics từ: {filepath}")
                # Dừng vòng lặp sau khi tìm thấy file ưu tiên nhất
                break
            # Xử lý ngoại lệ nếu file JSON lỗi
            except Exception as e:
                # Ghi log cảnh báo lỗi đọc JSON
                logger.warning(f"Lỗi khi đọc {filepath}: {e}")
                
    # Danh sách các tên file config khả dĩ theo thứ tự ưu tiên
    config_files = [
        "training_config.yaml",
        "config.yaml",
        "config.json",
    ]
    
    # Tìm kiếm và đọc file config trong thư mục
    for filename in config_files:
        # Tạo đường dẫn đầy đủ tới file config
        filepath = os.path.join(normalized_path, filename)
        # Kiểm tra file có tồn tại
        if os.path.isfile(filepath):
            # Thử mở và đọc dữ liệu cấu hình
            try:
                # Mở file cấu hình
                with open(filepath, "r", encoding="utf-8") as f:
                    # Kiểm tra nếu là file YAML
                    if filename.endswith(".yaml") or filename.endswith(".yml"):
                        # Parse YAML thành dictionary
                        raw_config = yaml.safe_load(f) or {}
                    # Nếu là file JSON
                    else:
                        # Parse JSON thành dictionary
                        raw_config = json.load(f) or {}
                # Ghi nhận log đọc config thành công
                logger.debug(f"Đã đọc config từ: {filepath}")
                # Dừng vòng lặp sau khi tìm thấy file cấu hình
                break
            # Xử lý ngoại lệ nếu đọc cấu hình thất bại
            except Exception as e:
                # Ghi log cảnh báo lỗi đọc cấu hình
                logger.warning(f"Lỗi khi đọc {filepath}: {e}")

    # Đọc thêm thông tin từ trainer_state.json nếu có
    trainer_state_path = os.path.join(normalized_path, "trainer_state.json")
    # Khởi tạo dictionary rỗng cho trainer state
    trainer_state: Dict[str, Any] = {}
    # Kiểm tra sự tồn tại của file trainer_state.json
    if os.path.isfile(trainer_state_path):
        # Thử đọc dữ liệu trainer_state
        try:
            # Mở file trainer_state
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                # Parse JSON trainer state
                trainer_state = json.load(f)
        # Bắt ngoại lệ nếu file bị lỗi
        except Exception as e:
            # Ghi debug log
            logger.debug(f"Không thể đọc {trainer_state_path}: {e}")

    # Kiểm tra adapter_config.json trong thư mục
    adapter_config_path = os.path.join(normalized_path, "adapter_config.json")
    # Cờ đánh dấu có file adapter_config
    has_adapter_config = os.path.isfile(adapter_config_path)
    # Khởi tạo dictionary cho adapter config
    adapter_config: Dict[str, Any] = {}
    # Nếu file adapter_config tồn tại
    if has_adapter_config:
        # Thử đọc nội dung adapter_config.json
        try:
            # Mở file adapter config
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                # Parse JSON adapter config
                adapter_config = json.load(f)
        # Bỏ qua ngoại lệ đọc file
        except Exception:
            # Bỏ qua lỗi
            pass

    # Trích xuất tên experiment từ config nếu có
    if "logging" in raw_config and isinstance(raw_config["logging"], dict):
        # Lấy experiment_name từ mục logging
        custom_name = raw_config["logging"].get("experiment_name")
        # Nếu có đặt tên riêng thì gán làm tên hiển thị
        if custom_name:
            # Gán tên định danh mới
            exp_name = str(custom_name)

    # Trích xuất thông tin Model
    model_name = "gpt2"
    # Kiểm tra model trong raw_config
    if "model" in raw_config and isinstance(raw_config["model"], dict):
        # Lấy tên model từ section model
        model_name = raw_config["model"].get("name", "gpt2")
    # Kiểm tra model_name trực tiếp từ raw_config
    elif "model_name" in raw_config:
        # Lấy model_name trực tiếp
        model_name = str(raw_config["model_name"])
    # Kiểm tra model_name từ metrics
    elif "model_name" in raw_metrics:
        # Lấy model_name từ metrics
        model_name = str(raw_metrics["model_name"])
    # Kiểm tra base_model_name_or_path từ adapter config
    elif "base_model_name_or_path" in adapter_config:
        # Lấy tên base model từ LoRA config
        model_name = str(adapter_config["base_model_name_or_path"])

    # Trích xuất thông tin Task Type
    task_type = "causal_lm"
    # Kiểm tra task_type trong raw_config
    if "model" in raw_config and isinstance(raw_config["model"], dict):
        # Lấy task_type từ section model
        task_type = raw_config["model"].get("task_type", "causal_lm")
    # Kiểm tra task_type trực tiếp
    elif "task_type" in raw_config:
        # Lấy task_type trực tiếp
        task_type = str(raw_config["task_type"])
    # Kiểm tra task trong raw_metrics
    elif "task_type" in raw_metrics:
        # Lấy task_type từ metrics
        task_type = str(raw_metrics["task_type"])

    # Trích xuất thông tin LoRA / PEFT
    lora_enabled = False
    # Khởi tạo rank r mặc định
    lora_r = 16
    # Kiểm tra peft trong raw_config
    if "peft" in raw_config and isinstance(raw_config["peft"], dict):
        # Lấy cờ bật LoRA từ config
        lora_enabled = bool(raw_config["peft"].get("enabled", False))
        # Lấy rank r của LoRA
        lora_r = int(raw_config["peft"].get("r", 16))
    # Kiểm tra cờ peft_enabled trực tiếp
    elif "peft_enabled" in raw_config:
        # Lấy giá trị boolean
        lora_enabled = bool(raw_config["peft_enabled"])
        # Lấy giá trị rank r nếu có
        lora_r = int(raw_config.get("peft_r", 16))
    # Nếu tìm thấy adapter_config.json thì chắc chắn là LoRA
    elif has_adapter_config:
        # Đánh dấu bật LoRA
        lora_enabled = True
        # Lấy rank r từ adapter config
        lora_r = int(adapter_config.get("r", 16))

    # Xây dựng chuỗi trạng thái LoRA
    if lora_enabled:
        # Định dạng chuỗi khi bật LoRA
        lora_status = f"enabled (r={lora_r})"
        # Tên hiển thị mô hình kèm thông tin LoRA
        model_display_name = f"{model_name} + LoRA r={lora_r}"
    else:
        # Định dạng chuỗi khi tắt LoRA
        lora_status = "disabled"
        # Tên hiển thị mô hình tiêu chuẩn
        model_display_name = model_name

    # Trích xuất train_loss
    train_loss: Optional[float] = None
    # Duyệt qua các trường khả dĩ chứa train loss
    for key in ["train_loss", "loss", "train/loss"]:
        # Kiểm tra key có trong raw_metrics
        if key in raw_metrics and raw_metrics[key] is not None:
            # Ép kiểu sang float an toàn
            try:
                # Gán giá trị float train loss
                train_loss = float(raw_metrics[key])
                # Dừng kiểm tra khi đã tìm thấy
                break
            # Bỏ qua lỗi ép kiểu
            except (ValueError, TypeError):
                # Tiếp tục tìm
                pass

    # Trích xuất eval_loss
    eval_loss: Optional[float] = None
    # Duyệt qua các trường khả dĩ chứa eval loss
    for key in ["eval_loss", "test_loss", "val_loss", "eval/loss"]:
        # Kiểm tra key có trong raw_metrics
        if key in raw_metrics and raw_metrics[key] is not None:
            # Ép kiểu sang float
            try:
                # Gán giá trị float eval loss
                eval_loss = float(raw_metrics[key])
                # Dừng kiểm tra khi đã tìm thấy
                break
            # Bỏ qua lỗi ép kiểu
            except (ValueError, TypeError):
                # Tiếp tục tìm
                pass

    # Trích xuất hoặc tính toán Perplexity
    perplexity: Optional[float] = None
    # Duyệt qua các trường chứa perplexity
    for key in ["perplexity", "eval_perplexity", "test_perplexity"]:
        # Kiểm tra key trong raw_metrics
        if key in raw_metrics and raw_metrics[key] is not None:
            # Ép kiểu float
            try:
                # Gán perplexity từ metrics
                perplexity = float(raw_metrics[key])
                # Dừng kiểm tra
                break
            # Bỏ qua lỗi ép kiểu
            except (ValueError, TypeError):
                # Tiếp tục tìm
                pass
                
    # Nếu chưa có perplexity nhưng có eval_loss hợp lệ và tác vụ là causal_lm/completion
    if perplexity is None and eval_loss is not None and task_type != "classification":
        # Thử tính perplexity = exp(eval_loss)
        try:
            # Tính lũy thừa exp của eval_loss
            perplexity = math.exp(eval_loss)
        # Bắt lỗi tràn số nếu loss quá lớn
        except OverflowError:
            # Gán vô cực cho perplexity
            perplexity = float("inf")

    # Trích xuất Accuracy (chủ yếu cho classification)
    accuracy: Optional[float] = None
    # Duyệt qua các trường chứa accuracy
    for key in ["accuracy", "eval_accuracy", "test_accuracy", "eval/accuracy"]:
        # Kiểm tra key trong raw_metrics
        if key in raw_metrics and raw_metrics[key] is not None:
            # Ép kiểu float
            try:
                # Gán accuracy
                accuracy = float(raw_metrics[key])
                # Dừng kiểm tra
                break
            # Bỏ qua lỗi
            except (ValueError, TypeError):
                # Tiếp tục tìm
                pass

    # Trích xuất F1 Score
    f1: Optional[float] = None
    # Duyệt qua các trường chứa f1
    for key in ["f1", "eval_f1", "test_f1", "f1_score", "eval/f1"]:
        # Kiểm tra key trong raw_metrics
        if key in raw_metrics and raw_metrics[key] is not None:
            # Ép kiểu float
            try:
                # Gán f1 score
                f1 = float(raw_metrics[key])
                # Dừng kiểm tra
                break
            # Bỏ qua lỗi
            except (ValueError, TypeError):
                # Tiếp tục tìm
                pass

    # Trích xuất thời gian huấn luyện (training time)
    training_time: Optional[float] = None
    # Duyệt qua các trường thời gian
    for key in ["train_runtime", "training_time", "total_train_time", "runtime"]:
        # Kiểm tra key trong raw_metrics
        if key in raw_metrics and raw_metrics[key] is not None:
            # Ép kiểu float
            try:
                # Gán thời gian tính bằng giây
                training_time = float(raw_metrics[key])
                # Dừng kiểm tra
                break
            # Bỏ qua lỗi
            except (ValueError, TypeError):
                # Tiếp tục tìm
                pass

    # Định dạng chuỗi hiển thị thời gian huấn luyện
    training_time_str = "N/A"
    # Nếu có thời gian huấn luyện
    if training_time is not None:
        # Sử dụng hàm format_duration từ src.utils
        training_time_str = format_duration(training_time)

    # Tính toán dung lượng mô hình trên đĩa
    size_bytes, is_adapter_dir = get_directory_model_size(normalized_path)
    # Định dạng chuỗi kích thước mô hình
    if size_bytes > 0:
        # Chuyển đổi bytes sang MB/GB
        formatted_size = format_size(size_bytes)
        # Thêm ghi chú (adapter) nếu là mô hình LoRA adapter
        if is_adapter_dir or lora_enabled:
            # Định dạng chuỗi kích thước adapter
            model_size_str = f"{formatted_size} (adapter)"
        else:
            # Định dạng chuỗi kích thước mô hình đầy đủ
            model_size_str = formatted_size
    else:
        # Nếu không có thông tin dung lượng
        model_size_str = "N/A"

    # Trích xuất tổng số epoch huấn luyện
    num_epochs: Optional[int] = None
    # Kiểm tra trong raw_config training.num_epochs
    if "training" in raw_config and isinstance(raw_config["training"], dict):
        # Lấy num_epochs từ cấu hình training
        num_epochs = raw_config["training"].get("num_epochs")
    # Kiểm tra num_epochs trực tiếp
    elif "num_epochs" in raw_config:
        # Lấy trực tiếp
        num_epochs = raw_config.get("num_epochs")

    # Trích xuất epoch tốt nhất hoặc epoch kết thúc
    best_epoch_str = "N/A"
    # Kiểm tra best_epoch từ raw_metrics
    if "best_epoch" in raw_metrics:
        # Lấy giá trị best_epoch
        best_ep = raw_metrics["best_epoch"]
        # Ghép thành dạng 'best_epoch/total_epochs'
        best_epoch_str = f"{best_ep}/{num_epochs}" if num_epochs else f"{best_ep}"
    # Kiểm tra epoch từ raw_metrics
    elif "epoch" in raw_metrics:
        # Lấy giá trị epoch
        ep = raw_metrics["epoch"]
        # Định dạng số nguyên hoặc làm tròn 1 chữ số
        ep_formatted = f"{int(ep)}" if isinstance(ep, (int, float)) and int(ep) == ep else f"{ep:.1f}"
        # Ghép chuỗi với tổng số epoch
        best_epoch_str = f"{ep_formatted}/{num_epochs}" if num_epochs else f"{ep_formatted}"
    # Kiểm tra epoch từ trainer_state
    elif "epoch" in trainer_state:
        # Lấy epoch từ trainer_state
        ep = trainer_state["epoch"]
        # Định dạng chuỗi epoch
        ep_formatted = f"{int(ep)}" if isinstance(ep, (int, float)) and int(ep) == ep else f"{ep:.1f}"
        # Ghép chuỗi
        best_epoch_str = f"{ep_formatted}/{num_epochs}" if num_epochs else f"{ep_formatted}"

    # Khởi tạo và trả về đối tượng ExperimentData
    return ExperimentData(
        name=exp_name,
        dir_path=normalized_path,
        model_name=model_name,
        model_display_name=model_display_name,
        task_type=task_type,
        lora_status=lora_status,
        lora_r=lora_r,
        train_loss=train_loss,
        eval_loss=eval_loss,
        perplexity=perplexity,
        accuracy=accuracy,
        f1=f1,
        training_time=training_time,
        training_time_str=training_time_str,
        model_size_str=model_size_str,
        best_epoch_str=best_epoch_str,
        raw_metrics=raw_metrics,
        raw_config=raw_config,
    )


def format_metric_value(val: Optional[float], metric_type: str = "general") -> str:
    """
    Định dạng giá trị số của metric thành chuỗi hiển thị gọn gàng.
    
    Args:
        val (Optional[float]): Giá trị số cần định dạng.
        metric_type (str): Loại metric ('loss', 'perplexity', 'percentage', 'general').
        
    Returns:
        str: Chuỗi số đã định dạng hoặc 'N/A' nếu giá trị None.
    """
    # Kiểm tra giá trị None hoặc không phải số
    if val is None or math.isnan(val):
        # Trả về N/A nếu không có giá trị
        return "N/A"
    # Kiểm tra giá trị vô cực
    if math.isinf(val):
        # Trả về inf nếu vô cực
        return "inf"
    # Định dạng theo tỷ lệ phần trăm (accuracy, f1)
    if metric_type == "percentage":
        # Nhân 100 và thêm ký hiệu %
        return f"{val * 100:.2f}%"
    # Định dạng loss với 4 chữ số thập phân
    elif metric_type == "loss":
        # Format 4 số thập phân
        return f"{val:.4f}"
    # Định dạng perplexity với 2 chữ số thập phân
    elif metric_type == "perplexity":
        # Format 2 số thập phân
        return f"{val:.2f}"
    # Mặc định định dạng thông thường
    else:
        # Format 4 số thập phân
        return f"{val:.4f}"


def find_best_indices(
    experiments: List[ExperimentData],
    metric_name: str,
    lower_is_better: bool = True,
) -> List[int]:
    """
    Tìm danh sách chỉ số các experiment đạt kết quả tốt nhất cho một metric cụ thể.
    
    Args:
        experiments (List[ExperimentData]): Danh sách các experiment cần so sánh.
        metric_name (str): Tên thuộc tính metric trong ExperimentData.
        lower_is_better (bool): True nếu giá trị nhỏ hơn là tốt hơn (loss, perplexity, time).
        
    Returns:
        List[int]: Danh sách các vị trí index đạt giá trị tối ưu nhất.
    """
    # Danh sách các cặp (index, giá trị) có giá trị hợp lệ
    valid_entries: List[Tuple[int, float]] = []
    # Duyệt qua từng experiment
    for idx, exp in enumerate(experiments):
        # Lấy giá trị của thuộc tính metric
        val = getattr(exp, metric_name, None)
        # Kiểm tra giá trị hợp lệ kiểu số thực và không vô cực
        if val is not None and isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
            # Thêm vào danh sách hợp lệ
            valid_entries.append((idx, float(val)))
            
    # Nếu không có experiment nào có giá trị hợp lệ
    if not valid_entries:
        # Trả về danh sách rỗng
        return []
        
    # Tìm giá trị tốt nhất tùy theo lower_is_better
    if lower_is_better:
        # Tìm giá trị nhỏ nhất
        best_val = min(val for _, val in valid_entries)
    else:
        # Tìm giá trị lớn nhất
        best_val = max(val for _, val in valid_entries)
        
    # Thu thập tất cả các index đạt giá trị tốt nhất (sai số 1e-6)
    best_indices = [idx for idx, val in valid_entries if abs(val - best_val) <= 1e-6]
    # Trả về danh sách index tốt nhất
    return best_indices


def build_comparison_table(
    experiments: List[ExperimentData],
    target_metric: str = "eval_loss",
) -> str:
    """
    Xây dựng bảng so sánh dạng Unicode box-drawing đẹp mắt giữa các experiments.
    
    Args:
        experiments (List[ExperimentData]): Danh sách các experiments cần so sánh.
        target_metric (str): Tên metric chính cần nhấn mạnh.
        
    Returns:
        str: Chuỗi bảng so sánh đã định dạng hoàn chỉnh.
    """
    # Nếu danh sách experiments rỗng
    if not experiments:
        # Trả về thông báo không có dữ liệu
        return "  (Không có dữ liệu experiment để so sánh)"

    # Tìm các experiment tốt nhất cho từng chỉ số
    best_train_loss = find_best_indices(experiments, "train_loss", lower_is_better=True)
    best_eval_loss = find_best_indices(experiments, "eval_loss", lower_is_better=True)
    best_perplexity = find_best_indices(experiments, "perplexity", lower_is_better=True)
    best_accuracy = find_best_indices(experiments, "accuracy", lower_is_better=False)
    best_f1 = find_best_indices(experiments, "f1", lower_is_better=False)
    best_training_time = find_best_indices(experiments, "training_time", lower_is_better=True)

    # Kiểm tra xem có bất kỳ experiment nào có accuracy hay không
    has_accuracy = any(exp.accuracy is not None for exp in experiments)
    # Kiểm tra xem có bất kỳ experiment nào có f1 score hay không
    has_f1 = any(exp.f1 is not None for exp in experiments)

    # Chuẩn bị tiêu đề các cột experiment
    exp_headers: List[str] = []
    # Duyệt qua từng experiment để tạo header cột
    for exp in experiments:
        # Tên cột cơ bản
        header = exp.name
        # Thêm nhãn (LoRA) vào tiêu đề cột nếu bật LoRA
        if "enabled" in exp.lora_status:
            # Header có chú thích LoRA
            header = f"{exp.name} (LoRA)"
        # Thêm header vào danh sách
        exp_headers.append(header)

    # Danh sách các dòng dữ liệu cần hiển thị trong bảng
    # Mỗi phần tử là tuple: (Tên hiển thị của Metric, Danh sách giá trị theo từng experiment)
    rows: List[Tuple[str, List[str]]] = []

    # 1. Dòng Model (hiển thị model_display_name: gpt2, gpt2 + LoRA r=16)
    model_row = [exp.model_display_name for exp in experiments]
    rows.append(("Model", model_row))

    # 2. Dòng Task
    task_row = [exp.task_type for exp in experiments]
    rows.append(("Task", task_row))

    # 3. Dòng LoRA
    lora_row = [exp.lora_status for exp in experiments]
    rows.append(("LoRA", lora_row))

    # 4. Dòng Train Loss (thêm sao ⭐ cho giá trị tốt nhất)
    train_loss_row = [
        f"{format_metric_value(exp.train_loss, 'loss')}{' ⭐' if i in best_train_loss and len(experiments) > 1 else ''}"
        for i, exp in enumerate(experiments)
    ]
    rows.append(("Train loss", train_loss_row))

    # 5. Dòng Eval Loss (thêm sao ⭐ cho giá trị tốt nhất)
    eval_loss_row = [
        f"{format_metric_value(exp.eval_loss, 'loss')}{' ⭐' if i in best_eval_loss and len(experiments) > 1 else ''}"
        for i, exp in enumerate(experiments)
    ]
    rows.append(("Eval loss", eval_loss_row))

    # 6. Dòng Perplexity (thêm sao ⭐ cho giá trị tốt nhất)
    perplexity_row = [
        f"{format_metric_value(exp.perplexity, 'perplexity')}{' ⭐' if i in best_perplexity and len(experiments) > 1 else ''}"
        for i, exp in enumerate(experiments)
    ]
    rows.append(("Perplexity", perplexity_row))

    # 7. Dòng Accuracy nếu có bài toán phân loại
    if has_accuracy:
        accuracy_row = [
            f"{format_metric_value(exp.accuracy, 'percentage')}{' ⭐' if i in best_accuracy and len(experiments) > 1 else ''}"
            for i, exp in enumerate(experiments)
        ]
        rows.append(("Accuracy", accuracy_row))

    # 8. Dòng F1 Score nếu có
    if has_f1:
        f1_row = [
            f"{format_metric_value(exp.f1, 'percentage')}{' ⭐' if i in best_f1 and len(experiments) > 1 else ''}"
            for i, exp in enumerate(experiments)
        ]
        rows.append(("F1 Score", f1_row))

    # 9. Dòng Training Time (thêm sao ⭐ cho thời gian nhanh nhất)
    time_row = [
        f"{exp.training_time_str}{' ⭐' if i in best_training_time and len(experiments) > 1 and exp.training_time is not None else ''}"
        for i, exp in enumerate(experiments)
    ]
    rows.append(("Training time", time_row))

    # 10. Dòng Model Size
    size_row = [exp.model_size_str for exp in experiments]
    rows.append(("Model size", size_row))

    # 11. Dòng Best Epoch
    epoch_row = [exp.best_epoch_str for exp in experiments]
    rows.append(("Best epoch", epoch_row))

    # Tính độ rộng của cột đầu tiên (Metric) có khoảng cách đệm
    metric_col_name = "Metric"
    # Tìm độ dài lớn nhất giữa tên cột và các nhãn dòng (+ 4 ký tự padding để thoáng mắt)
    col0_width = max(len(metric_col_name), max(len(row[0]) for row in rows)) + 4
    # Đảm bảo độ rộng tối thiểu là 16 ký tự
    col0_width = max(col0_width, 16)

    # Tính độ rộng cho từng cột experiment
    exp_col_widths: List[int] = []
    # Duyệt qua từng cột experiment
    for col_idx, header in enumerate(exp_headers):
        # Lấy độ dài của tiêu đề cột
        max_w = visible_len(header)
        # Duyệt qua từng dòng để tìm ô có độ dài lớn nhất trong cột này
        for _, row_values in rows:
            # Lấy độ dài chuỗi ô hiện tại
            cell_w = visible_len(row_values[col_idx])
            # Cập nhật độ dài lớn nhất
            if cell_w > max_w:
                max_w = cell_w
        # Bổ sung khoảng đệm padding 4 ký tự và tối thiểu 14 ký tự
        col_w = max(max_w + 4, 14)
        # Lưu độ rộng cột
        exp_col_widths.append(col_w)

    # Tổng độ rộng nội dung bảng (không tính viền ngoài 2 bên)
    total_inner_width = col0_width + sum(exp_col_widths) + len(exp_col_widths)

    # Danh sách các dòng để ghép thành bảng
    lines: List[str] = []

    # Tiêu đề bảng
    title_text = "📊 Experiment Comparison"

    # Dòng viền trên cùng của banner tiêu đề
    top_border = "┌" + "─" * total_inner_width + "┐"
    lines.append(top_border)

    # Dòng tiêu đề căn giữa
    title_line = "│" + pad_visible(title_text, total_inner_width, align="center") + "│"
    lines.append(title_line)

    # Dòng phân cách giữa tiêu đề và hàng Header các cột
    header_sep_top = "├" + "─" * col0_width + "".join("┬" + "─" * w for w in exp_col_widths) + "┤"
    lines.append(header_sep_top)

    # Dòng Header chứa tên các cột (thêm khoảng trắng đệm phía trước)
    header_cells = [pad_visible(f" {metric_col_name}", col0_width, align="left")]
    # Thêm tiêu đề từng experiment căn trái với khoảng trắng đệm
    for col_idx, header in enumerate(exp_headers):
        # Căn trái tiêu đề experiment
        header_cells.append(pad_visible(f" {header}", exp_col_widths[col_idx], align="left"))
    # Ghép dòng header với ký tự phân cách cột │
    header_line = "│" + "│".join(header_cells) + "│"
    lines.append(header_line)

    # Dòng phân cách giữa Header và phần dữ liệu
    header_sep_bot = "├" + "─" * col0_width + "".join("┼" + "─" * w for w in exp_col_widths) + "┤"
    lines.append(header_sep_bot)

    # Thêm từng dòng dữ liệu vào bảng
    for label, values in rows:
        # Căn lề nhãn của dòng kèm khoảng trắng đầu
        row_cells = [pad_visible(f" {label}", col0_width, align="left")]
        # Căn lề giá trị của từng experiment kèm khoảng trắng đầu
        for col_idx, val_str in enumerate(values):
            # Căn trái giá trị ô dữ liệu
            row_cells.append(pad_visible(f" {val_str}", exp_col_widths[col_idx], align="left"))
        # Ghép thành dòng hoàn chỉnh
        line = "│" + "│".join(row_cells) + "│"
        # Thêm dòng vào bảng
        lines.append(line)

    # Dòng viền đáy kết thúc bảng
    bottom_border = "└" + "─" * col0_width + "".join("┴" + "─" * w for w in exp_col_widths) + "┘"
    lines.append(bottom_border)

    # Ghép toàn bộ các dòng thành chuỗi bảng hoàn chỉnh
    table_str = "\n".join(lines)
    # Trả về bảng định dạng
    return table_str


def build_summary_text(
    experiments: List[ExperimentData],
    target_metric: str = "eval_loss",
) -> str:
    """
    Xây dựng đoạn văn bản tóm tắt các kết quả tốt nhất và xếp hạng theo target_metric.
    
    Args:
        experiments (List[ExperimentData]): Danh sách các experiments đã phân tích.
        target_metric (str): Tên metric chính làm căn cứ xếp hạng.
        
    Returns:
        str: Chuỗi văn bản tóm tắt kết quả.
    """
    # Nếu chỉ có 1 experiment, hiển thị bảng metrics chi tiết từ format_metrics_table
    if len(experiments) == 1:
        # Lấy experiment duy nhất
        exp = experiments[0]
        # Nếu có metrics thô thì hiển thị bảng metrics
        if exp.raw_metrics:
            # Tạo bảng metrics chi tiết
            detail_table = format_metrics_table(exp.raw_metrics, title=f"Raw Metrics: {exp.name}")
            # Trả về chuỗi chi tiết
            return f"\n{detail_table}"
        # Trả về chuỗi rỗng nếu không có metrics thô
        return ""

    # Nếu không có experiment
    if not experiments:
        # Trả về rỗng
        return ""

    # Danh sách các dòng tóm tắt
    summary_lines: List[str] = []
    # Thêm tiêu đề phần tóm tắt
    summary_lines.append("\n🏆 Tóm tắt kết quả nổi bật (Highlights):")

    # Kiểm tra metric Eval Loss
    best_eval = find_best_indices(experiments, "eval_loss", lower_is_better=True)
    if best_eval:
        # Lấy tên các exp tốt nhất
        names = ", ".join(experiments[i].name for i in best_eval)
        # Lấy giá trị loss tốt nhất
        loss_val = experiments[best_eval[0]].eval_loss
        # Thêm dòng tóm tắt eval loss
        summary_lines.append(f"  • Lowest Eval Loss:  {names} ({format_metric_value(loss_val, 'loss')})")

    # Kiểm tra metric Perplexity
    best_ppl = find_best_indices(experiments, "perplexity", lower_is_better=True)
    if best_ppl:
        # Lấy tên các exp
        names = ", ".join(experiments[i].name for i in best_ppl)
        # Lấy giá trị perplexity
        ppl_val = experiments[best_ppl[0]].perplexity
        # Thêm dòng tóm tắt perplexity
        summary_lines.append(f"  • Best Perplexity:   {names} ({format_metric_value(ppl_val, 'perplexity')})")

    # Kiểm tra metric Accuracy nếu có
    best_acc = find_best_indices(experiments, "accuracy", lower_is_better=False)
    if best_acc:
        # Lấy tên các exp
        names = ", ".join(experiments[i].name for i in best_acc)
        # Lấy giá trị accuracy
        acc_val = experiments[best_acc[0]].accuracy
        # Thêm dòng tóm tắt accuracy
        summary_lines.append(f"  • Highest Accuracy:  {names} ({format_metric_value(acc_val, 'percentage')})")

    # Kiểm tra thời gian huấn luyện
    best_time = find_best_indices(experiments, "training_time", lower_is_better=True)
    if best_time:
        # Lấy tên exp nhanh nhất
        names = ", ".join(experiments[i].name for i in best_time)
        # Lấy chuỗi thời gian
        time_str = experiments[best_time[0]].training_time_str
        # Thêm dòng tóm tắt thời gian
        summary_lines.append(f"  • Fastest Training:  {names} ({time_str})")

    # Ghép các dòng tóm tắt lại
    return "\n".join(summary_lines)


def compare_experiments(
    experiment_dirs: List[str],
    metric: str = "eval_loss",
    sort_by_metric: bool = False,
    output_file: Optional[str] = None,
) -> str:
    """
    Hàm điều phối chính thực hiện so sánh nhiều experiments.
    
    Args:
        experiment_dirs (List[str]): Danh sách đường dẫn tới các thư mục experiment.
        metric (str): Tên metric căn cứ so sánh (mặc định: eval_loss).
        sort_by_metric (bool): Có sắp xếp danh sách theo metric hay không.
        output_file (Optional[str]): Đường dẫn tập tin lưu kết quả bảng (nếu có).
        
    Returns:
        str: Chuỗi bảng so sánh và tóm tắt hoàn chỉnh.
    """
    # Ghi log bắt đầu tiến trình so sánh
    logger.info(f"Bắt đầu so sánh {len(experiment_dirs)} experiments...")

    # Khởi tạo danh sách chứa các experiment đã đọc thành công
    experiments: List[ExperimentData] = []

    # Duyệt qua từng đường dẫn được truyền vào
    for exp_dir in experiment_dirs:
        # Đọc dữ liệu experiment từ thư mục
        exp_data = load_experiment_data(exp_dir)
        # Nếu đọc thành công
        if exp_data is not None:
            # Thêm vào danh sách experiments
            experiments.append(exp_data)
        else:
            # Ghi log cảnh báo bỏ qua thư mục lỗi
            logger.warning(f"Bỏ qua thư mục không hợp lệ: {exp_dir}")

    # Kiểm tra số lượng experiments hợp lệ
    if not experiments:
        # Ghi log lỗi khi không có experiment hợp lệ
        logger.error("Không tìm thấy experiment hợp lệ nào để so sánh.")
        # Trả về thông báo lỗi
        return "❌ Không có experiment hợp lệ để so sánh."

    # Sắp xếp danh sách nếu người dùng yêu cầu
    if sort_by_metric:
        # Xác định chiều sắp xếp (loss/ppl/time: tăng dần = tốt hơn; acc/f1: giảm dần = tốt hơn)
        reverse = metric in ["accuracy", "f1", "eval_accuracy", "eval_f1"]
        # Hàm lấy giá trị metric an toàn để sắp xếp
        def get_sort_key(exp: ExperimentData) -> float:
            # Lấy giá trị metric
            val = getattr(exp, metric, None)
            # Nếu giá trị None trả về vô cực phù hợp để đứng cuối bảng
            if val is None:
                return float("-inf") if reverse else float("inf")
            # Trả về giá trị số thực
            return float(val)
        # Thực hiện sắp xếp danh sách experiments tại chỗ
        experiments.sort(key=get_sort_key, reverse=reverse)
        # Ghi log thông báo đã sắp xếp
        logger.info(f"Đã sắp xếp experiments theo metric: {metric}")

    # Xây dựng bảng so sánh dạng Unicode
    table = build_comparison_table(experiments, target_metric=metric)
    # Xây dựng đoạn văn bản tóm tắt
    summary = build_summary_text(experiments, target_metric=metric)
    # Ghép bảng và tóm tắt thành kết quả cuối cùng
    full_output = f"{table}\n{summary}" if summary else table

    # In kết quả trực tiếp ra console
    print(full_output)

    # Lưu kết quả ra file nếu có đường dẫn output_file
    if output_file:
        # Thử mở và ghi kết quả ra file
        try:
            # Tạo thư mục cha nếu chưa tồn tại
            parent_dir = os.path.dirname(output_file)
            # Nếu có đường dẫn thư mục cha
            if parent_dir:
                # Tạo thư mục đệ quy
                os.makedirs(parent_dir, exist_ok=True)
            # Mở file để ghi dữ liệu
            with open(output_file, "w", encoding="utf-8") as f:
                # Ghi toàn bộ nội dung bảng và tóm tắt
                f.write(full_output + "\n")
            # Ghi log thông báo đã lưu file thành công
            logger.info(f"Đã lưu bảng so sánh vào tập tin: {output_file}")
        # Xử lý ngoại lệ nếu không ghi được file
        except Exception as e:
            # Ghi log lỗi ghi file
            logger.error(f"Không thể lưu kết quả ra file {output_file}: {e}")

    # Trả về chuỗi kết quả
    return full_output


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Thiết lập và phân tích các tham số dòng lệnh (CLI arguments).
    
    Args:
        args (Optional[List[str]]): Danh sách các tham số dòng lệnh tùy chọn (cho unit test).
        
    Returns:
        argparse.Namespace: Đối tượng chứa các tham số đã phân tích.
    """
    # Khởi tạo đối tượng ArgumentParser với mô tả chi tiết bằng tiếng Việt
    parser = argparse.ArgumentParser(
        description="📊 GPT-2 Fine-Tuning Pipeline — Công cụ so sánh kết quả experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # So sánh 2 experiments cơ bản:
  python compare.py outputs/exp1 outputs/exp2

  # So sánh 3 experiments và chọn metric eval_loss để highlight:
  python compare.py outputs/exp1 outputs/exp2 outputs/exp3 --metric eval_loss

  # So sánh và sắp xếp theo accuracy, lưu kết quả ra file:
  python compare.py outputs/cls_exp1 outputs/cls_exp2 --metric accuracy --sort --output comparison.txt
        """,
    )

    # Tham số vị trí: danh sách các thư mục experiment
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Danh sách đường dẫn các thư mục experiment cần so sánh (VD: outputs/exp1 outputs/exp2)",
    )

    # Tham số tùy chọn: Metric chính để đánh giá và highlight
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="eval_loss",
        help="Metric chính dùng để so sánh và làm nổi bật (mặc định: eval_loss)",
    )

    # Tham số tùy chọn: Tự động sắp xếp experiments theo metric
    parser.add_argument(
        "--sort",
        "-s",
        action="store_true",
        help="Tự động sắp xếp các experiments theo thứ tự tối ưu của metric chính",
    )

    # Tham số tùy chọn: Lưu bảng so sánh ra tập tin
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Đường dẫn tập tin lưu bảng so sánh (VD: comparison_report.txt)",
    )

    # Thực hiện phân tích tham số và trả về kết quả
    return parser.parse_args(args)


def main() -> None:
    """
    Hàm thực thi chính khi gọi script từ dòng lệnh.
    """
    # Phân tích các tham số từ dòng lệnh
    parsed_args = parse_args()
    # Gọi hàm điều phối so sánh với các tham số đã phân tích
    compare_experiments(
        experiment_dirs=parsed_args.experiment_dirs,
        metric=parsed_args.metric,
        sort_by_metric=parsed_args.sort,
        output_file=parsed_args.output,
    )


# Điểm vào chính của script khi được thực thi trực tiếp
if __name__ == "__main__":
    # Gọi hàm main
    main()
