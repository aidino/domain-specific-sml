"""
Evaluation Module — Đánh giá mô hình trên tập kiểm thử (Test Set)
================================================================
Thực hiện đánh giá model sau quá trình huấn luyện, tính toán các metrics
như Perplexity, Loss, Accuracy, F1-score và định dạng báo cáo kết quả.
"""

# === Standard Library ===
import math  # Thư viện toán học tính exp(eval_loss) ra perplexity
from typing import Any, Dict  # Type hints cho Python 3.10+

# === Third-Party ===
from datasets import Dataset  # Kiểu dữ liệu Dataset của HuggingFace
from transformers import Trainer  # Lớp Trainer điều phối đánh giá của HuggingFace

# === Local ===
from src.config import PipelineConfig  # Cấu hình toàn bộ pipeline
from src.utils import format_metrics_table, setup_logger  # Tiện ích định dạng bảng và logger

# Khởi tạo logger cho module evaluation — đặt tên gpt2_finetune.evaluate
logger = setup_logger("gpt2_finetune.evaluate")


def run_evaluation(
    trainer: Trainer,
    test_dataset: Dataset,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """
    Thực hiện đánh giá mô hình trên tập test_dataset.

    Args:
        trainer: Đối tượng Trainer của HuggingFace đã hoàn thành huấn luyện
        test_dataset: Tập dữ liệu kiểm thử (test split)
        config: Cấu hình pipeline fine-tuning

    Returns:
        dict: Chứa tất cả metrics thu được từ quá trình đánh giá (loss, perplexity, accuracy, ...)
    """
    # Ghi log bắt đầu quá trình đánh giá trên test dataset
    logger.info("Bắt đầu đánh giá mô hình trên tập kiểm thử (Test Set)...")

    # Kiểm tra nếu test_dataset là None hoặc rỗng
    if test_dataset is None or len(test_dataset) == 0:
        # Ghi cảnh báo khi không có dữ liệu test
        logger.warning("Tập kiểm thử (test_dataset) rỗng hoặc không tồn tại, bỏ qua đánh giá.")
        # Trả về dictionary rỗng
        return {}

    # Gọi phương thức evaluate của Trainer trên tập test_dataset
    metrics = trainer.evaluate(eval_dataset=test_dataset)

    # Lấy giá trị eval_loss từ kết quả metrics trả về
    eval_loss = metrics.get("eval_loss")

    # Nếu không có eval_loss, thử lấy test_loss hoặc loss
    if eval_loss is None:
        # Thử lấy từ key test_loss hoặc loss nếu có
        eval_loss = metrics.get("test_loss", metrics.get("loss"))

    # Kiểm tra và tính perplexity khi có loss và task là causal_lm / completion hoặc chung
    if eval_loss is not None:
        # Xử lý ngoại lệ tràn số khi tính exp(eval_loss)
        try:
            # Tính toán perplexity = exp(eval_loss)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            # Xử lý trường hợp loss quá lớn gây tràn số float vô cực
            perplexity = float("inf")

        # Lưu perplexity vào metrics dictionary với key perplexity
        metrics["perplexity"] = perplexity
        # Đồng thời lưu vào key eval_perplexity để tương thích với các công cụ logging
        metrics["eval_perplexity"] = perplexity

    # Ghi log thông báo hoàn thành đánh giá test set
    logger.info("Đánh giá mô hình trên tập kiểm thử hoàn tất.")

    # Trả về dictionary chứa toàn bộ metrics
    return metrics


def generate_evaluation_report(
    metrics: Dict[str, Any],
    config: PipelineConfig,
) -> str:
    """
    Tạo báo cáo đánh giá định dạng đẹp để hiển thị ra console hoặc lưu file.

    Args:
        metrics: Dictionary chứa các metrics sau khi đánh giá
        config: Cấu hình pipeline fine-tuning

    Returns:
        str: Chuỗi báo cáo được định dạng đẹp mắt với box và header
    """
    # Lấy thông tin tên mô hình từ cấu hình
    model_name = config.model.name
    # Lấy thông tin loại tác vụ từ cấu hình
    task_type = config.model.task_type
    # Độ rộng khung báo cáo cố định
    width = 62

    # Khởi tạo danh sách các dòng cho báo cáo
    report_lines: list[str] = []

    # Thêm đường viền trên của khung báo cáo
    report_lines.append("╔" + "═" * width + "╗")
    # Thêm tiêu đề bước đánh giá ở giữa khung
    report_lines.append("║" + "📊 [Step 4/5] Evaluation (Test Set)".center(width) + "║")
    # Thêm đường phân cách ngang
    report_lines.append("╠" + "═" * width + "╣")
    # Thêm thông tin mô hình đã đánh giá
    report_lines.append("║" + f"  Model:     {model_name}".ljust(width) + "║")
    # Thêm thông tin loại tác vụ
    report_lines.append("║" + f"  Task:      {task_type}".ljust(width) + "║")
    # Thêm đường phân cách trước bảng metrics
    report_lines.append("╠" + "═" * width + "╣")

    # Lọc và định dạng các metrics quan trọng sử dụng format_metrics_table
    formatted_table = format_metrics_table(metrics, title="Kết quả chi tiết:")
    # Thêm từng dòng của bảng metrics vào báo cáo
    for line in formatted_table.split("\n"):
        # Căn lề từng dòng và đóng khung viền hai bên
        report_lines.append("║" + line.ljust(width) + "║")

    # Thêm đường viền dưới của khung báo cáo
    report_lines.append("╚" + "═" * width + "╝")

    # Ghép tất cả các dòng lại thành một chuỗi duy nhất
    report_str = "\n".join(report_lines)

    # Trả về chuỗi báo cáo đã định dạng
    return report_str
