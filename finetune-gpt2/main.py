"""
Main CLI Entry Point cho GPT-2 Fine-tuning Pipeline
===================================================
Kết nối tất cả các module trong thư mục src/ để tạo thành pipeline hoàn chỉnh.
Hỗ trợ chạy pipeline từ đầu đến cuối hoặc chạy từng bước đơn lẻ.
"""

# === Standard Library ===
import argparse  # Phân tích cú pháp đối số dòng lệnh
import sys       # Giao tiếp với hệ thống, exit
from typing import Dict, Any, Optional, List  # Gợi ý kiểu dữ liệu

# === Local ===
# Import config loaders
from src.config import load_config, config_to_dict
# Import utilities
from src.utils import setup_logger, detect_device, set_seed, StepTimer, format_banner
# Import các module cốt lõi của pipeline
from src.data import prepare_data
from src.model import create_model_init
from src.hp_search import run_hp_search
from src.trainer import run_training
from src.evaluate import run_evaluation, generate_evaluation_report
from src.inference import save_model, generate_text, load_model

# Khởi tạo logger cho module chính
logger = setup_logger("gpt2_finetune.main")


def _parse_overrides(unknown_args: List[str]) -> Dict[str, Any]:
    """
    Phân tích các đối số thừa trên dòng lệnh thành dictionary overrides.
    Hỗ trợ cú pháp chấm (dotted notation), ví dụ: --training.batch_size 4
    """
    overrides = {}  # Khởi tạo dictionary chứa config ghi đè
    i = 0           # Con trỏ duyệt mảng arguments
    
    # Duyệt qua các tham số chưa được nhận diện
    while i < len(unknown_args):
        arg = unknown_args[i]
        
        # Nếu arg bắt đầu bằng '--' (flag / biến)
        if arg.startswith("--"):
            key = arg[2:]  # Loại bỏ ký tự '--' ở đầu
            
            # Nếu tham số tiếp theo tồn tại và không phải là một flag khác
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                val = unknown_args[i + 1]
                
                # Cố gắng chuyển đổi kiểu dữ liệu cơ bản
                if val.isdigit():
                    val = int(val)         # Chuyển thành số nguyên
                else:
                    try:
                        val = float(val)   # Thử chuyển thành số thực
                    except ValueError:
                        # Nếu là chuỗi, kiểm tra boolean
                        if val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                
                overrides[key] = val       # Lưu vào dictionary
                i += 2                     # Bỏ qua cả key và giá trị vừa xử lý
            else:
                # Nếu không có giá trị đi kèm, giả định flag boolean True
                overrides[key] = True
                i += 1                     # Tiến 1 bước
        else:
            i += 1                         # Bỏ qua nếu không đúng chuẩn flag
            
    return overrides                       # Trả về dict cấu hình ghi đè


def parse_args():
    """
    Khai báo và phân tích cú pháp các đối số CLI.
    Trả về bộ 2: args (các tham số xác định) và unknown (để làm override).
    """
    # Khởi tạo parser cho command line
    parser = argparse.ArgumentParser(description="GPT-2 Fine-Tuning Pipeline CLI")
    
    # Đối số bắt buộc: file cấu hình YAML
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Đường dẫn tới file cấu hình YAML (bắt buộc)"
    )
    
    # Đối số tùy chọn: chỉ chạy một step cụ thể
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["data", "hp_search", "train", "evaluate", "generate"], 
        help="Chạy một step duy nhất: data | hp_search | train | evaluate | generate"
    )
    
    # Đối số tùy chọn: danh sách prompt dùng để sinh văn bản (generate step)
    parser.add_argument(
        "--prompts", 
        type=str, 
        nargs="+", 
        help="Danh sách các prompt để sinh văn bản (cho step 'generate')"
    )
    
    # Phân tích cú pháp, những biến chưa khai báo sẽ rơi vào unknown
    return parser.parse_known_args()


def run_pipeline(config, step: Optional[str] = None, prompts: Optional[List[str]] = None):
    """
    Hàm thực thi luồng chính của Pipeline.
    Nếu `step` được truyền, chỉ chạy step tương ứng. Nếu không, chạy full.
    """
    logger.info("Khởi tạo luồng Pipeline...")
    
    # Khai báo trước các biến state dùng chung giữa các step
    datasets = None
    collator = None
    tokenizer = None
    model_init = None
    best_hp = None
    trainer = None
    
    # ---------------------------------------------------------
    # STEP 1: Data Preparation
    # ---------------------------------------------------------
    if step in (None, "data", "hp_search", "train", "evaluate"):
        # Theo dõi thời gian thực hiện bước chuẩn bị dữ liệu
        with StepTimer("Data Preparation", logger):
            datasets, collator, tokenizer = prepare_data(config)
            
    # ---------------------------------------------------------
    # STEP 2: Model Initialization
    # ---------------------------------------------------------
    if step in (None, "hp_search", "train", "evaluate"):
        # Theo dõi thời gian tạo hàm model_init
        with StepTimer("Model Initialization", logger):
            model_init = create_model_init(config)
            
    # ---------------------------------------------------------
    # STEP 3 & 4: Hyperparameter Search & Training
    # ---------------------------------------------------------
    if step in (None, "hp_search", "train"):
        # Theo dõi thời gian quá trình huấn luyện
        with StepTimer("Training & HP Search", logger):
            # run_training tự động xử lý logic run_hp_search nếu được cấu hình bật
            # và nếu best_hp = None.
            trainer, train_metrics = run_training(
                config=config, 
                datasets=datasets, 
                data_collator=collator, 
                tokenizer=tokenizer, 
                model_init=model_init,
                best_hp=None
            )
            
    # ---------------------------------------------------------
    # STEP 5: Evaluation (Kiểm thử)
    # ---------------------------------------------------------
    if step in (None, "evaluate") and trainer is not None:
        # Theo dõi thời gian quá trình đánh giá
        with StepTimer("Evaluation", logger):
            # Tiến hành đánh giá trên tập test (nếu có)
            test_dataset = datasets.get("test")
            test_metrics = run_evaluation(trainer, test_dataset, config)
            # Tạo báo cáo kết quả và in ra console
            report = generate_evaluation_report(test_metrics, config)
            logger.info("\n" + report)
            
    # ---------------------------------------------------------
    # STEP 6: Save Model (Lưu kết quả)
    # ---------------------------------------------------------
    if step in (None, "train"):
        # Theo dõi thời gian lưu mô hình và cấu hình
        with StepTimer("Save Model", logger):
            saved_path = save_model(trainer, tokenizer, config, train_metrics)
            
    # ---------------------------------------------------------
    # Bổ trợ: Generation (Sinh thử nghiệm)
    # ---------------------------------------------------------
    if step == "generate" and prompts:
        # Tải mô hình từ đường dẫn đã huấn luyện để chuẩn bị sinh văn bản
        model, infer_tokenizer = load_model(config.output.model_dir)
        # Sinh văn bản với danh sách các prompt
        results = generate_text(model, infer_tokenizer, prompts)
        # In ra từng kết quả
        for i, res in enumerate(results):
            logger.info(f"Kết quả {i+1}: {res}")
            
    # ---------------------------------------------------------
    # Tổng kết Pipeline: In lệnh theo dõi TensorBoard
    # ---------------------------------------------------------
    if step in (None, "train"):
        logger.info("=" * 60)
        logger.info("Pipeline đã hoàn tất. Chạy lệnh sau để xem log TensorBoard:")
        logger.info(f"tensorboard --logdir {config.logging.tensorboard_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    try:
        # Bước 1: Phân tích đối số từ dòng lệnh
        args, unknown_args = parse_args()
        
        # Bước 2: Phân tích overrides dạng cấu trúc phân cấp (dotted)
        overrides = _parse_overrides(unknown_args)
        
        # Bước 3: Tải file cấu hình, chèn overrides và validate hợp lệ
        config = load_config(args.config, overrides)
        
        # Thiết lập seed đảm bảo tính lặp lại (reproducibility)
        set_seed(config.training.seed)
        
        # Nhận diện phần cứng (CPU/GPU) phục vụ huấn luyện
        device_info = detect_device()
        
        # Trích xuất config phẳng để in lên banner hiển thị
        config_dict = config_to_dict(config)
        
        # In banner Pipeline tuyệt đẹp
        print(format_banner(config_dict, device_info))
        
        # Bước 4: Gọi Pipeline chính, điều khiển theo step hoặc full
        run_pipeline(config, args.step, args.prompts)
        
    except Exception as e:
        # Xử lý khi xảy ra lỗi bất ngờ, log chi tiết lỗi ra màn hình
        logger.error(f"Đã xảy ra lỗi nghiêm trọng trong hệ thống: {e}")
        # Dừng hệ thống với trạng thái lỗi (mã exit code 1)
        sys.exit(1)
