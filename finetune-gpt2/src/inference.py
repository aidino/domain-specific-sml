"""
Module inference hỗ trợ lưu, tải mô hình và sinh văn bản.
"""

# stdlib imports
import json  # Import thư viện json để đọc/ghi file json
import os  # Import thư viện os để thao tác với hệ thống file
from typing import Any  # Import Any từ typing để gán kiểu dữ liệu linh hoạt

# third-party imports
import torch  # Import thư viện PyTorch
from peft import PeftModel, PeftConfig  # Import PeftModel và PeftConfig từ thư viện peft để xử lý LoRA
from transformers import (  # Import các lớp cần thiết từ transformers
    AutoModelForCausalLM,  # Lớp AutoModel cho bài toán Language Modeling
    AutoModelForSequenceClassification,  # Lớp AutoModel cho bài toán Classification
    AutoTokenizer,  # Lớp tự động nhận diện và tải tokenizer
    GenerationConfig,  # Lớp cấu hình tham số sinh văn bản
    PreTrainedModel,  # Lớp cơ sở cho mô hình đã huấn luyện
    PreTrainedTokenizerBase,  # Lớp cơ sở cho tokenizer
    Trainer,  # Lớp Trainer dùng để huấn luyện và lưu mô hình
)

# local imports
from src.config import PipelineConfig, save_config  # Import config và hàm lưu config từ module local
from src.utils import setup_logger  # Import hàm cài đặt logger từ module local

# Khởi tạo logger cho module inference
logger = setup_logger("inference")


def save_model(
    trainer: Trainer,  # Nhận vào đối tượng Trainer đã huấn luyện xong
    tokenizer: PreTrainedTokenizerBase,  # Nhận vào tokenizer tương ứng với mô hình
    config: PipelineConfig,  # Nhận vào cấu hình pipeline của toàn bộ quá trình
    metrics: dict[str, Any],  # Nhận vào từ điển chứa các chỉ số đánh giá (metrics)
) -> str:  # Hàm trả về đường dẫn tới thư mục lưu mô hình
    """
    Lưu mô hình, tokenizer, cấu hình sinh văn bản, cấu hình pipeline và kết quả huấn luyện.
    """
    # Lấy đường dẫn thư mục lưu mô hình từ cấu hình
    model_dir = config.output.model_dir
    
    # Tạo thư mục nếu chưa tồn tại, cho phép tạo các thư mục cha (exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # Ghi log thông báo bắt đầu lưu mô hình
    logger.info(f"Bắt đầu lưu mô hình tại: {model_dir}")
    
    # Lưu trọng số mô hình thông qua đối tượng trainer
    trainer.save_model(model_dir)
    # Lưu tokenizer vào cùng thư mục
    tokenizer.save_pretrained(model_dir)
    
    # Tạo cấu hình sinh văn bản với các tham số mặc định hợp lý (nhiệt độ=0.7, top_p=0.9, ...)
    gen_config = GenerationConfig(
        temperature=0.7,  # Mức độ ngẫu nhiên khi sinh từ
        top_p=0.9,  # Chọn các từ có tổng xác suất tích lũy là 0.9
        repetition_penalty=1.1,  # Phạt các từ bị lặp lại
        do_sample=True,  # Bật chế độ lấy mẫu thay vì greedy search
        pad_token_id=tokenizer.eos_token_id,  # Gán token padding bằng token kết thúc chuỗi (cho GPT-2)
    )
    # Lưu cấu hình sinh văn bản vào thư mục
    gen_config.save_pretrained(model_dir)
    
    # Định nghĩa đường dẫn file cấu hình YAML để đảm bảo tính tái tạo
    config_path = os.path.join(model_dir, "training_config.yaml")
    # Gọi hàm lưu pipeline config
    save_config(config, config_path)
    
    # Định nghĩa đường dẫn file JSON chứa metrics
    metrics_path = os.path.join(model_dir, "training_metrics.json")
    # Mở file để ghi kết quả đánh giá (metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        # Chuyển đổi từ điển metrics sang chuỗi JSON và ghi vào file
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    # Lấy danh sách các file đã được lưu trong thư mục
    saved_files = os.listdir(model_dir)
    # Tính tổng kích thước của thư mục bằng cách cộng dồn kích thước các file
    total_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in saved_files if os.path.isfile(os.path.join(model_dir, f)))
    # Chuyển đổi kích thước từ byte sang MB
    total_size_mb = total_size / (1024 * 1024)
    
    # Ghi log báo cáo danh sách file và tổng dung lượng
    logger.info(f"Đã lưu các file: {saved_files}")
    # Ghi log tổng dung lượng của thư mục lưu mô hình
    logger.info(f"Tổng dung lượng: {total_size_mb:.2f} MB")
    
    # Trả về đường dẫn của thư mục chứa mô hình
    return model_dir


def load_model(model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Tải mô hình từ thư mục, tự động nhận diện mô hình gốc hay LoRA.
    """
    # Ghi log thông báo bắt đầu tải mô hình
    logger.info(f"Bắt đầu tải mô hình từ: {model_path}")
    
    # Đường dẫn kiểm tra file cấu hình LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    # Kiểm tra xem file cấu hình LoRA có tồn tại không
    is_lora = os.path.exists(adapter_config_path)
    
    # Đường dẫn kiểm tra file cấu hình mô hình (config.json)
    model_config_path = os.path.join(model_path, "config.json")
    # Khởi tạo cờ nhận diện bài toán phân loại là False
    is_classification = False
    # Nếu file cấu hình mô hình tồn tại, mở ra kiểm tra
    if os.path.exists(model_config_path):
        # Đọc nội dung file config.json
        with open(model_config_path, "r", encoding="utf-8") as f:
            # Parse nội dung JSON thành từ điển
            cfg = json.load(f)
            # Kiểm tra xem id bài toán phân loại có nằm trong config hay không
            if "id2label" in cfg or "label2id" in cfg:
                # Đánh dấu đây là mô hình phân loại (classification)
                is_classification = True
    elif is_lora:  # Nếu là LoRA thì cấu hình mô hình gốc nằm ở adapter_config
        # Đọc nội dung file adapter_config.json
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            # Parse cấu hình LoRA
            peft_cfg = json.load(f)
            # Kiểm tra task_type xem có phải phân loại chuỗi không
            if peft_cfg.get("task_type") == "SEQ_CLS":
                # Đánh dấu đây là bài toán phân loại
                is_classification = True

    # Tải tokenizer từ thư mục lưu
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Đảm bảo token padding được thiết lập (GPT-2 không có padding token mặc định)
    if tokenizer.pad_token is None:
        # Gán token padding bằng token kết thúc chuỗi (eos_token)
        tokenizer.pad_token = tokenizer.eos_token
        
    # Thiết lập padding side cho bài toán phân loại theo best practices (trái/phải)
    if is_classification:
        # Với bài toán classification dùng GPT-2, cần padding bên trái
        tokenizer.padding_side = "left"
        # Ghi log báo cài đặt padding trái cho bài toán phân loại
        logger.info("Đã cấu hình padding_side='left' cho bài toán classification.")
    else:
        # Với các bài toán sinh văn bản thông thường, thường dùng padding bên phải
        tokenizer.padding_side = "right"
        
    # Nhánh xử lý nếu mô hình được huấn luyện bằng LoRA
    if is_lora:
        # Ghi log phát hiện mô hình LoRA
        logger.info("Phát hiện adapter LoRA, tiến hành tải base model và merge.")
        # Tải cấu hình Peft từ thư mục
        peft_config = PeftConfig.from_pretrained(model_path)
        # Lấy tên hoặc đường dẫn mô hình gốc từ PeftConfig
        base_model_name = peft_config.base_model_name_or_path
        
        # Nếu bài toán là phân loại văn bản
        if is_classification:
            # Tải mô hình cơ sở cho phân loại
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
        else:
            # Tải mô hình cơ sở cho ngôn ngữ sinh (causal LM)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
        # Nạp LoRA adapter vào mô hình cơ sở
        model = PeftModel.from_pretrained(base_model, model_path)
        # Hợp nhất trọng số LoRA vào mô hình gốc và giải phóng bộ nhớ của adapter
        model = model.merge_and_unload()
    else:  # Nhánh xử lý nếu là mô hình fine-tune toàn bộ (full fine-tuning)
        # Ghi log phát hiện mô hình full
        logger.info("Tải mô hình full fine-tuning.")
        # Tùy thuộc vào loại bài toán để gọi AutoModel phù hợp
        if is_classification:
            # Tải mô hình nguyên bản cho phân loại chuỗi
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # Tải mô hình nguyên bản cho sinh văn bản
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
    # Ghi log quá trình tải mô hình đã hoàn tất
    logger.info("Tải mô hình và tokenizer thành công.")
    # Trả về bộ (mô hình, tokenizer) cho inference
    return model, tokenizer


def generate_text(
    model: PreTrainedModel,  # Nhận vào mô hình đã được tải lên
    tokenizer: PreTrainedTokenizerBase,  # Nhận vào tokenizer tương ứng
    prompts: list[str],  # Nhận vào danh sách các câu prompt đầu vào
    **kwargs: Any,  # Các tham số sinh văn bản tùy chọn (như max_new_tokens, temperature)
) -> list[str]:  # Hàm trả về danh sách các câu văn bản được sinh ra
    """
    Sinh văn bản dựa trên danh sách các câu prompt.
    """
    # Ghi log số lượng prompt đang xử lý
    logger.info(f"Đang sinh văn bản cho {len(prompts)} prompts...")
    
    # Khai báo các tham số mặc định cho thuật toán sinh văn bản
    generation_kwargs = {
        "max_new_tokens": 100,  # Số token mới tối đa được sinh ra thêm
        "temperature": 0.7,  # Nhiệt độ (ảnh hưởng tính ngẫu nhiên)
        "top_p": 0.9,  # Ngưỡng tích lũy xác suất cho quá trình lấy mẫu
        "do_sample": True,  # Bật tính năng lấy mẫu ngẫu nhiên (sampling)
        "repetition_penalty": 1.1,  # Trọng số phạt cho các từ bị lặp lại
        "pad_token_id": tokenizer.pad_token_id,  # ID của token padding để điền vào chuỗi
        "eos_token_id": tokenizer.eos_token_id,  # ID của token kết thúc chuỗi để dừng quá trình
    }
    # Cập nhật các tham số mặc định bằng các tham số người dùng truyền vào (nếu có)
    generation_kwargs.update(kwargs)
    
    # Di chuyển mô hình lên GPU nếu có sẵn thiết bị CUDA, ngược lại dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Gửi trọng số mô hình lên thiết bị tương ứng
    model.to(device)
    # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
    model.eval()
    
    # Tokenize danh sách prompt (thêm padding, truncation, dạng tensor cho PyTorch)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    # Chuyển dữ liệu đầu vào lên cùng thiết bị với mô hình
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Không tính toán gradient trong bước sinh văn bản để tiết kiệm bộ nhớ
    with torch.no_grad():
        # Gọi hàm generate của mô hình với input và các tham số sinh
        outputs = model.generate(**inputs, **generation_kwargs)
        
    # Giải mã các token dự đoán thành văn bản con người đọc được, bỏ qua các token đặc biệt (như padding)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Ghi log báo hiệu quá trình sinh kết thúc
    logger.info("Hoàn tất sinh văn bản.")
    # Trả về danh sách văn bản
    return generated_texts
