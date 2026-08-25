# Import các thư viện chuẩn (stdlib)
import json
import os
from typing import Any, Callable, Optional, Tuple

# Import các thư viện của bên thứ ba (third-party)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Import các module nội bộ (local)
from src.config import PipelineConfig
from src.utils import setup_logger

# Khởi tạo logger cho module model
logger = setup_logger("model")


def create_model_init(config: PipelineConfig) -> Callable[[Optional[Any]], PreTrainedModel]:
    # Hàm này trả về một hàm khởi tạo mô hình (model_init)
    # Hàm này được sử dụng bởi Trainer, hỗ trợ Optuna
    def model_init(trial: Optional[Any] = None) -> PreTrainedModel:
        # Lấy tên mô hình từ cấu hình
        model_name = config.model.name
        # Lấy loại tác vụ (task_type) từ cấu hình
        task_type = config.model.task_type

        # Ghi log quá trình khởi tạo mô hình
        logger.info(f"Khởi tạo mô hình {model_name} cho tác vụ {task_type}")

        # Kiểm tra nếu tác vụ là causal_lm hoặc completion
        if task_type in ["causal_lm", "completion"]:
            # Tải mô hình GPT-2 cho mô hình ngôn ngữ nhân quả
            model = GPT2LMHeadModel.from_pretrained(model_name)
            # Xác định loại tác vụ cho LoRA
            peft_task_type = TaskType.CAUSAL_LM
        # Kiểm tra nếu tác vụ là phân loại (classification)
        elif task_type == "classification":
            # Lấy số lượng nhãn từ cấu hình
            num_labels = config.model.num_labels
            # Tải mô hình GPT-2 cho phân loại chuỗi
            model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # Xác định loại tác vụ cho LoRA
            peft_task_type = TaskType.SEQ_CLS
        # Nếu loại tác vụ không hợp lệ, ném ngoại lệ
        else:
            # Ném lỗi ValueError báo loại tác vụ không được hỗ trợ
            raise ValueError(f"Loại tác vụ không được hỗ trợ: {task_type}")

        # Đặt pad_token_id bằng eos_token_id vì GPT-2 không có pad_token mặc định
        model.config.pad_token_id = model.config.eos_token_id

        # Kiểm tra nếu cấu hình PEFT (LoRA) được bật
        if config.peft.enabled:
            # Lấy giá trị rank cho LoRA từ cấu hình
            lora_r = config.peft.r
            # Nếu đang chạy Optuna (trial không phải None)
            if trial is not None:
                # Gợi ý giá trị rank từ các lựa chọn [8, 16, 32]
                lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
                # Ghi log giá trị rank được gợi ý
                logger.info(f"Optuna gợi ý lora_r = {lora_r}")

            # Khởi tạo cấu hình LoRA với các tham số từ config và target modules của GPT-2
            lora_config = LoraConfig(
                task_type=peft_task_type,
                r=lora_r,
                lora_alpha=config.peft.lora_alpha,
                lora_dropout=config.peft.lora_dropout,
                target_modules=["c_attn", "c_proj", "c_fc"]
            )
            # Bọc mô hình gốc bằng PEFT model
            model = get_peft_model(model, lora_config)
            # In ra số lượng tham số có thể huấn luyện
            model.print_trainable_parameters()

        # Trả về mô hình đã được khởi tạo
        return model

    # Trả về hàm model_init
    return model_init


def load_model_for_inference(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # Ghi log bắt đầu tải mô hình cho inference
    logger.info(f"Tải mô hình từ đường dẫn: {model_path}")
    
    # Tải tokenizer từ cùng một đường dẫn
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Đường dẫn đến file cấu hình adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    # Kiểm tra xem mô hình có phải là LoRA adapter hay không
    if os.path.exists(adapter_config_path):
        # Ghi log phát hiện mô hình LoRA
        logger.info("Phát hiện adapter_config.json, tiến hành tải mô hình LoRA")
        
        # Mở file adapter_config.json để đọc
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            # Phân tích nội dung JSON
            adapter_config = json.load(f)
            
        # Lấy tên mô hình gốc từ cấu hình adapter
        base_model_name = adapter_config.get("base_model_name_or_path")
        # Lấy loại tác vụ từ cấu hình adapter
        peft_task_type = adapter_config.get("task_type")
        
        # Kiểm tra nếu tác vụ là phân loại chuỗi
        if peft_task_type == "SEQ_CLS":
            # Tải cấu hình mô hình từ thư mục hiện tại để lấy số lượng nhãn
            model_config = AutoConfig.from_pretrained(model_path)
            # Lấy số lượng nhãn, mặc định là 2
            num_labels = getattr(model_config, "num_labels", 2)
            # Tải mô hình cơ sở cho phân loại chuỗi
            base_model = GPT2ForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
            # Đặt padding_side là left cho tokenizer phân loại
            tokenizer.padding_side = "left"
        # Nếu không phải tác vụ phân loại
        else:
            # Tải mô hình cơ sở cho sinh văn bản
            base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
            
        # Tải mô hình PEFT kết hợp mô hình cơ sở và adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        # Gộp trọng số adapter vào mô hình cơ sở và giải phóng bộ nhớ của adapter
        model = model.merge_and_unload()
    # Nếu không tìm thấy file cấu hình adapter
    else:
        # Ghi log tải mô hình đầy đủ (full model)
        logger.info("Tải mô hình đầy đủ (full model)")
        
        # Tải cấu hình mô hình
        model_config = AutoConfig.from_pretrained(model_path)
        
        # Lấy danh sách kiến trúc mô hình
        architectures = getattr(model_config, "architectures", [])
        # Kiểm tra xem kiến trúc có chứa ForSequenceClassification không
        if architectures and any("ForSequenceClassification" in arch for arch in architectures):
            # Tải mô hình phân loại chuỗi từ đường dẫn
            model = GPT2ForSequenceClassification.from_pretrained(model_path)
            # Đặt padding_side là left cho tokenizer phân loại
            tokenizer.padding_side = "left"
        # Nếu không chứa ForSequenceClassification
        else:
            # Tải mô hình sinh ngôn ngữ nhân quả từ đường dẫn
            model = GPT2LMHeadModel.from_pretrained(model_path)
            
    # Kiểm tra xem tokenizer đã có pad_token chưa
    if tokenizer.pad_token is None:
        # Đặt pad_token bằng eos_token
        tokenizer.pad_token = tokenizer.eos_token
        
    # Đặt pad_token_id bằng eos_token_id trong cấu hình mô hình
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Trả về mô hình và tokenizer đã sẵn sàng cho inference
    return model, tokenizer
