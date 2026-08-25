"""
Data Pipeline Module.
Xử lý dữ liệu: tải, tiền xử lý, tăng cường dữ liệu và chuẩn bị cho mô hình.
"""

import os
import random
from typing import Any, Tuple

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    DataCollatorForCompletionOnlyLM = None

from src.config import PipelineConfig
from src.utils import setup_logger, StepTimer

# Khởi tạo logger cho data module
logger = setup_logger("data")

def _load_tokenizer(config: PipelineConfig) -> PreTrainedTokenizerBase:
    """
    Tải tokenizer và thiết lập các token cần thiết.
    """
    # Tải tokenizer từ tên mô hình hoặc đường dẫn
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    # GPT-2 không có pad_token mặc định, cần gán bằng eos_token
    if tokenizer.pad_token is None:
        # Gán pad_token bằng eos_token
        tokenizer.pad_token = tokenizer.eos_token
        
    # Đối với bài toán phân loại, mô hình sinh dự đoán ở token cuối cùng, nên phải padding bên trái
    if config.model.task_type == "classification":
        # Thiết lập padding bên trái cho task phân loại
        tokenizer.padding_side = "left"
    else:
        # Thiết lập padding bên phải cho task sinh văn bản (causal LM)
        tokenizer.padding_side = "right"
        
    # Trả về tokenizer đã được cấu hình
    return tokenizer

def _load_raw_data(config: PipelineConfig) -> DatasetDict | Dataset:
    """
    Tải dữ liệu thô từ HuggingFace Hub hoặc local file.
    """
    # Kiểm tra nguồn dữ liệu là huggingface hay local
    if config.data.source == "huggingface":
        # Tải từ HuggingFace Hub
        logger.info(f"Đang tải dataset {config.data.dataset_name} từ HuggingFace Hub.")
        
        # Tạo dictionary chứa các tham số tùy chọn
        kwargs = {}
        # Nếu có cấu hình cụ thể cho dataset (dataset_config)
        if hasattr(config.data, 'dataset_config') and config.data.dataset_config:
            # Truyền dataset_config vào tham số name
            kwargs["name"] = config.data.dataset_config
            
        # Gọi hàm load_dataset của thư viện datasets
        dataset = load_dataset(config.data.dataset_name, **kwargs)
        # Trả về dataset vừa tải
        return dataset
        
    elif config.data.source == "local":
        # Tải từ local file (csv, json, text)
        logger.info(f"Đang tải dataset từ file local: {config.data.local_path}")
        
        # Kiểm tra xem đường dẫn file có tồn tại không
        if not config.data.local_path or not os.path.exists(config.data.local_path):
            # Ném lỗi nếu file không tồn tại
            raise FileNotFoundError(f"Không tìm thấy file: {config.data.local_path}")
        
        # Xác định định dạng file dựa trên phần mở rộng của file
        ext = config.data.local_path.split(".")[-1].lower()
        
        # Kiểm tra định dạng có được hỗ trợ hay không
        if ext in ["csv", "json", "txt"]:
            # Ánh xạ đuôi txt thành định dạng text
            file_type = "text" if ext == "txt" else ext
            # Tải dataset từ file local
            dataset = load_dataset(file_type, data_files=config.data.local_path)
            # Trả về dataset đã tải
            return dataset
        else:
            # Báo lỗi nếu định dạng file không nằm trong danh sách hỗ trợ
            raise ValueError(f"Định dạng file không hỗ trợ: {ext}")
    else:
        # Báo lỗi nếu nguồn dữ liệu không hợp lệ
        raise ValueError(f"Nguồn dữ liệu không hợp lệ: {config.data.source}")

def _split_dataset(dataset: DatasetDict | Dataset, ratios: tuple) -> DatasetDict:
    """
    Chia dataset thành train, validation, test nếu chưa được chia.
    """
    # Nếu đã là DatasetDict có đủ các phần train và validation thì trả về luôn
    if isinstance(dataset, DatasetDict) and "train" in dataset and "validation" in dataset:
        # Ghi log thông báo dataset đã chia sẵn
        logger.info("Dataset đã được chia sẵn.")
        # Trả về nguyên bản dataset
        return dataset
        
    # Nếu chỉ có train, gộp lại thành 1 Dataset để chia
    if isinstance(dataset, DatasetDict):
        # Lấy khóa của phần tử đầu tiên (thường là "train" hoặc "default")
        key = list(dataset.keys())[0]
        # Lấy dữ liệu của phần tử đó
        dataset = dataset[key]
        
    # Xác định tỷ lệ chia từ tham số đầu vào
    train_ratio, val_ratio, test_ratio = ratios
    # Ghi log quá trình chia tỷ lệ
    logger.info(f"Đang chia dataset với tỷ lệ {train_ratio}:{val_ratio}:{test_ratio}")
    
    # Tính tỷ lệ thực tế cho test set
    test_size = test_ratio / (train_ratio + val_ratio + test_ratio)
    # Tách test set từ dataset gốc với seed cố định
    split_1 = dataset.train_test_split(test_size=test_size, seed=42)
    # Lấy tập test
    test_set = split_1["test"]
    
    # Tính tỷ lệ thực tế cho validation set từ phần còn lại
    val_size = val_ratio / (train_ratio + val_ratio)
    # Tách train và validation set
    split_2 = split_1["train"].train_test_split(test_size=val_size, seed=42)
    
    # Trả về một đối tượng DatasetDict mới chứa đủ 3 phần
    return DatasetDict({
        "train": split_2["train"],
        "validation": split_2["test"],
        "test": test_set
    })

def _filter_and_clean(dataset: DatasetDict, text_column: str) -> DatasetDict:
    """
    Lọc bỏ các mẫu dữ liệu rỗng hoặc quá ngắn.
    """
    # Ghi log bắt đầu lọc
    logger.info("Đang lọc bỏ các mẫu văn bản ngắn hơn 10 ký tự.")
    # Duyệt qua từng split trong dataset (train, validation, test)
    for split in dataset.keys():
        # Lọc dữ liệu bằng hàm lambda, yêu cầu chiều dài sau khi strip > 10
        dataset[split] = dataset[split].filter(lambda x: len(str(x[text_column]).strip()) > 10)
    # Trả về dataset đã làm sạch
    return dataset

def _augment_data(dataset: DatasetDict, config: PipelineConfig) -> DatasetDict:
    """
    Tăng cường dữ liệu bằng các phép biến đổi đơn giản trên tập train.
    """
    # Nếu tính năng augmentation bị tắt hoặc không có kỹ thuật nào được chọn thì bỏ qua
    if not config.data.augmentation.enabled or not config.data.augmentation.techniques:
        # Trả về dataset giữ nguyên
        return dataset
        
    # Ghi log tiến hành augmentation
    logger.info("Đang tiến hành tăng cường dữ liệu trên tập train.")
    # Lấy tập dữ liệu train
    train_set = dataset["train"]
    # Lấy tên cột chứa text
    text_col = config.data.text_column
    
    # Tính số lượng mẫu cần augment dựa trên augment_ratio
    num_samples = int(len(train_set) * config.data.augmentation.augment_ratio)
    
    def random_delete(words, p=0.15):
        """Xóa ngẫu nhiên các từ với xác suất p."""
        # Nếu chuỗi chỉ có 1 từ, trả về luôn để tránh mảng rỗng
        if len(words) == 1:
            return words
        # Tạo danh sách các từ giữ lại thông qua sinh ngẫu nhiên
        new_words = [w for w in words if random.random() > p]
        # Nếu xóa hết các từ, chọn lại ngẫu nhiên một từ trong danh sách gốc
        if len(new_words) == 0:
            return [random.choice(words)]
        # Trả về danh sách từ mới
        return new_words

    def random_swap(words, n=1):
        """Đổi vị trí ngẫu nhiên n cặp từ trong văn bản."""
        # Tạo bản sao danh sách từ
        new_words = words.copy()
        # Lặp n lần để thực hiện swap
        for _ in range(n):
            # Chỉ swap nếu số lượng từ >= 2
            if len(new_words) >= 2:
                # Chọn ngẫu nhiên 2 chỉ số
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                # Đổi chỗ 2 phần tử
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        # Trả về kết quả sau khi swap
        return new_words

    # Chọn ngẫu nhiên các chỉ số mẫu để augment (số lượng tối đa là len(train_set))
    augment_indices = random.sample(range(len(train_set)), min(num_samples, len(train_set)))
    # Khởi tạo dictionary chứa dữ liệu được augment
    augmented_data = {col: [] for col in train_set.column_names}
    
    # Lặp qua các chỉ số đã chọn
    for idx in augment_indices:
        # Lấy từng item trong tập train
        item = train_set[idx]
        # Lấy text và chuyển về chuỗi
        text = str(item[text_col])
        # Tách chuỗi thành danh sách từ
        words = text.split()
        
        # Chọn ngẫu nhiên một kỹ thuật augmentation từ danh sách cấu hình
        technique = random.choice(config.data.augmentation.techniques)
        
        # Áp dụng kỹ thuật tương ứng
        if technique == "random_delete":
            augmented_words = random_delete(words)
        elif technique == "random_swap":
            augmented_words = random_swap(words)
        else:
            # Nếu không tìm thấy technique hợp lệ thì giữ nguyên
            augmented_words = words
            
        # Nối danh sách từ lại thành câu văn bản
        new_text = " ".join(augmented_words)
        
        # Ghi các cột vào từ điển augmented_data
        for col in item.keys():
            if col == text_col:
                # Lưu câu văn bản đã biến đổi
                augmented_data[col].append(new_text)
            else:
                # Lưu nguyên các giá trị khác
                augmented_data[col].append(item[col])
                
    # Tạo đối tượng Dataset từ dictionary chứa dữ liệu augmented
    aug_dataset = Dataset.from_dict(augmented_data)
    
    # Nối dữ liệu augmented vào tập train ban đầu
    dataset["train"] = concatenate_datasets([train_set, aug_dataset])
    # Ghi log số lượng mẫu được tăng cường
    logger.info(f"Đã thêm {len(aug_dataset)} mẫu vào tập train.")
    
    # Trả về toàn bộ dataset
    return dataset

def _tokenize_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, config: PipelineConfig) -> DatasetDict:
    """
    Tokenize dataset mà không sử dụng static padding.
    """
    # Ghi log bắt đầu tokenization
    logger.info("Đang tiến hành tokenize dataset.")
    # Lấy tên cột text từ config
    text_col = config.data.text_column
    # Lấy độ dài tối đa (max_length)
    max_length = config.data.max_length
    
    def tokenize_function(examples):
        """Hàm xử lý cho map."""
        # Gọi tokenizer, cắt đuôi nếu quá dài (truncation=True)
        # Bắt buộc padding=False để thực hiện DYNAMIC PADDING tại DataCollator
        result = tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
            padding=False
        )
        # Trả về kết quả
        return result
        
    # Áp dụng hàm map cho tất cả tập dữ liệu theo batch
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
    )
    
    # Trả về dataset đã mã hóa
    return tokenized_dataset

def _get_data_collator(tokenizer: PreTrainedTokenizerBase, config: PipelineConfig) -> Any:
    """
    Trả về DataCollator phù hợp dựa trên loại task_type.
    """
    # Lấy task_type từ cấu hình mô hình
    task_type = config.model.task_type
    # Ghi log task_type
    logger.info(f"Khởi tạo DataCollator cho task: {task_type}")
    
    # Xử lý theo từng task cụ thể
    if task_type == "causal_lm":
        # Cho Causal LM, tắt chức năng Masked Language Modeling
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif task_type == "classification":
        # Cho Classification, sử dụng padding động theo độ dài lớn nhất trong batch
        return DataCollatorWithPadding(tokenizer=tokenizer)
    elif task_type == "completion":
        # Cho Instruction tuning/completion, dùng DataCollatorForCompletionOnlyLM
        if DataCollatorForCompletionOnlyLM is None:
            # Báo lỗi nếu thiếu thư viện trl
            raise ImportError("Không tìm thấy thư viện trl. Hãy cài đặt: pip install trl")
        # Định nghĩa template của phần response (thường là "### Response:\n")
        response_template = "### Response:\n"
        # Trả về data collator cho phần completion
        return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)
    else:
        # Nếu task không hỗ trợ, ném ra ngoại lệ
        raise ValueError(f"Task type không hỗ trợ: {task_type}")

def prepare_data(config: PipelineConfig) -> tuple[DatasetDict, Any, PreTrainedTokenizerBase]:
    """
    Pipeline chính để chuẩn bị data.
    """
    # Sử dụng bộ đếm thời gian cho bước Data Preparation
    with StepTimer("Data Preparation"):
        # 1. Khởi tạo và nạp tokenizer
        tokenizer = _load_tokenizer(config)
        
        # 2. Lấy dữ liệu thô từ cấu hình chỉ định
        raw_dataset = _load_raw_data(config)
        
        # 3. Phân chia dữ liệu thành tập train/validation/test
        dataset = _split_dataset(raw_dataset, config.data.split_ratios)
        
        # 4. Loại bỏ các dữ liệu rác, ngắn hơn giới hạn
        dataset = _filter_and_clean(dataset, config.data.text_column)
        
        # 5. Sinh thêm dữ liệu huấn luyện (augmentation) nếu được cấu hình
        dataset = _augment_data(dataset, config)
        
        # 6. Chuyển đổi văn bản sang token (chỉ truncation, không static padding)
        tokenized_dataset = _tokenize_dataset(dataset, tokenizer, config)
        
        # 7. Khởi tạo DataCollator tương ứng với bài toán
        data_collator = _get_data_collator(tokenizer, config)
        
        # Ghi log hoàn thành quá trình
        logger.info("Hoàn tất chuẩn bị dữ liệu.")
        # Lặp in ra số lượng dòng dữ liệu theo từng split
        for split, ds in tokenized_dataset.items():
            logger.info(f" - {split}: {len(ds)} samples")
            
        # Trả về bộ ba gồm tập dữ liệu, collator, và tokenizer
        return tokenized_dataset, data_collator, tokenizer
