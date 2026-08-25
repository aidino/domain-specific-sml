# GPT-2 Fine-Tuning Pipeline — Design Specification

> **Ngày tạo:** 2026-08-25
> **Trạng thái:** Approved
> **Vị trí code:** `finetune-gpt2/`

---

## 1. Tổng Quan (Overview)

Pipeline chuyên nghiệp để fine-tune GPT-2 model, được thiết kế **module hóa** với các đặc điểm:

- **Reusable**: Dùng lại cho nhiều dataset và task khác nhau qua config YAML
- **Multi-task**: Hỗ trợ causal LM, text classification, instruction completion
- **Multi-hardware**: CPU, single GPU, multi-GPU (auto-detect)
- **Professional logging**: Rich console output, TensorBoard integration
- **Best practices**: LoRA/PEFT, Optuna HP search, early stopping, mixed precision

### Phạm vi (Scope)

| Trong phạm vi | Ngoài phạm vi |
|---|---|
| Fine-tune GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl) | Pre-training from scratch |
| HuggingFace Hub + local data (CSV/JSON/TXT) | Streaming datasets |
| Optuna hyperparameter search | Ray Tune, Weights & Biases |
| LoRA/PEFT efficient fine-tuning | QLoRA, quantization |
| TensorBoard visualization | MLflow, Neptune |
| Single node (1+ GPUs) | Multi-node distributed training |

---

## 2. Kiến Trúc (Architecture)

### 2.1 Cấu Trúc Thư Mục

```
finetune-gpt2/
├── configs/                      # YAML configs cho từng experiment
│   ├── default.yaml              # Config mặc định cho causal LM text generation
│   ├── classification.yaml       # Config mẫu cho text classification
│   └── lora.yaml                 # Config mẫu cho LoRA fine-tuning
├── src/                          # Source code chính
│   ├── __init__.py               # Package init
│   ├── config.py                 # Dataclass config + YAML loader + CLI override
│   ├── data.py                   # Data loading, preprocessing, augmentation
│   ├── model.py                  # Model factory (full / LoRA / classification)
│   ├── hp_search.py              # Optuna hyperparameter search
│   ├── trainer.py                # Training orchestration với custom callbacks
│   ├── evaluate.py               # Validation & test với metrics theo task
│   ├── inference.py              # Save/load model + text generation
│   ├── callbacks.py              # Custom callbacks (logging, generation, TensorBoard)
│   └── utils.py                  # Logger, device detection, seed, timer
├── main.py                       # CLI entry point - chạy full hoặc từng step
├── compare.py                    # So sánh experiments
└── requirements.txt              # Dependencies
```

### 2.2 Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐
│    Config    │───▶│     Data     │───▶│   HP Search   │───▶│  Train   │───▶│ Evaluate │───▶│    Save    │
│  (YAML +    │    │ Preparation  │    │   (Optuna)    │    │  Model   │    │  (Test)  │    │   Model    │
│  Dataclass) │    │              │    │  (optional)   │    │          │    │          │    │            │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────┘    └──────────┘    └────────────┘
       │                                                          │
       │                                                          ▼
       │                                                   ┌──────────────┐
       │                                                   │  TensorBoard │
       │                                                   │   Logging    │
       └───────────────────────────────────────────────────│──────────────│
                                                           └──────────────┘
```

### 2.3 Module Dependencies

```
main.py
  ├── src/config.py      ← YAML parsing, dataclass validation
  ├── src/utils.py       ← Logger, device, seed (used by all modules)
  ├── src/data.py         ← Load & tokenize data
  │     └── depends on: config, utils
  ├── src/model.py        ← Create model (full/LoRA)
  │     └── depends on: config, utils
  ├── src/hp_search.py    ← Optuna search
  │     └── depends on: config, model, data, utils
  ├── src/trainer.py      ← Training orchestration
  │     └── depends on: config, model, data, callbacks, utils
  ├── src/callbacks.py    ← Custom Trainer callbacks
  │     └── depends on: utils
  ├── src/evaluate.py     ← Test evaluation
  │     └── depends on: config, utils
  └── src/inference.py    ← Save/load/generate
        └── depends on: config, utils
```

---

## 3. Chi Tiết Thiết Kế Từng Module

### 3.1 `src/config.py` — Config System

**Mục đích:** Parse YAML config vào nested dataclasses, validate types, hỗ trợ CLI override.

**Dataclass structure:**

```python
@dataclass
class ModelConfig:
    name: str = "gpt2"                    # Tên model HuggingFace hoặc local path
    task_type: str = "causal_lm"          # "causal_lm" | "classification" | "completion"
    num_labels: int = 2                   # Số labels (chỉ cho classification)

@dataclass
class PeftConfig:
    enabled: bool = False                 # Bật/tắt LoRA
    r: int = 16                           # LoRA rank
    lora_alpha: int = 32                  # Scaling factor
    lora_dropout: float = 0.05            # Dropout
    target_modules: list = ("c_attn",)    # Target layers cho GPT-2

@dataclass
class DataConfig:
    source: str = "huggingface"           # "huggingface" | "local"
    dataset_name: str = "wikitext"        # HuggingFace dataset name
    dataset_config: str = "wikitext-2-raw-v1"
    local_path: str = None                # Path to local file
    text_column: str = "text"             # Tên cột text
    label_column: str = "label"           # Tên cột label
    max_length: int = 512                 # Max sequence length
    split_ratios: tuple = (0.8, 0.1, 0.1) # Train/val/test split
    augmentation_enabled: bool = False
    augmentation_techniques: list = ()
    augmentation_ratio: float = 0.2

@dataclass
class TrainingConfig:
    num_epochs: int = 5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    mixed_precision: str = "auto"         # "auto" | "bf16" | "fp16" | "none"

@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 3
    threshold: float = 0.001

@dataclass
class HPSearchConfig:
    enabled: bool = False
    n_trials: int = 15
    direction: str = "minimize"
    metric: str = "eval_loss"
    search_space: dict = None             # Parsed from YAML

@dataclass
class LoggingConfig:
    level: str = "INFO"
    logging_steps: int = 50
    tensorboard_dir: str = "./runs"
    experiment_name: str = "gpt2-finetune"

@dataclass
class OutputConfig:
    checkpoint_dir: str = "./checkpoints"
    model_dir: str = "./final_model"
    save_total_limit: int = 3
    save_strategy: str = "epoch"          # "epoch" | "steps"
    save_steps: int = 500

@dataclass
class PipelineConfig:
    model: ModelConfig
    peft: PeftConfig
    data: DataConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    hp_search: HPSearchConfig
    logging: LoggingConfig
    output: OutputConfig
```

**Key functions:**
- `load_config(yaml_path, cli_overrides) -> PipelineConfig`: Load YAML, merge CLI overrides, validate, return typed config
- `save_config(config, path)`: Save config back to YAML (for reproducibility)

**CLI override mechanism:**
- Dùng `argparse` với format `--section.field value`
- VD: `--training.batch_size 4` override `training.batch_size` trong YAML
- Nested dot notation: parse string → navigate dataclass hierarchy → set value

---

### 3.2 `src/data.py` — Data Pipeline

**Mục đích:** Load, preprocess, tokenize data từ nhiều nguồn. Trả về datasets sẵn sàng cho training.

**Public API:**

```python
def prepare_data(config: PipelineConfig) -> tuple[DatasetDict, DataCollator, PreTrainedTokenizer]:
    """
    Pipeline chính để chuẩn bị data.

    Returns:
        datasets: DatasetDict với keys "train", "validation", "test"
        data_collator: DataCollator phù hợp với task_type
        tokenizer: Tokenizer đã cấu hình pad_token
    """
```

**Internal flow:**

1. **`_load_tokenizer(config)`** → Load tokenizer, set `pad_token = eos_token`
2. **`_load_raw_data(config)`** → Load từ Hub hoặc local file
   - Hub: `datasets.load_dataset(name, config)`
   - Local: `datasets.load_dataset("csv"/"json"/"text", data_files=path)`
3. **`_split_dataset(dataset, ratios)`** → Chia train/val/test nếu chưa có split
4. **`_filter_and_clean(dataset)`** → Loại bỏ samples rỗng, quá ngắn
5. **`_augment_data(train_dataset, config)`** → Data augmentation trên train set only
6. **`_tokenize(dataset, tokenizer, config)`** → Tokenize với `truncation=True`, KHÔNG pad cố định
7. **`_get_data_collator(tokenizer, config)`** → Trả collator theo task:
   - `causal_lm` → `DataCollatorForLanguageModeling(mlm=False)`
   - `classification` → `DataCollatorWithPadding`
   - `completion` → `DataCollatorForCompletionOnlyLM`

**Data augmentation:**
- Import `nlpaug` hoặc implement đơn giản:
  - `synonym_replace`: Thay từ bằng synonym (dùng WordNet hoặc word embeddings)
  - `random_delete`: Xóa random 10-20% từ
  - `random_swap`: Swap vị trí 2 từ ngẫu nhiên
  - `random_insert`: Chèn từ ngẫu nhiên từ vocabulary
- Augmentation tạo thêm samples mới, append vào train set
- Chỉ augment `augment_ratio` % của train set

---

### 3.3 `src/model.py` — Model Factory

**Mục đích:** Tạo model đúng loại, hỗ trợ cả full fine-tuning và LoRA.

**Public API:**

```python
def create_model_init(config: PipelineConfig) -> Callable:
    """
    Trả về model_init function cho Trainer.
    model_init(trial=None) -> PreTrainedModel

    Khi trial=None: tạo model với config mặc định
    Khi trial có giá trị: LoRA rank có thể được Optuna search
    """

def load_model_for_inference(model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model đã save cho inference.
    Auto-detect: full model vs LoRA adapter.
    """
```

**Model creation logic:**

1. Load base model theo `task_type`:
   - `causal_lm` / `completion` → `GPT2LMHeadModel.from_pretrained(name)`
   - `classification` → `GPT2ForSequenceClassification.from_pretrained(name, num_labels=N)`
2. Set `model.config.pad_token_id = tokenizer.pad_token_id`
3. Nếu `peft.enabled`:
   - Tạo `LoraConfig(task_type=..., r=..., target_modules=["c_attn"])`
   - Wrap model: `get_peft_model(model, lora_config)`
   - Log trainable parameters count
4. Return model

**GPT-2 specific:**
- GPT-2 dùng `Conv1D` thay vì `nn.Linear` → LoRA target phải là `"c_attn"`, `"c_proj"`, `"c_fc"` (không phải `"q_proj"`, `"v_proj"` như LLaMA)
- Không có native pad token → phải set `pad_token = eos_token`

---

### 3.4 `src/hp_search.py` — Optuna Hyperparameter Search

**Mục đích:** Tìm hyperparameters tối ưu dùng Optuna, tích hợp qua `Trainer.hyperparameter_search()`.

**Public API:**

```python
def run_hp_search(
    config: PipelineConfig,
    trainer: Trainer,
) -> dict:
    """
    Chạy Optuna hyperparameter search.

    Returns:
        best_params: dict với hyperparameters tốt nhất
    """
```

**Search configuration:**

- **Sampler**: `TPESampler(seed=42)` — Tree-structured Parzen Estimator, thông minh hơn random search
- **Pruner**: `MedianPruner(n_startup_trials=3, n_warmup_steps=1)` — dừng sớm trials kém hơn median
- **hp_space function**: Build từ config YAML `search_space` section
  - `learning_rate` → `trial.suggest_float(log=True)` — log scale vì LR vary nhiều bậc
  - `batch_size` → `trial.suggest_categorical()` — phải là giá trị cố định
  - `weight_decay` → `trial.suggest_float()` — uniform scale
  - `warmup_ratio` → `trial.suggest_float()`
  - `num_epochs` → `trial.suggest_int()`
  - Nếu LoRA enabled: `lora_r` → `trial.suggest_categorical([8, 16, 32])`

**Post-search:**
- Log best trial results đẹp mắt
- Apply best params vào `TrainingArguments` cho final training run
- Optuna study được save để có thể resume

---

### 3.5 `src/trainer.py` — Training Orchestration

**Mục đích:** Orchestrate toàn bộ quá trình training, wrap HuggingFace Trainer.

**Public API:**

```python
def run_training(
    config: PipelineConfig,
    datasets: DatasetDict,
    data_collator: DataCollator,
    tokenizer: PreTrainedTokenizer,
    model_init: Callable,
    best_hp: dict = None,          # Từ HP search (optional)
) -> tuple[Trainer, dict]:
    """
    Chạy training pipeline.

    Returns:
        trainer: Trained Trainer instance
        metrics: Final training metrics
    """
```

**TrainingArguments construction:**
- Map `PipelineConfig` → `TrainingArguments` fields
- Mixed precision auto-detect:
  ```python
  if config.training.mixed_precision == "auto":
      bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
      fp16 = not bf16 and torch.cuda.is_available()
  ```
- `eval_strategy` = `save_strategy` (bắt buộc khi `load_best_model_at_end=True`)
- `report_to=["tensorboard"]`
- `logging_dir` = `{tensorboard_dir}/{experiment_name}`

**Callbacks attached:**
1. `EarlyStoppingCallback(patience, threshold)` — nếu `early_stopping.enabled`
2. `GenerationSampleCallback` — generate text mẫu cuối mỗi epoch (chỉ cho causal_lm/completion)
3. `TensorBoardMetricsCallback` — log thêm perplexity vào TensorBoard
4. `RichLoggingCallback` — beautiful console output

**Quan trọng:**
- Khi có `best_hp` từ Optuna → apply vào `TrainingArguments` trước khi tạo Trainer
- Dùng `model_init` (không phải model instance) → Trainer tạo model mới

---

### 3.6 `src/evaluate.py` — Evaluation

**Mục đích:** Đánh giá model trên test set, tạo report cuối cùng.

**Public API:**

```python
def run_evaluation(
    trainer: Trainer,
    test_dataset: Dataset,
    config: PipelineConfig,
) -> dict:
    """
    Evaluate model trên test set.

    Returns:
        metrics: Dict với tất cả metrics theo task_type
    """

def generate_evaluation_report(
    metrics: dict,
    config: PipelineConfig,
) -> str:
    """
    Tạo evaluation report dạng formatted string để in ra console.
    """
```

**Metrics theo task:**

| Task | Metrics | Cách tính |
|---|---|---|
| `causal_lm` | eval_loss, perplexity, sample generations | `ppl = exp(eval_loss)` |
| `classification` | accuracy, f1, precision, recall, confusion matrix | `evaluate` library |
| `completion` | eval_loss, perplexity, ROUGE-1/2/L, BLEU | Generate → compare với reference |

**Perplexity computation:**
- Tính từ `eval_loss` (cross-entropy): `perplexity = math.exp(eval_loss)`
- KHÔNG tính trong `compute_metrics` vì logits quá lớn (batch × seq × 50257) → OOM risk

---

### 3.7 `src/inference.py` — Save/Load/Generate

**Mục đích:** Lưu model + metadata, load lại cho inference, generate text.

**Public API:**

```python
def save_model(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizer,
    config: PipelineConfig,
    metrics: dict,
) -> str:
    """
    Save tất cả artifacts vào output.model_dir.
    Returns: path đến thư mục đã save.
    """

def load_model(model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model từ local path. Auto-detect full vs LoRA.
    Nếu LoRA: load base + adapter, merge_and_unload().
    """

def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    **generation_kwargs,
) -> list[str]:
    """Generate text từ list prompts."""
```

**Save artifacts:**

```
final_model/
├── config.json              # HuggingFace model config
├── model.safetensors        # Model weights (full) hoặc adapter_model.safetensors (LoRA)
├── tokenizer.json           # Tokenizer config + vocabulary
├── special_tokens_map.json  # Special tokens mapping
├── generation_config.json   # Default generation params (temperature, top_p, max_tokens)
├── training_config.yaml     # Copy YAML config đã dùng → reproducibility hoàn toàn
└── training_metrics.json    # Final metrics (loss, perplexity, accuracy, training time)
```

---

### 3.8 `src/callbacks.py` — Custom Callbacks

**3 callbacks:**

#### `RichLoggingCallback(TrainerCallback)`
- `on_train_begin`: In banner pipeline (model info, device, precision)
- `on_log`: Format metrics đẹp mắt mỗi `logging_steps`
- `on_epoch_end`: In epoch summary với train/eval metrics
- `on_train_end`: In tổng kết (total time, best metric, model path)

#### `GenerationSampleCallback(TrainerCallback)`
- `on_epoch_end`: Generate 1 sample text từ prompt mẫu, in ra console
- Chỉ active khi `task_type` in `["causal_lm", "completion"]`
- Dùng `model.generate()` với `temperature=0.7, max_new_tokens=50`

#### `TensorBoardMetricsCallback(TrainerCallback)`
- `on_evaluate`: Tính perplexity từ eval_loss, ghi vào TensorBoard
- `on_train_end`: Close SummaryWriter

---

### 3.9 `src/utils.py` — Utilities

**Functions:**

```python
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup logger với format đẹp: timestamp, level (colored), module name, message."""

def detect_device() -> dict:
    """
    Auto-detect hardware. Returns:
    {"device": "cuda", "n_gpus": 2, "gpu_names": ["RTX 4090", "RTX 4090"],
     "total_vram_gb": 48.0, "bf16_supported": True}
    """

def set_seed(seed: int):
    """Set seed cho torch, numpy, random, transformers."""

class StepTimer:
    """Context manager đo thời gian mỗi pipeline step."""
    def __enter__(self): ...
    def __exit__(self): ...  # Log elapsed time

def format_banner(config: PipelineConfig, device_info: dict) -> str:
    """Tạo banner đẹp hiển thị đầu pipeline run."""

def format_metrics_table(metrics: dict) -> str:
    """Format metrics thành table dạng aligned columns."""
```

---

### 3.10 `main.py` — CLI Entry Point

**Usage:**

```bash
# Full pipeline (tất cả steps tuần tự)
python main.py --config configs/default.yaml

# Chạy từng step
python main.py --config configs/default.yaml --step data
python main.py --config configs/default.yaml --step hp_search
python main.py --config configs/default.yaml --step train
python main.py --config configs/default.yaml --step evaluate
python main.py --config configs/default.yaml --step generate --prompts "Ngày xửa ngày xưa"

# Override config
python main.py --config configs/default.yaml --training.learning_rate 1e-4 --peft.enabled true

# So sánh experiments
python compare.py ./runs/exp1 ./runs/exp2
```

**Step execution order (full pipeline):**

1. Load & validate config
2. Setup logger, seed, detect device
3. Print banner
4. `prepare_data()` → datasets, collator, tokenizer
5. `create_model_init()` → model_init function
6. (Optional) `run_hp_search()` → best_hp
7. `run_training()` → trainer, train_metrics
8. `run_evaluation()` → test_metrics
9. `save_model()` → saved path
10. Print final summary

---

### 3.11 `compare.py` — Experiment Comparison

**Mục đích:** So sánh kết quả giữa nhiều experiments.

**Input:** Paths tới TensorBoard run directories hoặc `training_metrics.json` files.

**Output:** Bảng so sánh formatted:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    📊 Experiment Comparison                         │
├────────────────┬──────────────┬──────────────┬──────────────────────┤
│ Metric         │ exp1         │ exp2         │ exp3 (LoRA)          │
├────────────────┼──────────────┼──────────────┼──────────────────────┤
│ Model          │ gpt2         │ gpt2-medium  │ gpt2 + LoRA r=16    │
│ Train loss     │ 2.22         │ 1.89         │ 2.45                 │
│ Eval loss      │ 2.39         │ 2.01         │ 2.52                 │
│ Perplexity     │ 10.9         │ 7.5          │ 12.4                 │
│ Training time  │ 1h 12m       │ 3h 45m       │ 28m                  │
│ Model size     │ 487 MB       │ 1.4 GB       │ 3.2 MB (adapter)     │
│ Best epoch     │ 4/5          │ 3/5          │ 5/5                  │
└────────────────┴──────────────┴──────────────┴──────────────────────┘
```

---

## 4. Logging & Console Output

### 4.1 Log Format

```
[2026-08-25 10:30:15] ℹ️  INFO  | data     | Loading dataset from HuggingFace Hub: wikitext
[2026-08-25 10:30:18] ℹ️  INFO  | data     | Train: 36,718 samples | Val: 3,760 | Test: 4,358
[2026-08-25 10:30:20] ℹ️  INFO  | model    | Created GPT2LMHeadModel + LoRA (r=16, target=c_attn)
[2026-08-25 10:30:20] ℹ️  INFO  | model    | Trainable: 294,912 / 124,734,720 (0.24%)
[2026-08-25 10:30:25] ⚠️  WARN  | trainer  | Early stopping triggered at epoch 4 (patience=3)
[2026-08-25 10:30:30] ❌ ERROR | data     | File not found: ./data/missing.csv
```

### 4.2 Pipeline Banner

```
╔══════════════════════════════════════════════════════════════╗
║                  🚀 GPT-2 Fine-Tuning Pipeline              ║
╠══════════════════════════════════════════════════════════════╣
║  Model:     gpt2                                            ║
║  Task:      causal_lm                                       ║
║  LoRA:      enabled (r=16, target=c_attn)                   ║
║  Device:    NVIDIA RTX 4090 (24GB) × 1                      ║
║  Precision: bf16                                             ║
╚══════════════════════════════════════════════════════════════╝
```

### 4.3 Step Progress

```
📦 [Step 1/5] Data Preparation
   ├─ Source:    HuggingFace Hub → wikitext/wikitext-2-raw-v1
   ├─ Train:     36,718 samples
   ├─ Valid:      3,760 samples
   ├─ Test:      4,358 samples
   ├─ Max length: 512 tokens
   └─ ✅ Done in 12.3s

🏋️ [Step 3/5] Training
   ├─ Epoch 1/5: train_loss=3.12 | eval_loss=2.95 | ppl=19.1
   ├─ Epoch 2/5: train_loss=2.78 | eval_loss=2.61 | ppl=13.6
   ├─ 📝 Sample: "Ngày xửa ngày xưa" → "có một chàng trai..."
   └─ ✅ Done in 1h 12m
```

---

## 5. Config YAML — Comment Tiếng Việt Chi Tiết

Mỗi config file YAML sẽ chứa:
- Comment tiếng Việt cho **từng dòng** giải thích ý nghĩa
- Với mỗi config có nhiều lựa chọn: **liệt kê tất cả options**, giải thích ý nghĩa, và **gợi ý khi nào dùng option nào**
- Ví dụ chi tiết đã được approve ở Design Section 1

---

## 6. Dependencies

```
# Core
transformers>=4.40.0          # HuggingFace Transformers (Trainer, models, tokenizers)
datasets>=2.18.0              # HuggingFace Datasets (load, process)
torch>=2.0.0                  # PyTorch backend
peft>=0.10.0                  # LoRA / parameter-efficient fine-tuning
optuna>=3.6.0                 # Hyperparameter optimization
accelerate>=0.28.0            # Multi-GPU, mixed precision support

# Evaluation
evaluate>=0.4.0               # HuggingFace Evaluate (accuracy, f1, rouge, bleu)
scikit-learn>=1.4.0           # Confusion matrix, classification report

# Logging & Visualization
tensorboard>=2.16.0           # Training visualization
rich>=13.7.0                  # Beautiful console output

# Data
numpy>=1.26.0                 # Numerical operations
pandas>=2.2.0                 # Data manipulation (local files)

# Optional - Data Augmentation
nlpaug>=1.1.11                # Text augmentation techniques (optional)

# Utilities
pyyaml>=6.0                   # YAML config parsing
safetensors>=0.4.0            # Safe model serialization
trl>=0.8.0                    # DataCollatorForCompletionOnlyLM (optional, cho completion task)
```

---

## 7. Error Handling Strategy

- **Config validation**: Fail fast nếu YAML invalid, field thiếu, type sai → message rõ ràng
- **Data loading**: Try/except với message cụ thể (file not found, column missing, download fail)
- **GPU/VRAM**: Catch CUDA OOM → suggest giảm batch_size hoặc bật LoRA
- **Training**: Checkpoint recovery — nếu training bị gián đoạn, có thể resume từ checkpoint
- **Tất cả errors**: Log đầy đủ traceback + suggestion fix

---

## 8. Testing Strategy

- **Unit tests** cho mỗi module: config parsing, data loading, model creation
- **Integration test**: Full pipeline chạy end-to-end với tiny dataset (10 samples, 1 epoch)
- **Config test**: Validate tất cả sample configs load thành công
- Test files đặt trong `finetune-gpt2/tests/`

---

## 9. Future Extensions (Ngoài scope hiện tại)

- Weights & Biases integration
- QLoRA (quantized LoRA)
- Streaming datasets cho data rất lớn
- Multi-node distributed training
- Model deployment (ONNX export, TorchServe)
- Prompt engineering utilities
