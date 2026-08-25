# 🚀 Hướng Dẫn Toàn Diện: Fine-Tuning GPT-2 Pipeline

## 1. Giới thiệu & Tính năng (Overview & Features)

Chào mừng bạn đến với tài liệu hướng dẫn toàn diện của **GPT-2 Fine-Tuning Pipeline**. Đây là một hệ thống chuyên nghiệp, được thiết kế theo hướng module hóa, sẵn sàng triển khai trong môi trường sản xuất (production). Hệ thống giúp tối ưu hóa quá trình huấn luyện các mô hình ngôn ngữ dựa trên kiến trúc Transformer của OpenAI (đặc biệt là họ GPT-2).

### 🎯 Các tính năng nổi bật:
- **Tái sử dụng cao (Reusable):** Thông qua hệ thống cấu hình YAML linh hoạt, bạn có thể áp dụng pipeline cho hàng chục tập dữ liệu và bài toán khác nhau mà không cần sửa code.
- **Đa tác vụ (Multi-task):** Hỗ trợ toàn diện các tác vụ NLP phổ biến:
  - Sinh văn bản tự do (Causal Language Modeling).
  - Phân loại văn bản (Text Classification).
  - Hoàn thành câu lệnh/văn bản (Instruction Completion).
- **Tự động nhận diện phần cứng (Multi-hardware):** Hệ thống tự động tối ưu hóa tài nguyên (CPU, Single GPU, Multi-GPU) thông qua HuggingFace Accelerate.
- **Theo dõi chuyên nghiệp (Professional logging):** Giao diện console trực quan với thư viện `Rich` và tích hợp sâu với `TensorBoard` để theo dõi loss, perplexity và các metrics khác theo thời gian thực.
- **Thực hành chuẩn (Best practices):** Tích hợp sẵn các phương pháp tiên tiến nhất:
  - LoRA/PEFT (Parameter-Efficient Fine-Tuning) giúp huấn luyện mô hình lớn trên phần cứng hạn chế.
  - Tìm kiếm siêu tham số tự động (Hyperparameter Search) bằng `Optuna`.
  - Dừng sớm (Early Stopping) ngăn chặn overfitting.
  - Tính toán độ chính xác hỗn hợp (Mixed Precision - FP16, BF16).

---

## 2. Cài đặt (Installation)

Quá trình cài đặt đơn giản, yêu cầu Python phiên bản 3.10 trở lên. Khuyến nghị sử dụng môi trường ảo (virtual environment) để tránh xung đột thư viện.

### Các bước cài đặt:

```bash
# 1. Clone repository và di chuyển vào thư mục dự án
cd finetune-gpt2

# 2. Tạo môi trường ảo
python3 -m venv .venv

# 3. Kích hoạt môi trường ảo
# Đối với Linux/MacOS:
source .venv/bin/activate
# Đối với Windows:
# .venv\Scripts\activate

# 4. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 📦 Các phụ thuộc (Dependencies) chính:
- **HuggingFace Stack:** `transformers`, `datasets`, `peft`, `evaluate` (để load model, data, tối ưu hóa và đánh giá).
- **Deep Learning:** `torch` (PyTorch backend cốt lõi), `accelerate` (hỗ trợ phân tán).
- **Tối ưu hóa:** `optuna` (tìm hyperparameter).
- **Giao diện & Logging:** `tensorboard` (biểu đồ), `rich` (in màu console log).

---

## 3. Quick Start (Chạy thử trong 5 phút)

Cách nhanh nhất để làm quen với pipeline là sử dụng cấu hình mặc định được cung cấp sẵn trong `configs/default.yaml`. Nó sẽ tải một dataset cơ bản (Wikitext) và fine-tune mô hình GPT-2 nhỏ nhất.

### Cấu hình mẫu (`configs/default.yaml`)
Một file config YAML có cấu trúc phân tầng rõ ràng:

```yaml
# Cấu hình Mô hình
model:
  name: "gpt2"                    # Tên mô hình trên HuggingFace
  task_type: "causal_lm"          # Loại tác vụ (Causal Language Modeling)

# Cấu hình Dữ liệu
data:
  source: "huggingface"           # Nguồn (HuggingFace Hub hoặc local file)
  dataset_name: "wikitext"        # Tên dataset
  dataset_config: "wikitext-2-raw-v1" # Cấu hình của dataset
  text_column: "text"             # Cột chứa văn bản
  max_length: 512                 # Độ dài chuỗi tối đa
  split_ratios: [0.8, 0.1, 0.1]   # Tỷ lệ Train/Validation/Test

# Cấu hình Huấn luyện
training:
  num_epochs: 3                   # Số epoch
  batch_size: 4                   # Kích thước batch cho mỗi GPU
  learning_rate: 5e-5             # Tốc độ học (Learning Rate)
  mixed_precision: "auto"         # Tự động dùng BF16/FP16 nếu có thể
```

### Lệnh chạy Pipeline toàn diện
Để thực thi quy trình từ A-Z (Data -> HP Search (nếu bật) -> Train -> Evaluate -> Save):

```bash
python main.py --config configs/default.yaml
```

### Sinh văn bản sau khi huấn luyện (Inference)
Sau khi có mô hình, bạn có thể sinh văn bản ngay từ CLI:

```bash
python main.py \
  --config configs/default.yaml \
  --step generate \
  --prompts "Trong một tương lai không xa, trí tuệ nhân tạo sẽ"
```

---

## 4. 8 Production Use Cases (Các Trường Hợp Sử Dụng Thực Tế)

Dưới đây là 8 kịch bản ứng dụng toàn diện, sẵn sàng cho môi trường doanh nghiệp (Production). Mỗi use case mô tả bài toán cụ thể, cung cấp cấu hình chuẩn xác và các mẹo tối ưu độc quyền.

### 🤖 UC1: Customer Service Chatbot
- **Mô tả bài toán:** Xây dựng chatbot chăm sóc khách hàng tự động, giúp giải đáp thắc mắc, giảm tải cho nhân viên CSKH trong các ngành E-commerce, SaaS, và Telecom.
- **Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (Tập dữ liệu Bitext chất lượng cao).
- **Cấu hình YAML mẫu (`configs/uc1_chatbot.yaml`):**

```yaml
model:
  name: "microsoft/DialoGPT-medium"
  task_type: "causal_lm"
peft:
  enabled: true
  r: 16
  target_modules: ["c_attn", "c_proj"]
data:
  dataset_name: "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
  text_column: "text"
training:
  num_epochs: 5
  batch_size: 4
  learning_rate: 3e-5
  warmup_ratio: 0.1
```

- **Lệnh chạy:** `python main.py --config configs/uc1_chatbot.yaml`
- **Metrics kỳ vọng:** Perplexity (PPL) từ 12-18, BLEU-4 22-28%, ROUGE-L 35-42%.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* Mô hình lặp vô hạn câu trả lời (Repetition loop).
    - *Fix:* Thêm các tham số `repetition_penalty=1.2` và `no_repeat_ngram_size=3` vào hàm generate.
  - *Lỗi:* Bot tự sinh cả phần hỏi của user.
    - *Fix:* Bắt buộc thêm special tokens (ví dụ: `<|user|>`, `<|bot|>`) vào từ điển của tokenizer.
  - *Lỗi:* Cảnh báo thiếu pad_token.
    - *Fix:* Cấu hình ngầm định `tokenizer.pad_token = tokenizer.eos_token` trong mã nguồn.
- **Tips & Best Practices:** Sử dụng `DialoGPT-medium` làm base model tốt hơn `gpt2-medium` vì nó đã được pre-train trên hàng triệu hội thoại từ Reddit.

---

### 📝 UC2: Marketing Content Generation (SEO/Ad Copy)
- **Mô tả bài toán:** Tự động sinh nội dung quảng cáo (Ad Copy), mô tả sản phẩm (Product Description), và bài viết chuẩn SEO cho ngành Digital Marketing.
- **Dataset:** `McAuley-Lab/Amazon-Reviews-2023` hoặc `hezarfen/marketing-product-generator`.
- **Cấu hình YAML mẫu (`configs/uc2_marketing.yaml`):**

```yaml
model:
  name: "gpt2-medium"
  task_type: "completion"
peft:
  enabled: true
  r: 32
data:
  dataset_name: "hezarfen/marketing-product-generator"
  text_column: "text"
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 2e-4
  weight_decay: 0.05
```

- **Lệnh chạy:** `python main.py --config configs/uc2_marketing.yaml`
- **Metrics kỳ vọng:** Perplexity 10.5-15, ROUGE-1 40-46%.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* Model chỉ sao chép lại thông số kỹ thuật (bullet points) thay vì viết thành đoạn văn mượt mà.
    - *Fix:* Sử dụng `task_type="completion"`, hệ thống tự động thiết lập nhãn (labels) bằng -100 cho phần prompt (chỉ tính loss trên câu trả lời).
  - *Lỗi:* Hết bộ nhớ VRAM (OOM).
    - *Fix:* Áp dụng LoRA, giảm batch_size và bật gradient checkpointing.
- **Tips & Best Practices:** Cấu trúc prompt mẫu cần nhất quán trong dữ liệu (ví dụ: `Tên sản phẩm: ... Tính năng: ... => Mô tả:`).

---

### 💻 UC3: Code Generation (Python Autocomplete)
- **Mô tả bài toán:** Phát triển plugin cho IDE (như VSCode extension) hỗ trợ lập trình viên sinh mã nguồn (autocomplete) dự đoán dòng code tiếp theo.
- **Dataset:** `code_search_net`, `openai_humaneval`, hoặc `bigcode/the-stack-smol`.
- **Cấu hình YAML mẫu (`configs/uc3_code.yaml`):**

```yaml
model:
  name: "microsoft/CodeGPT-small-py"
  task_type: "causal_lm"
data:
  dataset_name: "code_search_net"
  dataset_config: "python"
  text_column: "func_code_string"
  max_length: 512
training:
  num_epochs: 4
  batch_size: 8
  learning_rate: 5e-5
```

- **Lệnh chạy:** `python main.py --config configs/uc3_code.yaml`
- **Metrics kỳ vọng:** Perplexity 4.5-8, Pass@1 18-28%, Syntax validity (tỷ lệ code hợp lệ cú pháp) > 92%.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* Code sinh ra bị thụt lề sai (indentation errors).
    - *Fix:* Tuyệt đối sử dụng tokenizer chuyên biệt cho code (của CodeGPT), bảo toàn các ký tự khoảng trắng và tab.
  - *Lỗi:* Mô hình không dừng lại sau khi sinh xong hàm.
    - *Fix:* Cần chỉ định custom stopping criteria, dừng lại khi gặp ký tự sang dòng không có thụt lề hoặc thẻ `<|endoftext|>`.
- **Tips & Best Practices:** Luôn dùng `CodeGPT` (được pretrain trên mã nguồn) thay vì GPT-2 thường.

---

### 📊 UC4: Sentiment Analysis / Text Classification
- **Mô tả bài toán:** Phân loại sắc thái cảm xúc văn bản (Tích cực/Tiêu cực/Trung tính) trong các ngành FinTech, Brand Monitoring, và mạng xã hội.
- **Dataset:** `financial_phrasebank`, `imdb`, hoặc `tweet_eval`.
- **Cấu hình YAML mẫu (`configs/uc4_classification.yaml`):**

```yaml
model:
  name: "gpt2"
  task_type: "classification"
  num_labels: 3
data:
  dataset_name: "financial_phrasebank"
  dataset_config: "sentences_allagree"
  text_column: "sentence"
  label_column: "label"
  max_length: 128
training:
  num_epochs: 5
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
```

- **Lệnh chạy:** `python main.py --config configs/uc4_classification.yaml`
- **Metrics kỳ vọng:** Accuracy 91-95%, F1-score 0.88-0.93.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* Độ chính xác chỉ loanh quanh 33% (tương đương đoán mò).
    - *Fix:* **Bắt buộc** cấu hình `tokenizer.padding_side = "left"`. Vì hệ thống dùng token cuối cùng cho bộ phân loại (classifier), nếu đệm bên phải (right-padding), token cuối cùng bị đưa vào mô hình sẽ là token đệm vô nghĩa `[PAD]`.
  - *Lỗi:* Exception kiểu dữ liệu labels.
    - *Fix:* Mã nguồn cần ép kiểu `labels = labels.to(torch.long)`.
  - *Lỗi:* Model dự đoán nghiêng về một class (Imbalanced classes).
    - *Fix:* Sử dụng `CrossEntropyLoss(weight=class_weights)`.
- **Tips & Best Practices:** Sequence length cho classification không cần quá dài, 128 hoặc 256 là đủ, giúp tiết kiệm bộ nhớ.

---

### 🏥 UC5: Domain-Specific (Legal/Medical/Financial)
- **Mô tả bài toán:** Huấn luyện mô hình am hiểu kiến thức chuyên ngành sâu (Y tế, Pháp lý, Tài chính). Yêu cầu độ chính xác rất cao và văn phong chuyên nghiệp.
- **Dataset:** `pile-of-law`, `gamino/wiki_medical_terms`, hoặc `lex_glue`.
- **Cấu hình YAML mẫu (`configs/uc5_domain.yaml`):**

```yaml
model:
  name: "gpt2-large"
  task_type: "causal_lm"
peft:
  enabled: true
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
data:
  dataset_name: "pile-of-law"
  text_column: "text"
training:
  num_epochs: 2
  batch_size: 4
  learning_rate: 2e-5  # Phải giữ LR thấp
```

- **Lệnh chạy:** `python main.py --config configs/uc5_domain.yaml`
- **Metrics kỳ vọng:** Perplexity 8.5-14, ROUGE-L 42-48%.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* "Hallucination" sinh ra thông tin sai (ví dụ sai liều lượng thuốc).
    - *Fix:* Không dùng mô hình độc lập, phải kết hợp với cấu trúc RAG (Retrieval-Augmented Generation) để đối chiếu tài liệu thật.
  - *Lỗi:* "Catastrophic forgetting" (Quên ngữ pháp cơ bản).
    - *Fix:* Bắt buộc dùng LoRA thay vì full fine-tuning để giữ nguyên base weights.
  - *Lỗi:* Các từ viết tắt bị chia cắt (ví dụ: HIPAA thành H-I-P-A-A).
    - *Fix:* Khai báo bổ sung qua `tokenizer.add_tokens(["HIPAA", "SOAP", ...])`.
- **Tips & Best Practices:** Cực kỳ cẩn trọng với learning rate. Cần giữ ở mức `1e-5` đến `2e-5`.

---

### 🇻🇳 UC6: Vietnamese Language Tasks
- **Mô tả bài toán:** Xây dựng mô hình ngôn ngữ hoặc trợ lý ảo hoạt động tối ưu riêng cho tiếng Việt. Dùng trong truyền thông, báo chí, xuất bản.
- **Dataset:** `bkai-foundation-models/BKAINewsCorpus` hoặc `vietgpt/wikipedia_vi`.
- **Cấu hình YAML mẫu (`configs/uc6_vietnamese.yaml`):**

```yaml
model:
  name: "NlpHUST/gpt2-vietnamese"
  task_type: "causal_lm"
data:
  dataset_name: "bkai-foundation-models/BKAINewsCorpus"
  text_column: "content"
training:
  num_epochs: 4
  batch_size: 8
  learning_rate: 5e-5
```

- **Lệnh chạy:** `python main.py --config configs/uc6_vietnamese.yaml`
- **Metrics kỳ vọng:** Perplexity 15-22, BLEU-2 35-42%.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* Tiếng Việt bị vỡ vụn thành các byte sequences rời rạc, không đọc được.
    - *Fix:* KHÔNG ĐƯỢC dùng tiếng Anh (gpt2) làm base. Bắt buộc dùng mô hình có tokenizer đã train trên tiếng Việt như `NlpHUST/gpt2-vietnamese`.
  - *Lỗi:* Sinh thơ lục bát sai vần điệu (Bằng/Trắc).
    - *Fix:* Cần huấn luyện bổ sung (fine-tune thêm 1 bước) riêng trên một tập dữ liệu thơ đã được sàng lọc.
- **Tips & Best Practices:** Độ dài ngữ cảnh của tiếng Việt thường dài hơn do đặc thù nhiều âm tiết ghép, nên để `max_length: 512` hoặc lớn hơn.

---

### 📖 UC7: Creative Writing & Storytelling
- **Mô tả bài toán:** Phát triển ứng dụng kể chuyện, viết tiểu thuyết, game nhập vai Text RPG.
- **Dataset:** `roneneldan/TinyStories`, `writingprompts`, hoặc `roc_stories`.
- **Cấu hình YAML mẫu (`configs/uc7_creative.yaml`):**

```yaml
model:
  name: "gpt2-medium"
  task_type: "causal_lm"
data:
  dataset_name: "roneneldan/TinyStories"
  text_column: "text"
training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 5e-5
```

- **Lệnh chạy:** `python main.py --config configs/uc7_creative.yaml`
- **Metrics kỳ vọng:** Perplexity 16-24, Tỷ lệ từ vựng độc nhất (Distinct-2) > 0.75.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* "Perplexity Trap" - Hàm mất mát (loss) cực thấp, perplexity tuyệt vời, nhưng văn bản sinh ra cực kỳ nhàm chán và lặp lại.
    - *Fix:* Điều chỉnh các thông số khi sinh: `temperature=0.8` hoặc `0.9`, sử dụng `top_p=0.9` (Nucleus Sampling) hoặc `top_k=50`.
  - *Lỗi:* Cốt truyện bị đứt gãy mạch logic sau 200 từ.
    - *Fix:* Triển khai chiến lược hierarchical conditioning (Tóm tắt cốt truyện -> Lên dàn ý -> Viết chi tiết từng chương).
- **Tips & Best Practices:** Trong tác vụ sáng tạo, một model "chưa học quá thuộc" (loss nhỉnh hơn một chút) lại thường sinh văn hay hơn một model bị overfit.

---

### 📑 UC8: Data Extraction & Structured Output (JSON)
- **Mô tả bài toán:** Trích xuất thông tin thực thể, thuộc tính từ văn bản thô và chuyển thành định dạng cấu trúc (JSON, Bảng) cho Business Intelligence và Data Engineering.
- **Dataset:** `web_nlg`, `e2e_nlg`, hoặc `wiki_bio`.
- **Cấu hình YAML mẫu (`configs/uc8_json.yaml`):**

```yaml
model:
  name: "gpt2-medium"
  task_type: "completion"
data:
  dataset_name: "web_nlg"
  text_column: "text"
training:
  num_epochs: 5
  batch_size: 8
  learning_rate: 5e-5
```

- **Lệnh chạy:** `python main.py --config configs/uc8_json.yaml`
- **Metrics kỳ vọng:** BLEU-4 48-55, JSON validity (tỷ lệ parse JSON hợp lệ) > 96%.
- **Lỗi thường gặp & cách fix:**
  - *Lỗi:* Cú pháp JSON bị sai (thiếu ngoặc nhọn, ngoặc kép, dấu phẩy).
    - *Fix:* Phải cấu hình sinh bằng Greedy decoding (`temperature=0.0`). Tối ưu nhất là dùng thư viện Constrained Decoding như `outlines` hoặc `jsonformer` để cưỡng ép tuân thủ schema JSON.
  - *Lỗi:* Mô hình tự tưởng tượng (hallucinate) ra các trường thông tin không có trong yêu cầu.
    - *Fix:* Dùng data augmentation để xáo trộn ngẫu nhiên thứ tự các trường (Key-Value) trong tập huấn luyện, giúp mô hình học cách chú ý vào logic thay vì học vẹt thứ tự.
- **Tips & Best Practices:** Định dạng dữ liệu huấn luyện cần sự đồng nhất tuyệt đối về khoảng trắng và dấu câu trong JSON.

---

## 5. Troubleshooting Guide (Tổng Hợp Lỗi Phổ Biến)

Hệ thống ghi nhận 7 lỗi phổ biến nhất và giải pháp tối ưu tương ứng:

| Mã Lỗi / Triệu Chứng (Symptom) | Nguyên nhân gốc rễ (Root Cause) | Cách Khắc Phục Nhanh (Quick Fix) |
|---|---|---|
| **`pad_token` error** | Kiến trúc mặc định của GPT-2 không định nghĩa token dùng để đệm (padding). | Cập nhật tokenizer: `tokenizer.pad_token = tokenizer.eos_token` |
| **Classification accuracy ~ random** (Độ chính xác rất thấp) | Mặc định GPT-2 đệm bên phải (right padding), đẩy các token nhiễu về cuối chuỗi phân loại. | Đặt cờ: `tokenizer.padding_side = "left"` để token cuối luôn chứa thông tin. |
| **CUDA OOM (Out of Memory)** | VRAM GPU không đủ để chứa toàn bộ context, weights và gradients. | 1) Giảm `batch_size`. 2) Bật `peft.enabled: true` (LoRA). 3) Cấu hình `gradient_accumulation_steps`. |
| **Repetitive output** (Đầu ra lặp lại liên tục) | Quá trình decoding ưu tiên cao cho các cụm từ vừa xuất hiện. | Truyền tham số lúc sinh: `repetition_penalty=1.2` và `no_repeat_ngram_size=3`. |
| **Catastrophic Forgetting** (Quên kỹ năng ngôn ngữ chung) | Full fine-tune làm thay đổi quá mạnh các layer sâu của mô hình trên dữ liệu hẹp. | 1) Luôn dùng LoRA cho các tác vụ domain hẹp. 2) Trộn thêm 5-10% dữ liệu gốc (general corpus). |
| **Loss NaN/Inf** (Loss không hội tụ) | Hiện tượng tràn số (Overflow) khi dùng tính toán điểm dấu phẩy động 16-bit (FP16). | Cấu hình trong file yaml: đổi `mixed_precision: "fp16"` thành `mixed_precision: "bf16"` (nếu GPU hỗ trợ). |
| **`eval_strategy` ≠ `save_strategy`** | Lỗi config của HuggingFace Trainer khi gọi `load_best_model_at_end=True`. | Đảm bảo 2 giá trị này hoàn toàn trùng khớp (ví dụ cùng là `"epoch"`). |

---

## 6. Bảng Tham Khảo Hyperparameters Theo Mô Hình

Bảng phân bổ tài nguyên và siêu tham số khuyến nghị (Best practices).

| Thông số (Param) | gpt2 (Base, 124M) | gpt2-medium (355M) | gpt2-large (774M) | gpt2-xl (1.5B) |
|---|---|---|---|---|
| **Learning Rate (LR)** | 3e-5 – 5e-5 | 2e-5 – 3e-5 | 1e-5 – 2e-5 | 1e-5 |
| **Batch Size (Max)** | 8 - 16 | 4 - 8 | 2 - 4 | 1 - 2 (Gradient Acc) |
| **VRAM Yêu Cầu tối thiểu** | 4 - 8 GB | 8 - 16 GB | 16 - 24 GB | 24 - 40 GB |
| **LoRA Rank (`r`)** | 8 - 16 | 16 - 32 | 16 - 32 | 32 - 64 |
| **LoRA Target Modules** | `["c_attn"]` | `["c_attn"]` | `["c_attn", "c_proj"]` | `["c_attn", "c_proj"]` |
| **Max Context (Tokens)** | 512 - 1024 | 512 - 1024 | 512 - 1024 | 1024 |
| **Trọng lượng suy giảm (Weight Decay)**| 0.01 | 0.01 | 0.05 | 0.05 |

*Lưu ý: Đối với GPT-2, tham số mục tiêu của LoRA bắt buộc là `c_attn` (sử dụng Conv1D), không phải `q_proj`/`v_proj` như trong kiến trúc LLaMA.*

---

## 7. CLI Reference (Tài Liệu Dòng Lệnh Toàn Diện)

Pipeline cung cấp một CLI mạnh mẽ, giúp bạn can thiệp sâu vào từng khâu xử lý.

### 🚀 Chạy Workflow Tiêu Chuẩn
Thực thi toàn bộ kịch bản được định nghĩa trong file cấu hình:
```bash
python main.py --config configs/default.yaml
```

### 🧩 Chạy Từng Phân Đoạn (Step-by-step Execution)
Hỗ trợ chia nhỏ quá trình để dễ dàng debug và kiểm thử:
```bash
# 1. Tải và xử lý dữ liệu, kiểm tra tokenizer
python main.py --config configs/default.yaml --step data

# 2. Chạy Optuna để tự động dò tìm siêu tham số tối ưu (Hyperparameter Search)
python main.py --config configs/default.yaml --step hp_search

# 3. Kích hoạt quá trình huấn luyện
python main.py --config configs/default.yaml --step train

# 4. Chạy kiểm thử trên tập Test (Đánh giá)
python main.py --config configs/default.yaml --step evaluate

# 5. Sinh thử nghiệm từ CLI
python main.py --config configs/default.yaml --step generate --prompts "AI trong tương lai"
```

### ⚙️ Ghi đè cấu hình động (Dynamic CLI Override)
Sử dụng cú pháp Dot-Notation (`--section.field`) để ghi đè bất kỳ giá trị nào mà không cần sửa file YAML:
```bash
python main.py \
  --config configs/default.yaml \
  --training.learning_rate 1e-4 \
  --training.batch_size 16 \
  --peft.enabled true \
  --peft.r 32 \
  --data.max_length 256
```

### 📈 So sánh kết quả huấn luyện (Experiment Comparison)
Trực quan hóa bảng so sánh nhiều lần chạy khác nhau:
```bash
python compare.py ./runs/experiment_v1 ./runs/experiment_v2 ./runs/experiment_lora
```
*Lệnh này sẽ trích xuất metrics từ file `training_metrics.json` và vẽ bảng so sánh đẹp mắt lên console.*

---

## 8. FAQ (Câu Hỏi Thường Gặp)

**Q: Khi nào tôi nên áp dụng Full Fine-tuning và khi nào dùng LoRA?**
> A: Bạn nên ưu tiên LoRA khi: tài nguyên phần cứng eo hẹp (chỉ có 1 GPU phổ thông), dữ liệu nhỏ (dưới 10.000 mẫu), hoặc bạn chỉ cần tinh chỉnh domain nhẹ (Domain adaptation). Full fine-tuning chỉ khuyến nghị khi bạn có cụm GPU lớn, có lượng data khổng lồ (hàng trăm MB text), và muốn model thay đổi kiến thức rất sâu.

**Q: Tại sao tôi finetune tác vụ Classification mà accuracy lại rất thấp?**
> A: Vấn đề cốt lõi của GPT-2 là kiến trúc "Decoder-only" với "Causal mask". Mô hình chỉ nhìn về bên trái. Nếu bạn đệm (padding) dữ liệu về bên phải (hành vi mặc định), token cuối cùng được đưa vào mạng MLP (bộ phân loại) sẽ bị rỗng (`[PAD]`). Để khắc phục, bạn **BẮT BUỘC** phải chuyển hướng padding: `tokenizer.padding_side = "left"`.

**Q: Cấu hình `mixed_precision: "auto"` hoạt động ra sao?**
> A: Hệ thống sẽ truy vấn kiến trúc GPU của bạn. Nếu GPU hỗ trợ Ampere architecture trở lên (như RTX 30/40 series, A100), nó tự động dùng `bf16` (bFloat16) - an toàn hơn, không bị tràn số (overflow). Các dòng card cũ hơn (như T4, V100) sẽ dùng `fp16`. CPU sẽ không dùng tính năng này.

**Q: Tôi có thể sử dụng pipeline này để train LLaMA hay Mistral không?**
> A: Pipeline này được thiết kế và tinh chỉnh sâu, chuyên biệt cho kiến trúc của họ HuggingFace GPT-2 (sử dụng Conv1D layer thay vì Linear, hệ thống block name đặc trưng như `c_attn`). Tuy nhiên, phương pháp luận, config logic, và cấu trúc thư mục thì hoàn toàn áp dụng được nếu bạn nâng cấp codebase.

**Q: Làm sao để theo dõi biểu đồ Loss trong khi train?**
> A: Hệ thống ngầm định lưu log TensorBoard vào thư mục `./runs`. Bạn mở terminal mới, chạy: `tensorboard --logdir ./runs` và truy cập `http://localhost:6006` để xem biểu đồ trực quan.

---

## Đóng Góp (Contributing)
Chúng tôi hoan nghênh mọi nỗ lực đóng góp mã nguồn, cải thiện tài liệu hoặc báo cáo lỗi.
- Vui lòng tạo Issue nếu bạn gặp bất kỳ trở ngại nào trong quá trình sử dụng.
- Fork repository và tạo Pull Request kèm theo mô tả chi tiết nếu bạn muốn tối ưu thêm tính năng.

**Trân trọng cảm ơn bạn đã sử dụng GPT-2 Fine-Tuning Pipeline! Chúc bạn huấn luyện thành công! 🚀**

## 9. Cấu Trúc Mã Nguồn (Code Architecture)

Để giúp các nhà phát triển dễ dàng tùy biến, dưới đây là sơ đồ cấu trúc của dự án:

```text
finetune-gpt2/
├── configs/                      # Thư mục chứa YAML config
│   ├── default.yaml              # Config mặc định cho Causal LM
│   ├── classification.yaml       # Config mẫu cho Classification
│   └── lora.yaml                 # Config mẫu cho LoRA Fine-Tuning
├── src/                          # Mã nguồn chính
│   ├── __init__.py               
│   ├── config.py                 # Xử lý Dataclass config & YAML loader
│   ├── data.py                   # Data loading, tokenization, augmentation
│   ├── model.py                  # Model factory (Full/LoRA/Classification)
│   ├── hp_search.py              # Tối ưu siêu tham số bằng Optuna
│   ├── trainer.py                # Wrapper cho quá trình training
│   ├── evaluate.py               # Chạy metrics trên Validation/Test set
│   ├── inference.py              # Save/Load model và Generator API
│   ├── callbacks.py              # Custom logging/TensorBoard callbacks
│   └── utils.py                  # Các tiện ích (Logger, Seed, Timer)
├── main.py                       # CLI Entry Point điều phối workflow
├── compare.py                    # Công cụ so sánh các runs
├── README.md                     # Tài liệu hướng dẫn
└── requirements.txt              # Danh sách thư viện phụ thuộc
```

Mỗi file trong `src/` tuân thủ nguyên tắc Single Responsibility (Đơn nhiệm) giúp bạn dễ dàng scale up hoặc thêm thuật toán mới.

## 10. Mẹo Xử Lý Dữ Liệu Nâng Cao (Advanced Data Tips)

Chất lượng của mô hình GPT-2 phụ thuộc phần lớn vào dữ liệu. Dưới đây là vài mẹo tiền xử lý:
- **Lọc dữ liệu rác (Noise filtering):** Loại bỏ các chuỗi chỉ chứa ký tự đặc biệt, URL trống hoặc HTML tags trước khi feed vào mô hình.
- **Cân bằng nhãn (Label balancing):** Đối với classification, hãy cân nhắc oversampling các lớp thiểu số (minority classes).
- **Tăng cường dữ liệu (Data Augmentation):** Nếu tập train nhỏ, bạn có thể thiết lập `augmentation_enabled: true` trong YAML config để áp dụng các kỹ thuật như thay thế từ đồng nghĩa (synonym replacement).

Chúc bạn đạt được kết quả xuất sắc với dự án này!
