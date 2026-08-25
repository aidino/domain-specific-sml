"""
Module cung cấp các custom callbacks cho HuggingFace Trainer.
Chứa các callbacks để log thông tin, sinh text mẫu, và ghi metric vào TensorBoard.
"""

import time # Thư viện để đo thời gian
import math # Thư viện toán học để tính perplexity
import logging # Thư viện ghi log

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments # Import các class từ transformers
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter để ghi TensorBoard
from src.utils import setup_logger, format_duration, format_metrics_table # Import các hàm tiện ích từ utils

# Khởi tạo logger cho file này
logger = setup_logger(__name__)

class RichLoggingCallback(TrainerCallback):
    """
    Callback để log thông tin training ra console đẹp mắt.
    Thay thế cho log mặc định của Trainer.
    """

    def __init__(self):
        """Khởi tạo callback, thiết lập thời gian bắt đầu training."""
        # Lưu thời gian bắt đầu training
        self.start_time = None
        # Biến lưu trữ loss nhỏ nhất để theo dõi
        self.best_loss = float("inf")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Được gọi khi bắt đầu quá trình training."""
        # Bắt đầu đếm thời gian
        self.start_time = time.time()
        # In ra thông tin banner bắt đầu training
        logger.info(f"Bắt đầu training: {state.max_steps} steps, {args.num_train_epochs} epochs")
        # In ra thông tin về thiết bị (GPU/CPU)
        logger.info(f"Thiết bị: {args.device}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict = None, **kwargs):
        """Được gọi mỗi khi Trainer ghi log (mỗi logging_steps)."""
        # Nếu không có log thì bỏ qua
        if not logs:
            return

        # Chỉ xử lý log của training (chứa loss)
        if "loss" in logs:
            # Lấy giá trị training loss
            loss = logs["loss"]
            # Lấy giá trị learning rate
            lr = logs.get("learning_rate", 0.0)
            # In ra log dưới dạng: Step X/Y | Loss: Z | LR: W
            logger.info(f"Step {state.global_step}/{state.max_steps} | Train Loss: {loss:.4f} | LR: {lr:.2e}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Được gọi khi kết thúc mỗi epoch."""
        # In ra dòng thông báo kết thúc epoch
        logger.info(f"Kết thúc epoch {round(state.epoch, 2)}")
        
        # Nếu có thông tin metric trong state (eval metrics)
        if state.log_history:
            # Lấy log gần nhất
            last_log = state.log_history[-1]
            
            # Cập nhật best loss nếu cần
            if "eval_loss" in last_log:
                # So sánh và cập nhật
                self.best_loss = min(self.best_loss, last_log["eval_loss"])
            
            # Lọc ra các metric liên quan đến epoch này
            metrics = {k: v for k, v in last_log.items() if k.startswith("eval_") or k == "loss"}
            
            # Nếu có metric để in
            if metrics:
                # Tính toán perplexity nếu có eval_loss
                if "eval_loss" in metrics:
                    try:
                        # Perplexity = exp(eval_loss)
                        metrics["perplexity"] = math.exp(metrics["eval_loss"])
                    except OverflowError:
                        # Xử lý khi loss quá lớn
                        metrics["perplexity"] = float("inf")
                
                # Format và in ra bảng metric
                table_str = format_metrics_table(metrics, title=f"Tổng kết Epoch {round(state.epoch, 2)}")
                logger.info("\n" + table_str)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Được gọi khi kết thúc toàn bộ quá trình training."""
        # Tính toán tổng thời gian training
        total_time = time.time() - self.start_time
        # Format thời gian thành chuỗi dễ đọc
        duration_str = format_duration(total_time)
        
        # In ra thông báo hoàn thành
        logger.info("🎉 Quá trình training đã hoàn tất!")
        # In ra tổng thời gian
        logger.info(f"Tổng thời gian: {duration_str}")
        
        # In ra best loss nếu có
        if self.best_loss != float("inf"):
            logger.info(f"Metric tốt nhất (eval_loss): {self.best_loss:.4f}")


class GenerationSampleCallback(TrainerCallback):
    """
    Callback để sinh text mẫu ở cuối mỗi epoch.
    Chỉ hoạt động đối với task language modeling.
    """

    def __init__(self, sample_prompt: str, tokenizer, task_type: str = "causal_lm"):
        """
        Khởi tạo callback.
        
        Args:
            sample_prompt: Câu prompt để bắt đầu sinh text.
            tokenizer: Tokenizer dùng để encode/decode.
            task_type: Loại task (chỉ active nếu là causal_lm hoặc completion).
        """
        # Lưu lại câu prompt mẫu
        self.sample_prompt = sample_prompt
        # Lưu tokenizer
        self.tokenizer = tokenizer
        # Lưu loại task
        self.task_type = task_type

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Sinh và in text mẫu ở cuối epoch."""
        # Nếu không phải task sinh text thì bỏ qua
        if self.task_type not in ["causal_lm", "completion"]:
            return
            
        # Nếu không có model thì không thể sinh text
        if model is None:
            return

        # Chuyển model sang chế độ đánh giá
        model.eval()
        # Đưa prompt vào tokenizer và lấy tensor input trên cùng thiết bị với model
        inputs = self.tokenizer(self.sample_prompt, return_tensors="pt").to(args.device)

        try:
            # Sinh text bằng model.generate()
            outputs = model.generate(
                **inputs,
                temperature=0.7,         # Nhiệt độ để tạo tính đa dạng (đã được fix = 0.7 theo thiết kế)
                max_new_tokens=50,       # Số lượng token sinh thêm tối đa
                do_sample=True,          # Bật chế độ lấy mẫu
                pad_token_id=self.tokenizer.eos_token_id, # Sử dụng eos_token_id làm pad_token_id
            )
            
            # Giải mã kết quả đầu ra thành text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # In kết quả sinh ra console
            logger.info(f"📝 Sample generation: '{self.sample_prompt}' → '{generated_text}'")
        except Exception as e:
            # Log lỗi nếu có trong quá trình sinh
            logger.error(f"Lỗi khi sinh text mẫu: {e}")
        
        # Chuyển model lại chế độ training
        model.train()


class TensorBoardMetricsCallback(TrainerCallback):
    """
    Callback tùy chỉnh để ghi thêm các metric đặc biệt (như perplexity) vào TensorBoard.
    """

    def __init__(self, log_dir: str):
        """
        Khởi tạo callback.
        
        Args:
            log_dir: Thư mục để lưu log TensorBoard.
        """
        # Khởi tạo SummaryWriter với đường dẫn log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict = None, **kwargs):
        """Ghi metric vào TensorBoard mỗi khi evaluate xong."""
        # Nếu không có metric thì bỏ qua
        if not metrics:
            return

        # Kiểm tra xem có eval_loss không để tính perplexity
        if "eval_loss" in metrics:
            # Lấy giá trị eval_loss
            eval_loss = metrics["eval_loss"]
            try:
                # Tính perplexity = exp(eval_loss)
                perplexity = math.exp(eval_loss)
                # Ghi giá trị perplexity vào TensorBoard tại bước hiện tại
                self.writer.add_scalar("eval/perplexity", perplexity, state.global_step)
            except OverflowError:
                # Nếu loss quá lớn dẫn đến tràn số thì bỏ qua
                pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Được gọi khi kết thúc training để đóng SummaryWriter."""
        # Đóng writer để xả dữ liệu ra file
        self.writer.close()
