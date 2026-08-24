# Vietnamese dataset for GPT fine-tuning
# Format: (prompt, completion) pairs for causal language model training

dataset_gpt = [
    # --- Truyện cổ tích / Fairy tales ---
    ("Ngày xửa ngày xưa, ở một ngôi làng nhỏ,",
     "có một cô gái mồ côi sống với bà ngoại trong căn nhà tranh rách nát."),
    ("Trong khu rừng sâu thẳm,",
     "có một con rồng già canh giữ kho báu từ nghìn năm trước."),
    ("Vị vua già nhìn ra ngoài cửa sổ và thở dài,",
     "vì ba hoàng tử đều không ai chịu kế thừa ngai vàng."),
    ("Chú bé chăn trâu ngồi trên lưng trâu,",
     "thổi sáo một bài ca vui vẻ, tiếng sáo vang khắp cánh đồng."),

    # --- Công nghệ / Technology ---
    ("Trí tuệ nhân tạo đang thay đổi thế giới,",
     "từ y tế, giáo dục cho đến sản xuất công nghiệp đều được tự động hóa."),
    ("Để huấn luyện một mô hình ngôn ngữ lớn,",
     "bạn cần một lượng dữ liệu khổng lồ và tài nguyên tính toán mạnh mẽ."),
    ("Học sâu (Deep Learning) là một nhánh của",
     "học máy, sử dụng mạng nơ-ron nhiều tầng để học các biểu diễn phức tạp từ dữ liệu."),
    ("Khi fine-tune một mô hình GPT,",
     "bước đầu tiên là chuẩn bị dữ liệu chất lượng cao phù hợp với miền ứng dụng."),

    # --- Ẩm thực / Food ---
    ("Phở là món ăn truyền thống nổi tiếng nhất của Việt Nam,",
     "với nước dùng đậm đà từ xương bò ninh hàng giờ, kèm bánh phở mềm và thịt thái mỏng."),
    ("Muốn nấu một nồi bún bò Huế ngon,",
     "bí quyết nằm ở việc chọn sả tươi, mắm ruốc Huế chính gốc và ninh xương heo thật kỹ."),
    ("Bánh mì Sài Gòn nổi tiếng khắp thế giới",
     "nhờ lớp vỏ giòn rụm, nhân phong phú với pa-tê, chả lụa, đồ chua và rau thơm."),
    ("Cà phê sữa đá Việt Nam đặc biệt ở chỗ",
     "cà phê được pha bằng phin, nhỏ từng giọt đậm đặc, hòa quyện với sữa đặc ngọt ngào."),

    # --- Du lịch / Travel ---
    ("Vịnh Hạ Long là di sản thiên nhiên thế giới,",
     "nơi hàng nghìn hòn đảo đá vôi nhô lên giữa làn nước xanh biếc, tạo nên cảnh quan kỳ vĩ."),
    ("Phố cổ Hội An về đêm",
     "lung linh ánh đèn lồng đủ sắc màu, du khách thả đèn hoa đăng trên sông Hoài thơ mộng."),
    ("Đà Lạt được mệnh danh là thành phố ngàn hoa,",
     "với khí hậu mát mẻ quanh năm, những đồi thông xanh ngát và vườn hoa rực rỡ."),

    # --- Động lực / Motivation ---
    ("Thành công không đến từ sự may mắn,",
     "mà đến từ sự kiên trì, nỗ lực không ngừng và lòng dũng cảm vượt qua thất bại."),
    ("Mỗi ngày là một cơ hội mới,",
     "hãy bắt đầu bằng một nụ cười và quyết tâm làm tốt hơn ngày hôm qua."),

    # --- Lịch sử / History ---
    ("Trận Điện Biên Phủ năm 1954",
     "là chiến thắng lịch sử chấn động địa cầu, kết thúc chín năm kháng chiến chống thực dân Pháp."),
    ("Văn Miếu - Quốc Tử Giám được xây dựng năm 1070,",
     "là trường đại học đầu tiên của Việt Nam, nơi đào tạo nhân tài cho đất nước suốt nhiều thế kỷ."),
]


if __name__ == "__main__":
    print(f"Total samples: {len(dataset_gpt)}\n")
    for i, (prompt, completion) in enumerate(dataset_gpt):
        print(f"[{i+1:2d}] Prompt:     {prompt}")
        print(f"     Completion: {completion}\n")
