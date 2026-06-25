# Lab 21 - Báo cáo đánh giá

**Học viên**: Đào Duy Quyền - 2A202600676  
**Ngày nộp**: 2026-06-25  
**Hình thức nộp**: Option B - Link GitHub  

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Phương pháp fine-tuning**: QLoRA 4-bit kết hợp LoRA adapters, sử dụng Unsloth và TRL `SFTTrainer`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, shuffle với seed 42 và chọn 200 mẫu
- **Chia tập dữ liệu**: 180 mẫu train, 20 mẫu eval
- **Định dạng dữ liệu**: Alpaca format gồm `instruction`, `input`, `output`, sau đó chuyển thành một trường `text` để huấn luyện SFT
- **Phân tích độ dài token**: min = 25, max = 738, p50 = 227, p95 = 562, p99 = 704
- **max_seq_length**: 1024, làm tròn lên từ p95 và giới hạn để phù hợp với GPU T4
- **GPU**: Tesla T4, 14.563 GB VRAM
- **Hyperparameters**: 3 epochs, learning rate 2e-4, cosine scheduler, warmup ratio 0.10, batch size 1, gradient accumulation 8, effective batch size 8, optimizer `adamw_8bit`, `packing=False`
- **LoRA target modules**: `q_proj`, `v_proj`
- **Chi phí huấn luyện ước tính**: khoảng 10.8 phút cho toàn bộ 3 adapter, tương đương khoảng $0.06-$0.07 nếu tính $0.35/giờ
- **Thư mục kết quả**: `lab21_lora_t4/`

## 2. Kết quả thí nghiệm rank

Ba adapter được huấn luyện trên cùng base model, cùng dataset split, cùng seed và cùng hyperparameters. Yếu tố thay đổi duy nhất là LoRA rank và alpha.

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 3.58 phút | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 3.66 phút | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 3.57 phút | 8.00 GB | 1.4768 | 4.3790 |

Kết quả perplexity đi theo xu hướng hợp lý: khi tăng rank, adapter có nhiều năng lực biểu diễn hơn nên eval loss và perplexity giảm. Rank 64 đạt perplexity tốt nhất, nhưng cần 14.75M tham số trainable, gấp 4 lần rank 16 và gấp 8 lần rank 8. Rank 16 là lựa chọn cân bằng nhất trong thí nghiệm này vì cải thiện rõ so với rank 8 nhưng vẫn giữ adapter nhỏ. Rank 64 có cải thiện thêm, tuy nhiên mức cải thiện nhỏ hơn so với chi phí tăng thêm về số tham số và VRAM.

## 3. Phân tích loss curve

Loss curve được lưu tại `lab21_lora_t4/loss_curve.png`. Trong notebook T4, eval trong lúc train được tắt để tránh áp lực VRAM, nên biểu đồ loss dựa trên training loss được log trong quá trình train. Evaluation cuối cùng được chạy sau khi lưu từng adapter.

Training loss giảm ở cả ba rank:

| Rank | Train Loss đầu | Train Loss cuối | Eval Loss cuối |
|---:|---:|---:|---:|
| 8 | 1.6165 | 1.4388 | 1.5577 |
| 16 | 1.6143 | 1.3942 | 1.5161 |
| 64 | 1.6016 | 1.2768 | 1.4768 |

Không có dấu hiệu overfitting nghiêm trọng từ eval loss cuối cùng, vì rank có capacity cao hơn không chỉ giảm training loss mà cũng giảm eval loss. Tuy nhiên, khoảng cách giữa training loss và eval loss của rank 64 lớn hơn, nên nếu dùng rank 64 cho production thì cần kiểm tra thêm trên tập eval lớn hơn. Với chỉ 20 mẫu eval, tôi không kết luận rank 64 luôn tốt hơn trong mọi trường hợp; tôi chỉ kết luận rằng rank 64 có kết quả tốt nhất trên split nhỏ này.

## 4. So sánh định tính

Phần so sánh định tính được sinh bằng base model và adapter fine-tuned rank 16. Kết quả đầy đủ được lưu trong `lab21_lora_t4/qualitative_comparison.csv`.

### Ví dụ 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base**: Base model giải thích machine learning là một nhánh của AI, học từ dữ liệu để dự đoán hoặc hành động.

**Fine-tuned r=16**: Mô hình fine-tuned trả lời trực tiếp hơn, nhấn mạnh việc học từ dữ liệu mà không cần hướng dẫn trực tiếp từ người dùng.

**Nhận xét**: Cải thiện. Câu trả lời của adapter rõ hơn cho người mới bắt đầu và gần với phong cách instruction-answer hơn.

### Ví dụ 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base**: Base model đề xuất dùng đệ quy hoặc vòng lặp, nhưng đoạn code bắt đầu với quy ước chỉ số chưa thật sạch.

**Fine-tuned r=16**: Adapter r=16 dùng kiểm tra input và vòng lặp với `a, b = 0, 1`, thực tế hơn khi triển khai.

**Nhận xét**: Cải thiện. Câu trả lời fine-tuned có tính ứng dụng tốt hơn và tránh mặc định dùng đệ quy kém hiệu quả.

### Ví dụ 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base**: Base model bắt đầu bằng phần giải thích dài về tính thân thiện với người dùng và hơi lan man.

**Fine-tuned r=16**: Adapter đưa ra danh sách các nguyên tắc như chuyển đổi, thích ứng, đơn giản và tương thích.

**Nhận xét**: Trung bình. Câu trả lời fine-tuned có dạng danh sách rõ hơn, nhưng một số cách diễn đạt chưa thật tự nhiên đối với chủ đề UI/UX.

### Ví dụ 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base**: Base model nhận diện đúng LoRA là Low-Rank Adaptation và QLoRA là Quantized LoRA, nhưng giải thích còn chung chung.

**Fine-tuned r=16**: Adapter r=16 mở rộng sai LoRA thành "Layer-wise Adaptive Regularization Optimization", đây là lỗi kiến thức.

**Nhận xét**: Giảm chất lượng. Đây là một failure case quan trọng: fine-tuning trên dataset nhỏ và khá tổng quát không đảm bảo cải thiện kiến thức chuyên sâu, thậm chí có thể làm mô hình sinh câu trả lời nghe hợp lý nhưng sai.

### Ví dụ 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base**: Base model giải thích được ba phương pháp, nhưng phần phân biệt còn rộng và hơi lặp.

**Fine-tuned r=16**: Adapter đưa ra phần so sánh có cấu trúc hơn và dùng tiếng Việt dễ đọc hơn khi nói về prompt engineering và fine-tuning.

**Nhận xét**: Cải thiện nhẹ. Câu trả lời dễ đọc hơn, nhưng nếu thêm ví dụ cụ thể cho từng phương pháp thì sẽ tốt hơn.

## 5. Kết luận về trade-off của rank

Trong thí nghiệm này, rank 64 có perplexity tốt nhất với PPL = 4.3790, nhưng đây không nhất thiết là lựa chọn tối ưu trong mọi tình huống. Rank 64 cần 14.75M tham số trainable, gấp 4 lần rank 16 và gấp 8 lần rank 8. Mức cải thiện từ rank 16 sang rank 64 là từ 4.5544 xuống 4.3790 PPL, có thật nhưng không lớn so với chi phí tăng thêm về kích thước adapter và VRAM. Rank 8 nhỏ nhất và vẫn học được style của dataset, nhưng eval loss cao nhất nên có vẻ bị thiếu capacity. Nếu deploy production cho dataset này, tôi chọn rank 16 vì ROI tốt nhất: adapter còn nhỏ, train nhanh, dùng VRAM thấp nhất trong run này và perplexity gần rank 64 hơn so với rank 8. Nếu yêu cầu chất lượng tuyệt đối và có GPU đủ mạnh, rank 64 đáng để thử tiếp. Tuy nhiên, với tập eval chỉ 20 mẫu, tôi sẽ ưu tiên rank 16 và mở rộng evaluation trước khi tăng rank.

Một điểm quan trọng là fine-tuning không nên được xem là cách bổ sung kiến thức. Ví dụ LoRA/QLoRA cho thấy adapter r=16 có thể tạo câu trả lời sai về tên đầy đủ của LoRA. Điều này phù hợp với lý thuyết: fine-tuning phù hợp để điều chỉnh style, format và hành vi trả lời, còn knowledge gap nên xử lý bằng RAG hoặc bằng dataset domain-specific chất lượng cao hơn. Kết quả này giúp tôi hiểu rõ hơn vì sao cần kết hợp quantitative metrics với qualitative review, thay vì chỉ nhìn vào perplexity.

## 6. Những điều tôi học được

- LoRA rank là trade-off giữa capacity và chi phí: rank cao hơn có thể giảm perplexity, nhưng lợi ích biên giảm nhanh khi số tham số trainable tăng mạnh.
- Perplexity chỉ là một phần của evaluation. Qualitative examples vẫn rất cần thiết vì mô hình có thể có PPL tốt hơn nhưng vẫn sinh câu trả lời sai về mặt kiến thức.
- QLoRA với base model 4-bit và optimizer `adamw_8bit` giúp fine-tune model 3B trên T4 khá nhẹ, nhưng vẫn cần quản lý VRAM cẩn thận: tắt mid-train eval, save adapter trước khi evaluate và clear cache giữa các lần train rank.
- Fine-tuning phù hợp nhất khi cần điều chỉnh style hoặc format. Nếu bài toán cần kiến thức chính xác, nên dùng RAG hoặc bổ sung dữ liệu domain-specific chất lượng cao trước khi kết luận adapter tốt.

## Checklist nộp bài

- `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`: notebook đã chạy thí nghiệm.
- `lab21_lora_t4/r8/`: adapter rank 8.
- `lab21_lora_t4/r16/`: adapter rank 16.
- `lab21_lora_t4/r64/`: adapter rank 64.
- `lab21_lora_t4/rank_experiment_summary.csv`: bảng metric cho 3 rank.
- `lab21_lora_t4/qualitative_comparison.csv`: so sánh định tính base vs r16.
- `lab21_lora_t4/loss_curve.png`: loss curve từ trainer logs.
