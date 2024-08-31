from my_transformer import Transformer
import torch

# 모델 초기화
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1
)

# 가짜 입력 데이터로 테스트
src = torch.randint(0, 10000, (32, 100))
tgt = torch.randint(0, 10000, (32, 100))

output = model(src, tgt)
print(output.shape)  # (32, 100, 10000)
