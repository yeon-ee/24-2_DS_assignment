import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# transformer.py에서 Transformer 불러오기
from my_transformer.my_transformer import Transformer

# 1. AG News 데이터셋 로드 및 전처리
tokenizer = get_tokenizer("basic_english")

# AG News 데이터셋에서 토큰 생성
def yield_tokens(data_iter):
    for _, line in data_iter:
        yield tokenizer(line)

# AG News 데이터셋 로드
train_iter, test_iter = AG_NEWS(split=('train', 'test'))

# 단어 사전(vocab) 생성
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 텍스트를 텐서로 변환
def text_pipeline(x):
    return vocab(tokenizer(x))

# 라벨을 텐서로 변환 (카테고리 번호로 변환)
def label_pipeline(x):
    return int(x) - 1  # AG_NEWS 라벨이 1부터 시작하므로 0부터 시작하도록 조정

# 데이터를 텐서로 변환하고 패딩 처리하는 함수
def collate_batch(batch):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(torch.tensor(label_pipeline(_label), dtype=torch.long))
        text_list.append(torch.tensor(text_pipeline(_text), dtype=torch.long))
    return torch.stack(label_list), pad_sequence(text_list, batch_first=True)

# DataLoader 생성
batch_size = 16
train_iter, test_iter = AG_NEWS(split=('train', 'test'))
train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# 2. 트랜스포머 모델 정의
src_vocab_size = len(vocab)  # AG News 데이터셋의 단어 사전 크기
tgt_vocab_size = 4  # AG News의 4개 클래스
d_model = 512  # 임베딩 차원
n_heads = 8  # 멀티헤드 어텐션의 헤드 수
d_ff = 2048  # 피드포워드 레이어의 차원
num_encoder_layers = 6  # 인코더 레이어 수
num_decoder_layers = 6  # 디코더 레이어 수
dropout = 0.1  # 드롭아웃 비율

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, num_encoder_layers, num_decoder_layers, dropout)

# 3. 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. 모델 학습 함수
def train_model(model, criterion, optimizer, train_dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for labels, inputs in train_dataloader:
            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 모델의 예측 결과
            outputs = model(inputs, inputs)

            # 손실 계산
            loss = criterion(outputs, labels)

            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

# 5. 모델 테스트 함수
def test_model(model, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, inputs in test_dataloader:
            outputs = model(inputs, inputs)
            _, predicted = torch.max(outputs, 1)  # 가장 높은 값의 인덱스를 예측 값으로 사용
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# 6. 모델 학습 및 테스트 실행
num_epochs = 5
train_model(model, criterion, optimizer, train_dataloader, num_epochs)
test_model(model, test_dataloader)
