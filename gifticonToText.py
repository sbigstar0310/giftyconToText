import os
import tensorflow as tf
import numpy as np
import easyocr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict


def encodeInput(text):
    encodedResult = []
    tokens = myTokenization(text)
    for token in tokens:
        if token_dict.get(token):
            encodedResult.append(token_dict[token])
        else:
            encodedResult.append(0)
    return encodedResult


def preprocess(input_texts):
    encoded_inputs = [encodeInput(input_text) for input_text in input_texts]
    padded_encoded_inputs = pad_sequences(
        encoded_inputs, maxlen=max_len, padding="post"
    )
    return padded_encoded_inputs


# 데이터의 문장 자체가 길지 않기 때문에 한 글자씩 떼어서 토큰화.
def myTokenization(text):
    result = list(text)
    return result


# 학습 데이터 (브랜드명, 상품명 등을 포함한 텍스트와 해당 레이블)
data_inputs = [
    ["Lifestyle Platform", 2],
    ["감사의 마음을 전합니다.", 2],
    ["유효기간", 2],
    ["유호기간", 2],
    ["(1588-9282)전화주문 홈페이지", 2],
    ["기프티콘", 2],
    ["주문번호", 2],
    ["수량/금액", 2],
    ["사용처", 2],
    ["사용기한", 2],
    ["교환처", 2],
    ["gifticon", 2],
    ["kakaotalk", 2],
    ["선물하기", 2],
    ["()", 2],
    ["2023년 09월 20일", 2],
    ["년 월 일", 2],
    ["배스킨라빈스", 0],
    ["스타벅스 상품권", 0],
    ["스타벅스", 0],
    ["이디야커피", 0],
    ["이마트, 신세계상품권", 0],
    ["도미노피자", 0],
    ["본죽", 0],
    ["메가MGC커피", 0],
    ["S-OIL", 0],
    ["투썸 플레이스", 0],
    ["이마트24", 0],
    ["동아제약", 0],
    ["BBQ", 0],
    ["스타벅스", 0],
    ["올리브영", 0],
    ["GS25", 0],
    ["스타벅스 상품권", 0],
    ["교촌치킨", 0],
    ["아웃백", 0],
    ["설빙", 0],
    ["60계치킨", 0],
    ["맘스터치", 0],
    ["뚜레주르", 0],
    ["교촌치킨", 0],
    ["배달의 민족", 0],
    ["배스킨라빈스", 0],
    ["BHC", 0],
    ["CU", 0],
    ["CGV", 0],
    ["롯데시네마", 0],
    ["메가커피", 0],
    ["파리바게트", 0],
    ["굽네치킨", 0],
    ["투썸플레이스", 0],
    ["씨스페이스", 0],
    ["롯데리아", 0],
    ["공차", 0],
    ["빕스", 0],
    ["멕시카나치킨", 0],
    ["샐러드바 2인 디너/주말/공휴일 식사권", 1],
    ["샐러드바 2인 디너/주말/공휴일 식사권 + 안심 스테이크 150g", 1],
    ["파리생제르맹 축구공케이크", 1],
    ["오리온)마이구미복숭아66g", 1],
    ["타로 로브 포션", 1],
    ["치킨버거 세트", 1],
    ["cs해태]자유시간골드36g", 1],
    ["항금올리브치컨반반+콜라 1.25L", 1],
    ["남양)초콜릿드링크초코에몽190ML", 1],
    ["배민상품권 2만원 교환권", 1],
    ["버라이어티팩(배달가능)", 1],
    ["아이스 카페 아메리카노 T", 1],
    ["마이넘버원 케이크(배달가능)", 1],
    [
        "아이스 카페 아메리카노 T 2잔 + 딸기 초코 레이어 케이크 + 딸기 수플레 치즈 케이크",
        1,
    ],
    ["GS25 모바일 상품권 5만원권", 1],
    ["GS25 모바일 상품권 3만원권", 1],
    ["GS25 모바일 상품권 5천원권", 1],
    ["모바일 상품권 20,000원", 1],
    ["모바일 상품권 10,000원", 1],
    ["모바일 상품권 5,000원", 1],
    ["허니콤보웨지감자세트", 1],
    ["미스터트리오L", 1],
    ["2인 PKG (관람권 2매 + 고소팝콘L 1개 + 콜라M 2개)", 1],
    ["모바일주유상품권 5만원권(교환권)", 1],
    ["콩닥콩닥 초코베어 (데코픽 포함)", 1],
    ["스트로베리 초콜릿 생크림", 1],
    ["신세계상품권 모바일교환권 100,000원(이마트 교환전용)", 1],
    ["파인트 아이스크림(배달가능)", 1],
    ["e카드 3만원 교환권", 1],
    ["카페 아메리카노 T 2잔 + 부드러운 생크림 카스텔라", 1],
    ["모바일 상품권 50,000원", 1],
    ["모바일 상품권 30,000원", 1],
    ["후라이드+콜라1.25L", 1],
    ["GS25 모바일 상품권 2만원권", 1],
    ["GS25 모바일 상품권 1만원권", 1],
    ["(홀)더 진한 캐롯케익", 1],
    ["(홀)러브 뉴 스트로베리 치즈케이크(데코픽 포함)", 1],
    ["샐러드바 2인 디너/주말/공휴일 식사권 + 안심 스테이크 150g", 1],
    ["후르츠 바스켓", 1],
    ["롯데 키세스쿠앤크52g", 1],
]

# 데이터 섞기
np.random.shuffle(data_inputs)

# 학습 데이터와 검증 데이터로 나누기
train_samples_index = int(len(data_inputs) * 0.75)

test_inputs = data_inputs[:train_samples_index]
val_inputs = data_inputs[train_samples_index:]

test_texts = [dic[0] for dic in test_inputs]
test_labels = np.array([dic[1] for dic in test_inputs])

val_texts = [dic[0] for dic in val_inputs]
val_labels = np.array([dic[1] for dic in val_inputs])

# 0: 브랜드명, 1: 상품명, 2: 이외
classes = {0: "브랜드명", 1: "상품명", 2: "이외"}

token_texts = []
encoded_texts = []
token_dict = dict()
dict_index = 0

# 토큰화
for text in test_texts:
    tokens = myTokenization(text)
    token_texts.append(tokens)

    for token in tokens:
        if not token_dict.get(token):
            token_dict[token] = dict_index
            dict_index += 1

    encoded_texts.append([token_dict[token] for token in tokens])

# 시퀀스 패딩
max_len = max([len(seq) for seq in encoded_texts])
padded_encoded_sequences = pad_sequences(encoded_texts, maxlen=max_len, padding="post")
val_padded_encoded_sequences = preprocess(val_texts)

# 모델 저장 디렉토리
checkpoint_dir = "./model_checkpoints"

# 모델 파일 이름 포맷
checkpoint_name_format = "model_checkpoint_val_acc_{val_accuracy:.4f}.h5"

# 모델 파일 경로
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name_format)

# 정확도가 최고로 갱신될 때 모델 저장하는 콜백 정의
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor="val_accuracy",  # 모니터링할 지표 선택 (여기서는 검증 데이터의 정확도)
    mode="max",  # 모델 저장 방식 선택 (여기서는 최대값일 때 저장)
    save_best_only=True,  # 최고 성능의 모델만 저장
)

# 모델 정의
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=len(token_dict) + 1, output_dim=32, input_length=max_len
        ),
        tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding="valid"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# 딕셔너리를 저장
with open("dictionary.txt", "w") as file:
    for key, value in token_dict.items():
        file.write(f"{key}: {value}\n")

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 모델 요약
model.summary()

# 모델 훈련
model.fit(
    x=padded_encoded_sequences,
    y=test_labels,
    validation_data=(val_padded_encoded_sequences, val_labels),
    epochs=200,
    verbose=2,
    callbacks=[checkpoint_callback],
)

# HDF5 모델 불러오기
model = tf.keras.models.load_model(
    "/content/model_checkpoints/model_checkpoint_val_acc_0.7500.h5"
)

# TensorFlow Lite 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TensorFlow Lite 모델 저장 경로
tflite_model_path = "my_model.tflite"

# TensorFlow Lite 모델 저장
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

reader = easyocr.Reader(
    ["ko", "en"], gpu=True
)  # this needs to run only once to load the model into memory

# ocr_result = reader.readtext('/content/test05.jpg', detail=0, paragraph=False)
ocr_result = [
    "모바일상품권",
    "GS25",
    "30,000",
    "Lifestyle Platform GS25",
    "GS25",
    "GS25 모바일 상품권 3만원권",
    "GS25",
    "9826 2429 4270 5221",
    "교환처",
    "유효기간",
    "주문번호",
    "GS25",
    "2020년 09월 10일",
    "678845935",
    "kakaotalk C 선물하기",
]
print(ocr_result)


def chooseBrandProductName(results):
    product_name_max = 0
    product_name_index = 0
    brand_name_max = 0
    brand_name_index = 0

    for index, result in enumerate(results):
        if result[0] > brand_name_max:
            brand_name_max = result[0]
            brand_name_index = index
        if result[1] > product_name_max:
            product_name_max = result[1]
            product_name_index = index

    return (brand_name_index, product_name_index)


def is_numeric_space(text):
    """
    문자열이 숫자와 공백으로만 이루어져 있는지 확인하는 함수
    """
    return all(char.isdigit() or char.isspace() for char in text)


def isContain(text):
    """
    다음 단어를 포함하는 경우 제외
    """
    del_text = [
        "유효",
        "기간",
        "주문",
        "번호",
        "교환",
        "합니다",
        "사용",
        "기한",
        "gifticon",
        "기프티콘",
        "수량",
        "금액",
        "kakaotalk",
    ]

    for dtext in del_text:
        if dtext in text:
            return True

    return False


input_texts = []

for r in ocr_result:
    if not is_numeric_space(r) and not isContain(r):
        input_texts.append(r)

# 중복 제거
input_texts = list(set(input_texts))

# print(input_texts)
preprocess_input = preprocess(input_texts)
# print(preprocess_input)
# print(token_dict)

model_path = "/content/model_checkpoints/model_checkpoint_val_acc_0.7500.h5"
loaded_model = tf.keras.models.load_model(model_path)
results = loaded_model.predict(preprocess_input)

for index, result in enumerate(results):
    print(
        f"{result[np.argmax(result)]} | {input_texts[index]} : {classes[np.argmax(result)]}"
    )

print(
    f"브랜드 이름: {input_texts[chooseBrandProductName(results)[0]]}, 상품명 : {input_texts[chooseBrandProductName(results)[1]]}"
)
