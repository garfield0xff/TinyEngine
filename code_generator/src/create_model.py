import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np

# 사람을 식별하는 MobileNetV2 모델 생성
def create_person_detection_model(input_shape=(224, 224, 3)):
    # MobileNetV2 모델 로드 (ImageNet 가중치 사용, 최상위 레이어 제외)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # 모델 동결 (미리 학습된 가중치 고정)
    base_model.trainable = False

    # 사람을 식별할 수 있도록 커스터마이징
    model = models.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')  # 사람/비사람 이진 분류를 위한 출력 레이어
    ])

    return model

# 모델 생성
model = create_person_detection_model()

# 모델 구조 출력
model.summary()

# 모델 컴파일 (이진 분류를 위해 binary_crossentropy 사용)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite 모델 저장
with open('person_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
