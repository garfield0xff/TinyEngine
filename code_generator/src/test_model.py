import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path="/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/person_detection_model.tflite")
interpreter.allocate_tensors()

# 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 예측할 이미지 로드 및 전처리
img_path = '/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/test/person3.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # 모델이 예상하는 범위로 정규화
# print(img_array);

# 모델에 이미지 입력 설정
interpreter.set_tensor(input_details[0]['index'], img_array)

# 추론 수행
interpreter.invoke()

# 결과 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data);

# 출력 결과 확인
if output_data[0] > 0.7:
    print("Prediction: This image contains a person.")
else:
    print("Prediction: This image does not contain a person.")
