import tensorflow as tf
import numpy as np
import cv2

# Convolution 연산 수행 함수
def conv2d(image, kernel, bias, stride=1, padding='valid'):
    (n, h, w, c) = image.shape
    (num_filters, kh, kw, kc) = kernel.shape
    kernel = kernel.astype(np.float32)
    image = image.astype(np.float32)
    print(f"image shape {image.shape}")

    if padding == 'same':
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
        image = np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    else:
        pad_h = 0
        pad_w = 0

    out_h = (h + 2 * pad_h - kh) // stride + 1
    print(out_h)
    out_w = (w + 2 * pad_w - kw) // stride + 1
    print(out_w)
    print(f"image shape {image.shape}")

    output = np.zeros((n, out_h, out_w, num_filters))
    
    # for i in range(10):
    #     print(f"image[0, 0, {i}, 0] value: {image[0, 0, i, 0]}")
    
    print(f"bias[0] value: {bias[0]}")

    flat_kernel = kernel.flatten()
    print(f"First element of flattened kernel: {flat_kernel[0]}")

    # (kh * kw * kc)는 각 필터의 크기입니다.
    kernel_size = kh * kw * kc
    print(out_h, out_w, num_filters, kh, kw, kc)
    flag = 0;

    for i in range(out_h):
        for j in range(out_w):
            for k in range(num_filters):
                acc = 0.0
                for p in range(kh):
                    for q in range(kw):
                        for r in range(kc):
                            # flat_kernel에서 직접 인덱스로 접근
                            flat_kernel_index = k * kernel_size + p * kw * kc + q * kc + r
                            acc += np.sum(image[:, i*stride+p, j*stride+q, r] * flat_kernel[flat_kernel_index])

                # 결과를 출력 텐서에 저장
                output[:, i, j, k] = acc

                
    
    output += bias

    return output

def depthwise_conv2d(image, depthwise_kernel, bias, stride=1, padding='valid'):
    (n, h, w, c) = image.shape  # image.shape = (1, 112, 112, 32)
    (kh, kw, kc, channel_multiplier) = depthwise_kernel.shape  # depthwise_kernel.shape = (1, 3, 3, 32)

    # 패딩 처리
    if padding == 'same':
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
        image = np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    else:
        pad_h = 0
        pad_w = 0

    out_h = (h + 2 * pad_h - kh) // stride + 1
    out_w = (w + 2 * pad_w - kw) // stride + 1

    # 출력 텐서 초기화 (출력 채널 크기를 입력 채널 크기와 동일하게 설정)
    output = np.zeros((n, out_h, out_w, c))

    # Depthwise Convolution 연산
    for i in range(out_h):
        for j in range(out_w):
            for k in range(c):  # 입력 채널에 대한 반복문
                output[:, i, j, k] = np.sum(
                    image[:, i*stride:i*stride+kh, j*stride:j*stride+kw, k] * depthwise_kernel[0, :, :, k], axis=(1, 2)
                )

    # 바이어스 추가
    output += bias

    return output



# ReLU6 활성화 함수 적용
def relu6(x):
    return np.minimum(np.maximum(0, x), 6)

# 모델과 이미지 불러오기
model_path = '/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/person_detection_model.tflite'
image_path = '/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/notPerson1.png'

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 입력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
all_ops = interpreter._get_ops_details()

# 이미지 불러오기 및 전처리
image = cv2.imread(image_path)
image = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32) / 255.0
imageBuffer = image.flatten()

for i in range(10):
    print(imageBuffer[i])

output_file_path = '/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/src/image.txt'  # 저장할 파일 경로를 설정하세요
input_y, input_x, input_ch = image.shape[1], image.shape[2], image.shape[3]  # 이미지 크기 및 채널 수
print(input_y, input_x, input_ch)

# 이미지를 1차원 배열로 변환


with open(output_file_path, 'w') as out_file:
    if out_file:
        for c in range(input_ch):
            out_file.write(f"Channel {c + 1}:\n")
            
            for y in range(input_y):
                for x in range(input_x):
                    index = (y * input_x + x) * input_ch + c
                    out_file.write(f"{int(imageBuffer[index])} ")  # 정수로 변환하여 저장
                out_file.write("\n")
            
            out_file.write("\n")
    else:
        print("Unable to open file for writing!")

print(f"Image saved to {output_file_path}")

# 첫 번째 연산자의 필터 및 바이어스 텐서 가져오기
first_op = all_ops[0]
conv_filter_index = first_op['inputs'][1]
conv_bias_index = first_op['inputs'][2] if len(first_op['inputs']) > 2 else None

conv_filter = interpreter.get_tensor(conv_filter_index)
conv_bias = interpreter.get_tensor(conv_bias_index) if conv_bias_index is not None else np.zeros(conv_filter.shape[0])

second_op = all_ops[1]
dep_conv_filter_index = second_op['inputs'][1]
dep_conv_bias_index = second_op['inputs'][2] if len(second_op['inputs']) > 2 else None

dep_conv_filter = interpreter.get_tensor(dep_conv_filter_index)
dep_conv_bias = interpreter.get_tensor(dep_conv_bias_index) if dep_conv_bias_index is not None else np.zeros(dep_conv_filter.shape[0])

# Convolution 및 ReLU6 연산 수행
conv_output = conv2d(image, conv_filter, conv_bias, stride=2, padding='same')
relu_output = relu6(conv_output)

print(f"relu shape is {relu_output.shape}")

dep_conv_output = depthwise_conv2d(conv_output,dep_conv_filter, dep_conv_bias, padding='same')
dep_relu_output = relu6(dep_conv_output)
print(f"relu shape is {dep_relu_output.shape}")



output_file_path = '/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/src/conv_relu_output2.txt'
with open(output_file_path, 'w') as f:
    channels = relu_output.shape[-1]
    height = relu_output.shape[1]
    width = relu_output.shape[2]
    
    for c in range(channels):
        f.write(f"Channel {c + 1}:\n")
        for y in range(height):
            values = " ".join([str(int(relu_output[0, y, x, c])) for x in range(width)])
            f.write(f"{values}\n")
        f.write("\n")

print(f"ReLU6 output saved to {output_file_path}")

