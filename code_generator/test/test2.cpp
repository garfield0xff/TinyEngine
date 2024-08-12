#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <cstring>  // memset 함수를 사용하기 위해 필요
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <fstream>
#include <ostream>

#define BIT_SET(a, b) ((a) |= (1ULL << (b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1ULL << (b)))
#define BIT_FLIP(a, b) ((a) ^= (1ULL << (b)))
#define BIT_CHECK(a, b) (!!((a) & (1ULL << (b))))  // '!!' to make sure this returns 0 or 1

#define BITMASK_SET(x, mask) ((x) |= (mask))
#define BITMASK_CLEAR(x, mask) ((x) &= (~(mask)))
#define BITMASK_FLIP(x, mask) ((x) ^= (mask))
#define BITMASK_CHECK_ALL(x, mask) (!(~(x) & (mask)))
#define BITMASK_CHECK_ANY(x, mask) ((x) & (mask))

int8_t *depthwise_kernel3x3_stride1 (
    const uint16_t output_y, const uint16_t output_x,
    const int32_t *bias, const int32_t *biasR, const int8_t *ksrc, const float *scales,
    int8_t *output, int8_t *output_mask, const int mask_idx, const int32_t output_offset,
    const int32_t act_min, const int32_t act_max,
    int8_t *cols_8b_iterptr, const uint16_t column_x, int channel_offset
)
{
    #define STRIDE 2
    int i, j;
    int8_t mask_value;

    for (i = 0; i < output_y; i++) {
        for(j = 0; j < output_x / 2; j++) {
            int8_t *cols_8b = cols_8b_iterptr;

            int32_t sum0 = bias[0] + biasR[0];
            int32_t sum1 = bias[0] + biasR[0];

            sum0 += cols_8b[0] * ksrc[0];
            sum0 += cols_8b[1] * ksrc[1];
            sum0 += cols_8b[2] * ksrc[2];
            sum1 += cols_8b[1] * ksrc[0];
            sum1 += cols_8b[2] * ksrc[1];
            sum1 += cols_8b[3] * ksrc[2];

            cols_8b += column_x + 2;

            sum0 += cols_8b[0] * ksrc[3];
            sum0 += cols_8b[1] * ksrc[4];
            sum0 += cols_8b[2] * ksrc[5];
            sum1 += cols_8b[1] * ksrc[3];
            sum1 += cols_8b[2] * ksrc[4];
            sum1 += cols_8b[3] * ksrc[5];

            cols_8b += column_x + 2;

            sum0 += cols_8b[0] * ksrc[6];
            sum0 += cols_8b[1] * ksrc[7];
            sum0 += cols_8b[2] * ksrc[8];
            sum1 += cols_8b[1] * ksrc[6];
            sum1 += cols_8b[2] * ksrc[7];
            sum1 += cols_8b[3] * ksrc[8];

            sum0 = (float)sum0 * *scales;
            sum0 += output_offset;
            mask_value = 1;
            if (sum0 < act_min) {
                sum0 = act_min;
                mask_value = 0;
            } 
            if(sum0 > act_max) {
                sum0 = act_max;
                mask_value = 0;
            }
            output[(i * output_x + j * 2) * channel_offset] = sum0;
            if(mask_value == 1) 
            {
                BIT_SET(output_mask[(i * output_x + j * 2) * channel_offset / 8], mask_idx);
            }
            else {
                BIT_CLEAR(output_mask[(i * output_x + j * 2) * channel_offset / 8], mask_idx);
            }

            sum1 = (float)sum1 * *scales;
            sum1 += output_offset;
            mask_value = 1;
            if (sum1 < act_min) {
                sum1 = act_min;
                mask_value = 0;
            } 
            if(sum1 > act_max) {
                sum1 = act_max;
                mask_value = 0;
            }
            output[(i * output_x + (j * 2 + 1)) * channel_offset] = sum1;
            if(mask_value == 1) 
            {
                BIT_SET(output_mask[(i * output_x + (j * 2 + 1)) * channel_offset / 8], mask_idx);
            }
            else {
                BIT_CLEAR(output_mask[(i * output_x + (j * 2 + 1)) * channel_offset / 8], mask_idx);
            }

            cols_8b_iterptr += STRIDE * 2;
        }

        // output_x가 홀 수 일 경우 마지막 연산
        if (output_x & 1)
        {
            int8_t *cols_8b = cols_8b_iterptr;
            int32_t sum = bias[0] + biasR[0];
            sum += cols_8b[0] * ksrc[0];
            sum += cols_8b[1] * ksrc[1];
            sum += cols_8b[2] * ksrc[2];
            cols_8b += column_x + 2;
            sum += cols_8b[0] * ksrc[3];
            sum += cols_8b[1] * ksrc[4];
            sum += cols_8b[2] * ksrc[5];
            cols_8b += column_x + 2;
            sum += cols_8b[0] * ksrc[6];
            sum += cols_8b[1] * ksrc[7];
            sum += cols_8b[2] * ksrc[8];

            sum = (float)sum * *scales;
            sum += output_offset;
            mask_value = 1;
            if(sum < act_min) {
                sum = act_min;
                mask_value = 0;
            }
            if(sum > act_max) {
                sum = act_max;
                mask_value = 0;
            }

            output[(i * output_x + output_x - 1) * channel_offset] = sum;
            if (mask_value == 1)
                BIT_SET(output_mask[(i * output_x + output_x - 1) * channel_offset / 8], mask_idx);
            else
                BIT_CLEAR(output_mask[(i * output_x + output_x - 1) * channel_offset / 8], mask_idx);

            cols_8b_iterptr += STRIDE;
        }

        cols_8b_iterptr += 1 * 2;
    }
    return output_mask;
}

int main()
{
    cv::Mat img = cv::imread("/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/test/person1.png", cv::IMREAD_GRAYSCALE);
    if(img.empty()) {
        std::cerr << "Could not read the image" << std::endl;
        return 1;
    }





    // 2. 이미지 크기와 채널 정보 가져오기
    uint16_t input_y = img.rows;
    uint16_t input_x = img.cols;
    uint16_t input_ch = 1; // 이미지를 회색조로 읽어들였으므로 채널은 1개

    // 3. 출력 이미지 크기 설정 (stride = 1인 경우 출력 크기와 입력 크기는 동일)
    uint16_t output_y = input_y;
    uint16_t output_x = input_x;
    uint16_t output_ch = input_ch;

    int32_t bias[output_ch];         // 각 채널의 바이어스
    int32_t biasR[output_ch];        // 각 채널의 추가 바이어스
    int8_t ksrc[9 * output_ch];      // 3x3 커널, 각 채널에 대해 9개의 필터 값 필요
    float scales[output_ch];         // 각 채널에 대한 스케일 값
    int8_t output[output_y * output_x * output_ch]; // 출력 결과를 저장할 배열
    int8_t output_mask[output_y * output_x * output_ch / 8];
    memset(output_mask, 0, sizeof(output_mask));
    int32_t output_offset = 128;     // 출력 오프셋 (예: 양자화된 값의 오프셋)
    int32_t act_min = 0;          // 활성화 최소값
    int32_t act_max = 6;           // 활성화 최대값
    uint16_t column_x = input_x + 2; // 패딩을 고려한 열 크기
    int channel_offset = 1;          // 채널 간 오프셋 (단일 채널이므로 1)
    int mask_idx = 0;                // 마스크 인덱스 (예: 채널의 비트 위치)

    int8_t runtime_buf[(input_y + 2) * (input_x + 2)];
    int8_t* cols_8b_iterptr = runtime_buf;

    // 6. 임시로 바이어스, 스케일, 커널 값을 초기화합니다.
    for (int i = 0; i < output_ch; ++i) {
        bias[i] = 0;            // 바이어스는 0으로 초기화
        biasR[i] = 0;           // 추가 바이어스도 0으로 초기화
        scales[i] = 1.0f;       // 스케일을 1로 설정
        for (int j = 0; j < 9; ++j) {
            ksrc[i * 9 + j] = 1;  // 커널 값을 1로 설정
        }
    }

    int8_t* result = depthwise_kernel3x3_stride1(
        output_y, output_x,
        bias, biasR, ksrc, scales,
        output, output_mask, mask_idx, output_offset,
        act_min, act_max,
        cols_8b_iterptr, column_x, channel_offset
    );

    cv::Mat output_image(output_y, output_x, CV_8SC1, output);
    cv::Mat output_image_normalized;
    cv::normalize(output_image, output_image_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("Input image", img);
    cv::imshow("Ouptut image", output_image_normalized);
    cv::waitKey(0);

    // // 8. 결과를 확인 (콘솔에 출력)
    // for (int y = 0; y < output_y; ++y) {
    //     for (int x = 0; x < output_x; ++x) {
    //         std::cout << static_cast<int>(output[y * output_x + x]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
