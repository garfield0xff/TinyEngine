// #include <opencv2/opencv.hpp>
// #include <iostream>

// // 가정: depthwise_kernel3x3_stride1 함수가 다음과 같이 선언되어 있다고 가정합니다.
// int8_t* depthwise_kernel3x3_stride1 (
//     const uint16_t output_y, const uint16_t output_x,
//     const int32_t* bias, const int32_t* biasR, const int8_t* ksrc, const float* scales,
//     int8_t* output, int8_t* output_mask, const int mask_idx, const int32_t output_offset,
//     const int32_t act_min, const int32_t act_max,
//     int8_t* cols_8b_iterptr, const uint16_t column_x, int channel_offset
// );

// int main() {
//     // 1. OpenCV를 사용하여 이미지를 읽습니다.
//     cv::Mat img = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
//     if (img.empty()) {
//         std::cerr << "Could not read the image" << std::endl;
//         return 1;
//     }

//     // 2. 이미지 크기와 채널 정보 가져오기
//     uint16_t input_y = img.rows;
//     uint16_t input_x = img.cols;
//     uint16_t input_ch = 1; // 이미지를 회색조로 읽어들였으므로 채널은 1개

//     // 3. 출력 이미지 크기 설정 (stride = 1인 경우 출력 크기와 입력 크기는 동일)
//     uint16_t output_y = input_y;
//     uint16_t output_x = input_x;
//     uint16_t output_ch = input_ch;

//     // 4. 다른 파라미터 설정
//     int32_t bias[output_ch];         // 각 채널의 바이어스
//     int32_t biasR[output_ch];        // 각 채널의 추가 바이어스
//     int8_t ksrc[9 * output_ch];      // 3x3 커널, 각 채널에 대해 9개의 필터 값 필요
//     float scales[output_ch];         // 각 채널에 대한 스케일 값
//     int8_t output[output_y * output_x * output_ch]; // 출력 결과를 저장할 배열
//     int8_t output_mask[output_y * output_x * output_ch / 8] = {0}; // 출력 마스크를 저장할 배열
//     int32_t output_offset = 128;     // 출력 오프셋 (예: 양자화된 값의 오프셋)
//     int32_t act_min = -128;          // 활성화 최소값
//     int32_t act_max = 127;           // 활성화 최대값
//     uint16_t column_x = input_x + 2; // 패딩을 고려한 열 크기
//     int channel_offset = 1;          // 채널 간 오프셋 (단일 채널이므로 1)
//     int mask_idx = 0;                // 마스크 인덱스 (예: 채널의 비트 위치)

//     // 5. cols_8b_iterptr: 임시 버퍼로 사용될 메모리 블록 설정 (필요한 크기만큼 할당)
//     int8_t runtime_buf[(input_y + 2) * (input_x + 2)]; // 패딩을 포함한 입력 이미지 크기
//     int8_t* cols_8b_iterptr = runtime_buf;

//     // 6. 임시로 바이어스, 스케일, 커널 값을 초기화합니다.
//     for (int i = 0; i < output_ch; ++i) {
//         bias[i] = 0;            // 바이어스는 0으로 초기화
//         biasR[i] = 0;           // 추가 바이어스도 0으로 초기화
//         scales[i] = 1.0f;       // 스케일을 1로 설정
//         for (int j = 0; j < 9; ++j) {
//             ksrc[i * 9 + j] = 1;  // 커널 값을 1로 설정
//         }
//     }

//     // 7. depthwise_kernel3x3_stride1 함수 호출
//     int8_t* result = depthwise_kernel3x3_stride1(
//         output_y, output_x,
//         bias, biasR, ksrc, scales,
//         output, output_mask, mask_idx, output_offset,
//         act_min, act_max,
//         cols_8b_iterptr, column_x, channel_offset
//     );

//     // 8. 결과를 확인 (콘솔에 출력)
//     for (int y = 0; y < output_y; ++y) {
//         for (int x = 0; x < output_x; ++x) {
//             std::cout << static_cast<int>(output[y * output_x + x]) << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }
