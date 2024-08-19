#include "validator.h"


void Validator::conv2d(
    float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, size_t kernelSize, int kernel_height, int kernel_width,
    const float *bias, 
    const float output_activation_min,
    const float output_activation_max, 
    float *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    uint8_t stride_value, int layer_index, bool relu_flag, bool pad_is_same
)
{
    int pad_h = 0, pad_w = 0;
    if (pad_is_same) {
        pad_h = (kernel_height - 1) / 2;
        pad_w = (kernel_width - 1) / 2;
    }

    int padded_x = input_x + 2 * pad_w;
    int padded_y = input_y + 2 * pad_h;
    std::vector<float> padded_input(padded_x * padded_y * input_ch, 0.0f);

    if (pad_is_same) {
        for (int c = 0; c < input_ch; ++c) {
            for (int y = 0; y < input_y; ++y) {
                for (int x = 0; x < input_x; ++x) {
                    int input_index = (y * input_x + x) * input_ch + c;
                    int padded_index = ((y + pad_h) * padded_x + (x + pad_w)) * input_ch + c;
                    padded_input[padded_index] = input[input_index];
                }
            }
        }
    } else {
        padded_input.assign(input, input + input_x * input_y * input_ch);
    }

    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    
    int batch_size = 1;
    int flag = 0;

    int kernel_size = kernel_width * kernel_height * input_ch; 

    for (int n = 0; n < batch_size; ++n) { 
        for (int oc = 0; oc < output_channels; ++oc) { 
            for (int oy = 0; oy < output_height; ++oy) { 
                for (int ox = 0; ox < output_width; ++ox) {
                    float acc = 0.0f;

                    for (int ic = 0; ic < input_ch; ++ic) { 
                        for (int ky = 0; ky < kernel_height; ++ky) {
                            for (int kx = 0; kx < kernel_width; ++kx) {
                                int ix = ox * stride_value + kx;
                                int iy = oy * stride_value + ky;

                                if (ix >= 0 && ix < padded_x && iy >= 0 && iy < padded_y) {
                                    int input_index = (n * padded_y + iy) * padded_x * input_ch + ix * input_ch + ic;
                                    int flat_kernel_index = oc * kernel_size + ky * kernel_width * input_ch + kx * input_ch + ic;
                                    acc += padded_input[input_index] * kernel[flat_kernel_index];
                                }
                            }
                        }
                    }

                    acc += bias[oc];

                    if (relu_flag) {
                        acc = std::max(output_activation_min, std::min(acc, output_activation_max));
                    }

                    int output_index = ((n * output_height + oy) * output_width + ox) * output_channels + oc;
                    output[output_index] = acc;
                }
            }
        }
    }

    std::ofstream out_file(layer_path + "outputConv2d" + std::to_string(layer_index) + ".txt");
    if (out_file.is_open()) {
        for (int oc = 0; oc < output_ch; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_y; ++oy) {
                for (int ox = 0; ox < output_x; ++ox) {
                    int output_index = (oy * output_x + ox) * output_ch + oc;
                    out_file <<  static_cast<int>(output[output_index]) << " ";
                }
                out_file << std::endl;
            }
            out_file << std::endl;
        }
        out_file.close();  
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}

void Validator::depthwiseConv2d(
    float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, size_t kernelSize, int kernel_height, int kernel_width,
    const float *bias,
    const float output_activation_min,
    const float output_activation_max, 
    float *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    uint8_t stride_value, int layer_index, bool relu_flag, bool pad_is_same
)
{
    int pad_h = 0, pad_w = 0;
    if (pad_is_same) {
        pad_h = (kernel_height - 1) / 2;
        pad_w = (kernel_width - 1) / 2;
    }

    // If padding is 'same', create a padded input tensor
    int padded_x = input_x + 2 * pad_w;
    int padded_y = input_y + 2 * pad_h;
    std::vector<float> padded_input(padded_x * padded_y * input_ch, 0.0f);

    if (pad_is_same) {
        for (int c = 0; c < input_ch; ++c) {
            for (int y = 0; y < input_y; ++y) {
                for (int x = 0; x < input_x; ++x) {
                    int input_index = (c * input_y * input_x) + (y * input_x + x);
                    int padded_index = (c * padded_y * padded_x) + ((y + pad_h) * padded_x + (x + pad_w));
                    padded_input[padded_index] = input[input_index];
                }
            }
        }
    } else {
        // No padding, use original input
        padded_input.assign(input, input + input_x * input_y * input_ch);
    }

    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    int flag = 0;

    for (int oc = 0; oc < output_channels; ++oc) { 
        for (int oy = 0; oy < output_height; ++oy) { 
            for (int ox = 0; ox < output_width; ++ox) {
                float acc = 0.0f;

                for (int ky = 0; ky < kernel_height; ++ky) {
                    for (int kx = 0; kx < kernel_width; ++kx) {
                        int ix = ox * stride_value + kx;
                        int iy = oy * stride_value + ky;

                        if (ix >= 0 && ix < padded_x && iy >= 0 && iy < padded_y) {
                            int input_index = (oc * padded_y * padded_x) + (iy * padded_x + ix);
                            int flat_kernel_index = ky * kernel_width + kx + oc * kernel_height * kernel_width;
                            acc += padded_input[input_index] * kernel[flat_kernel_index];

                        }
                    }
                }

                acc += bias[oc];

                if (relu_flag) {
                    acc = std::max(output_activation_min, std::min(acc, output_activation_max));
                }

                int output_index = (oy * output_width + ox) * output_channels + oc;
                output[output_index] = acc;
            }
        }
    }

    std::ofstream out_file(layer_path + "depthWiseConv2d_" + std::to_string(layer_index) + ".txt");
    if (out_file.is_open()) {
        for (int oc = 0; oc < output_ch; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_y; ++oy) {
                for (int ox = 0; ox < output_x; ++ox) {
                    int output_index = (oy * output_x + ox) * output_ch + oc;
                    out_file <<  static_cast<int>(output[output_index]) << " ";
                }
                out_file << std::endl;
            }
            out_file << std::endl;
        }
        out_file.close();  
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}


void Validator::pad2d(
    float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const int32_t* paddings,  
    float *output,  uint16_t output_x, uint16_t output_y, uint16_t output_ch, 
    int8_t pad_value, int layer_index
)
{
    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    
    float *input_ptr = input;

    for (int c = 0; c < input_ch; ++c) {
        for (int oy = 0; oy < output_height; ++oy) {
            for (int ox = 0; ox < output_width; ++ox) {
                int output_index = (oy * output_width + ox) * output_channels + c;
                
                if (oy < paddings[2] || oy >= (output_height - paddings[3]) ||
                    ox < paddings[4] || ox >= (output_width - paddings[5])) {
                    output[output_index] = static_cast<int>(pad_value);
                } else {
                    int input_index = ((oy - paddings[2]) * input_x + (ox - paddings[4])) * input_ch + c;
                    output[output_index] = input[input_index];
                }
            }
        }
    }

    std::cout << "padding x size is : " << output_x <<  std::endl;
    std::cout << "padding y size is : " << output_y <<  std::endl;


    std::cout << "padding ouptut indexing..." << std::endl;
    for(int j = 0; j < 10; j++)
    {
        std::cout << output[j] << " ";
    }
    std::cout << std::endl;

    std::ofstream out_file(layer_path + "pad_" + std::to_string(layer_index) + ".txt");
    if (out_file.is_open()) {
        for (int oc = 0; oc < output_ch; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_height; ++oy) {
                for (int ox = 0; ox < output_width; ++ox) {
                    int output_index = (oy * output_width + ox) * output_channels + oc;
                    out_file << static_cast<int>(output[output_index]) << " ";
                }
                out_file << std::endl;
            }
            out_file << std::endl;
        }
        out_file.close(); 
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}

void Validator::add(
    float* input_tensor1, float* input_tensor2, 
    const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    float* output, int layer_index
)
{
    for(size_t i = 0; i < input_x * input_y * input_ch; ++i) {
        output[i] = input_tensor1[i] + input_tensor2[i];
    }

    std::ofstream out_file(layer_path + "add_" + std::to_string(layer_index) + ".txt");
    if(out_file.is_open()) {
        for(int oc = 0; oc < input_ch; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for(int oy = 0; oy < input_y; ++oy) {
                for(int ox = 0; ox < input_x ; ++ox) {
                    int output_index = ( oy * input_x + ox ) * input_ch + oc;
                    out_file << static_cast<int>(output[output_index]) << " ";
                }
                out_file << std::endl;
            }
            out_file << std::endl;
        }
        out_file.close();
    }
}

