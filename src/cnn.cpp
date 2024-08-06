#include "cnn.h"



rgb31_t CNN::conv2d(const rgb8_t& _input, const rgb7_t& _kernel) {
    q8_t num_channels = _input.size();
    q8_t kernel_size = _kernel[0][0].size();
    q8_t pad = kernel_size / 2;
    q16_t input_size = _input[0].size();
    q16_t output_size = input_size - kernel_size + 1;

    rgb31_t output(1, vector<vector<q31_t>>(output_size, vector<q31_t>(output_size, 0)));

    for(int i = 0; i < output_size; ++i)
    {
        for(int j = 0; j < output_size; ++j) {
            q31_t sum = 0;
            for(int c = 0; c < num_channels; ++c) {
                for(int ki = 0; ki < kernel_size; ++ki) {
                    for(int kj = 0; kj < kernel_size; ++kj) {
                        sum += _input[c][i + ki][j + kj] * _kernel[c][ki][kj];
                    }
                }
            }
            output[0][i][j] = sum;
        }
    }

    return output;
};

rgb31_t CNN::relu(const rgb31_t& _input) {
    rgb31_t output = _input;
    for(auto& channel : output) {
        for(auto& row : channel) {
            for(auto& val : row) {
                val = max(static_cast<q31_t>(0), val);
            }
        } 
    }
    return output;
};

rgb31_t CNN::max_pool(const rgb31_t& _input, q8_t _pool_size, q8_t _stride) {
    int input_size = _input[0].size();
    int output_size = (input_size - _pool_size) / _stride + 1;

    rgb31_t output(1, vector<vector<q31_t>>(output_size, vector<q31_t>(output_size, 0)));

    for(int i = 0; i < output_size; ++i) {
        for(int j = 0; j < output_size; ++j) {
            q31_t max_val = numeric_limits<q31_t>::min();
            for(int pi = 0; pi < _pool_size; ++pi) {
                for(int pj = 0; pj < _pool_size; ++pj) {
                    int row = i * _stride + pi;
                    int col = j * _stride + pj;
                    max_val = max(max_val, _input[0][row][col]);
                }
            }
            output[0][i][j] = max_val;
        }
    }

    return output;
};

vector<double> CNN::forward(const rgb8_t& input) {
    rgb31_t conv1_output = conv2d(input, conv1_weights);
    rgb31_t relu1_output = relu(conv1_output);
    rgb31_t pool1_output = max_pool(relu1_output, 2, 2);

    rgb31_t conv2_output = conv2d(input, conv2_weights);
    rgb31_t relu2_output = relu(conv2_output);
    rgb31_t pool2_output = max_pool(relu2_output, 2, 2);

    vector<q32_t> flat;
    for(const auto& row: pool2_output[0]) {
        for(const auto& val : row) {
            flat.push_back(static_cast<q31_t>(val));
        }
    }

    float output = 0.0;
    for (size_t i = 0; i < flat.size(); ++i) {
        output += flat[i] * fc_weights[i];   
    }
    output += fc_bias;

    return {output};
}

