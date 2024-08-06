#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

float relu(float x) {
    return x > 0 ? x : 0;
}

std::vector<std::vector<float>> convolution(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>>& filter, int stride = 1) {
    int input_size = input.size();
    int filter_size = filter.size();
    int output_size = (input_size - filter_size) / stride + 1;

    std::vector<std::vector<float>> output(output_size, std::vector<float>(output_size, 0));

    for (int i = 0; i < output_size; ++i) {
        for(int j = 0; j < output_size; ++j) {
            float sum = 0.0;
            for(int m = 0; m < filter_size; ++m) {
                for(int n = 0; n < filter_size; ++n) {
                    sum += input[i * stride + m][j * stride + n] * filter[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

std::vector<std::vector<float>> max_pooling(const std::vector<std::vector<float>>& input, int filter_size = 2, int stride = 2) {
    int input_size = input.size();
    int output_size = (input_size - filter_size) / stride + 1;
    
    std::vector<std::vector<float>> output(output_size, std::vector<float>(output_size, 0));

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float max_val = -INFINITY;
            for (int m = 0; m < filter_size; ++m) {
                for (int n = 0; n < filter_size; ++n) {
                    max_val = std::max(max_val, input[i * stride + m][j * stride + n]);
                }
            }
            output[i][j] = max_val;
        }
    }
    return output;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& input) {
    std::vector<float> output;
    for(const auto& row : input) {
        for (float val : row) {
            output.push_back(val);
        }
    }

    return output;
}


std::vector<float> fully_connected(const std::vector<float>&input , const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
    
}

