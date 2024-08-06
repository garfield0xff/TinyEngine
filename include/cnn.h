#ifndef GEN_CNN_H
#define GEN_CNN_H

#include "nn.h"


class CNN : public NeuralNetwork {
private:
    rgb7_t conv1_weights;
    rgb7_t conv2_weights;
    vector<double> fc_weights;
    double fc_bias;

public:
    CNN();
    ~CNN();
    vector<double> forward(const rgb8_t& _input);
    rgb31_t conv2d(const rgb8_t& _input, const rgb7_t& _kernel);
    rgb31_t relu(const rgb31_t& _input);
    rgb31_t max_pool(const rgb31_t& _input, q8_t _pool_size, q8_t _stride);
};


#endif //GEN_CNN_H