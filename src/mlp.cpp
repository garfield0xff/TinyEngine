#include "mlp.h"
#include <algorithm>
#include <ctime>
#include <random>
#include <iostream>

double MLP::linearTransform(q8_t _x, float _weights, float _bias) {
    return _weights * _x + _bias;
}

double MLP::getError(q8_t _x, q8_t _y, float _weights, float _bias) {
    return pow((linearTransform(_x, _weights, _bias) - _y), 2);
}

double MLP::gradientUpdate(q8_t idx, const string& grad_name) {
    float h = 1e-4;
    if (grad_name == "weights") {
        float forward = getError(x_data[idx], y_data[idx], weights + h, bias);
        float backward = getError(x_data[idx], y_data[idx], weights - h, bias);
        return (forward - backward) / (2 * h);
    } else if (grad_name == "bias") {
        float forward = getError(x_data[idx], y_data[idx], weights, bias + h);
        float backward = getError(x_data[idx], y_data[idx], weights, bias - h);
        return (forward - backward) / (2 * h);
    }
    return 0.0;
}

float MLP::sgdLoss() {
    vector<int> indices(x_data.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    shuffle(indices.begin(), indices.end(), std::default_random_engine(std::time(0)));
    for (int i : indices) {
        double grad_w = gradientUpdate(i, "weights");
        double grad_b = gradientUpdate(i, "bias");
        weights = weights - learning_rate * grad_w;
        bias = bias - learning_rate * grad_b;
    }
    return weights;
}


void MLP::startMLP(int epoch) {
    for (int i = 0; i < epoch; ++i) {
            cout << "epoch : " << i << "  weights : [" << weights << "]  bias : [" << bias << "]" << endl;
            weights = sgdLoss();
    }
    cout << "Updated weights : " << weights << "  Updated bias : " << bias << endl;
}

MLP::~MLP() {}
