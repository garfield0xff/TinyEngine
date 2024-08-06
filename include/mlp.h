#ifndef GEN_MLP_H_
#define GEN_MLP_H_

#include "nn.h"
#include <vector>

using namespace std;

class MLP : public NeuralNetwork {

private:
    vector<q8_t> x_data;
    vector<q8_t> y_data;
    float learning_rate;
    float weights;
    float bias;
    

public:
    MLP(vector<q8_t> _x_data, vector<q8_t> _y_data, double _learning_rate)
        : x_data(_x_data), y_data(_y_data), learning_rate(_learning_rate), 
        weights(((float)rand() / RAND_MAX)), bias(((float)rand() / RAND_MAX)) {};

    double linearTransform(q8_t _x, float _weights, float _bias);
    double getError(q8_t _x, q8_t _y, float _weights, float _bias); 
    double gradientUpdate(q8_t idx, const string& grad_name);
    float sgdLoss();
    void startMLP(int epoch);
    
};


#endif //GEN_MLP_H_