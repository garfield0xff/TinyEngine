#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "engine.h"
#include <vector>

using namespace std;

class NeuralNetwork : public TinyEngine {
public:



    // NeuralNetwork();
    // ~NeuralNetwork();

    double gradientUpdate();
    void activateFunction();
    void backPropogation();
    void batchNormalization();
};

#endif // NEURAL_NETWORK_H