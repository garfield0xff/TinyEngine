#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "engine.h"
#include <vector>

class NeuralNetwork : public TinyEngine {

public:
    
    double gradientUpdate();
    


    void activateFunction();
    void backPropogation();
    
    void batchNormalization();
};

#endif // NEURAL_NETWORK_H