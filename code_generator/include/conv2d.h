#ifndef CONV2D_H
#define CONV2D_H

#include "codegen.h"

class Conv2d : public CodeGenerator{
public:
    Conv2d();
    ~Conv2d();


    void generateConv2D(const Layer& layer);
    void generateReLU(const Layer& layer);
    void generateMaxPooling(const Layer& layer);
    void generateDense(const Layer& layer);
    void applyLoopUnrolling(string& code);

};

#endif