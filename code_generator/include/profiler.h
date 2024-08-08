#ifndef TENSOR_PROFILER_H
#define TENSOR_PROFILER_H

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>

using namespace tflite;

class TensorProfiler {

public:
    void printTensorInfo(const Tensor *tensor, const Model *model);
    
    void printOperatorInfo(const Operator *op, const Model *model, const SubGraph *subgraph);
    


};

#endif //TENSOR_PROFILER_H