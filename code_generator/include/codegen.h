#ifndef CODE_GENERATOR_H
#define CODE_GENERATOR_H

#include <string>
#include <vector>

using namespace std;

class CodeGenerator {
protected:
    struct Layer {
        string type;
        vector<int> params;
    };

    string generate_code;
    vector<Layer> layers;

public:
    CodeGenerator();
    ~CodeGenerator();
    

    void setHeaders();
    void addLayer(const string& layerType, const vector<int>& params);
    void generateCode(const string& outputPath);
    void optimizePipeline();
    
    void generateConv2D(const Layer& layer);
    void generateReLU(const Layer& layer);
    void generateMaxPooling(const Layer& layer);
    void generateDense(const Layer& layer);
    void applyLoopUnrolling(string& code);
    
    
};

#endif // CODE_GENERATOR_H