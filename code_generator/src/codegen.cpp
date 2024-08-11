#include "codegen.h"
#include <fstream>
#include <iostream>
// #include <sstream>
// #include <filesystem>
// #include <cerrno>
#include <cstring>

CodeGenerator::CodeGenerator() {}
CodeGenerator::~CodeGenerator() {}

void CodeGenerator::parseModel(const tflite::Model* tf_model)
{
    if(tf_model->subgraphs()->size() == 0) {
        std::cerr << "Model has no subgraphs" << "\n";
        return;
    }

    set<string> op_headers_set;

    // generate Headers
    genHeaders();

    const tflite::SubGraph* subgraph = tf_model->subgraphs()->Get(0);
    std::cout << "Number of Tensors: " << subgraph->tensors()->size() << "\n";
    std::cout << "Number of Operaotrs: " << subgraph->operators()->size() << "\n";

    int func_count = 0;

    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        auto op = subgraph->operators()->Get(i);
        auto opcode_index = op->opcode_index();
        auto opcode = tf_model->operator_codes()->Get(opcode_index)->builtin_code();

        string op_built_operator;
        
        if(const char* op_name = tflite::EnumNameBuiltinOperator(opcode)) {
            op_built_operator = op_name;

            std::transform(op_built_operator.begin(), op_built_operator.end(), op_built_operator.begin(),
                [](unsigned char c){ return std::tolower(c); });

        }
        
        switch (opcode) {
            case tflite::BuiltinOperator_CONV_2D:
                if (op->builtin_options_type() == tflite::BuiltinOptions_Conv2DOptions) {
                    const tflite::Conv2DOptions* conv_options = op->builtin_options_as_Conv2DOptions();

                    // get weights tensor shape
                    int weights_tensor_index = op->inputs()->Get(1);  // 첫 번째 입력 텐서가 가중치 텐서
                    const tflite::Tensor* weights_tensor = subgraph->tensors()->Get(weights_tensor_index);
                    auto weights_shape = weights_tensor->shape();

                    int kernel_h = weights_shape->Get(1);
                    int kernel_w = weights_shape->Get(2);

                    auto padding = conv_options->padding();

                    // Constructing the operator string with kernel size and stride information
                    op_built_operator += "_kernel" + std::to_string(kernel_h) + "x" + std::to_string(kernel_w);  
                    op_built_operator += "_stride";
                    op_built_operator += std::to_string(conv_options->stride_w());

                    
                    if(op_headers_set.find(op_built_operator) == op_headers_set.end()) {
                        op_headers_set.insert(op_built_operator);
                        genConv2D(op_built_operator, conv_options);    
                        genCppModel(op_built_operator, false);      
                        gen_code.str("");             
                    }
            
                    func_count++;
                } else {
                    std::cout << "conv2d 가 문제야 문제 " << std::endl;
                }
                break;

            
            case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
                if (op->builtin_options_type() == tflite::BuiltinOptions_DepthwiseConv2DOptions) {
                    const tflite::DepthwiseConv2DOptions* conv_options = op->builtin_options_as_DepthwiseConv2DOptions();
        
                    // get weights tensor shape
                    int weights_tensor_index = op->inputs()->Get(1);  // 첫 번째 입력 텐서가 가중치 텐서
                    const tflite::Tensor* weights_tensor = subgraph->tensors()->Get(weights_tensor_index);
                    auto weights_shape = weights_tensor->shape();

                    int kernel_h = weights_shape->Get(1);
                    int kernel_w = weights_shape->Get(2);

                    auto padding = conv_options->padding();

                    // Constructing the operator string with kernel size and stride information
                    op_built_operator += "_kernel" + std::to_string(kernel_h) + "x" + std::to_string(kernel_w);  
                    op_built_operator += "_stride";
                    op_built_operator += std::to_string(conv_options->stride_w());

                    if(op_headers_set.find(op_built_operator) == op_headers_set.end()) {
                        op_headers_set.insert(op_built_operator);
                        genDWConv2D(op_built_operator, conv_options);                        
                        genCppModel(op_built_operator, false);
                        gen_code.str("");
                    }

                    // std::cout << op_built_operator << std::endl;
                    func_count++;
                } else {
                    cout << "depth conv2d 가 문제야 문제 " << endl;
                }
                break;

            case tflite::BuiltinOperator_ADD:
                // std::cout << op_built_operator << std::endl;
                genADD(op_built_operator);
                gen_code.str("");
                func_count++;                
                break;

            case tflite::BuiltinOperator_PAD:
                // std::cout << op_built_operator << std::endl;
                genPAD(op_built_operator);
                gen_code.str("");
                func_count++;                
                break;

            case tflite::BuiltinOperator_MEAN:
                // std::cout << op_built_operator << std::endl;
                genMean(op_built_operator);
                gen_code.str("");
                func_count++;                
                break;

            case tflite::BuiltinOperator_FULLY_CONNECTED:
                if (op->builtin_options_type() == tflite::BuiltinOptions_FullyConnectedOptions) {  
                    // std::cout << op_built_operator << std::endl;
                    genFullyConnectedLayer(op_built_operator);
                    gen_code.str("");
                    func_count++;
                } 
                break;

            case tflite::BuiltinOperator_LOGISTIC:
                // std::cout << op_built_operator << std::endl;
                genLogisticRegression(op_built_operator);
                gen_code.str("");
                func_count++;                
                break;
            

            default:
                // std::cout << "이거 채워야됌  : " << op_built_operator << std::endl;
                func_count++;
                break;
        }
    }

    genHeaderFile(op_headers_set);
    genCppModel("gen_code", true);

    std::cout << "all function : "  << subgraph->operators()->size() << std::endl;
    std::cout << "function call :  " <<  func_count << std::endl;
}

void CodeGenerator::genHeaders() 
{
    gen_code << "#include <iostream>\n#include <vector>\n\n";
    
}

void CodeGenerator::genHeaderFile(set<string> op_headers_set) {
    gen_code << "#ifndef GEN_CODE_H\n#define GEN_CODE_H\n\n";

    for(const std::string& header: op_headers_set) {
        std::cout << header << std::endl;
        gen_code << "void " << header << "()" << "{\n";
        gen_code << "}\n\n";
    }
    
    gen_code << "#endif";
    
};


void CodeGenerator::genConv2D(const string& op_func, const tflite::Conv2DOptions* conv_options) 
{
    std::stringstream ss;
    string params = "char str_w, char str_h, char pad_w, char pad_h";
    
    ss << "void " << op_func << "(" << params  << ")  { \n";

    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::genDWConv2D(const string& op_func, const tflite::DepthwiseConv2DOptions* conv_options) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::genADD(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::genPAD(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::genMean(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::genFullyConnectedLayer(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::genLogisticRegression(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code << ss.str();
}

void CodeGenerator::setCodeGenPath(const string& _codegen_path) 
{
    cout << "setCodeGenPath" << endl;
    codegen_path = _codegen_path;
    try {
        std::filesystem::path dir = std::filesystem::path(codegen_path).parent_path();
        if (std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir / "src");
            std::filesystem::create_directories(dir / "include");
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
}

void CodeGenerator::genCppModel(const string& file_name, bool is_header)
{
    cout << "genCppModel" << endl;
    string output_path;

    if(is_header) output_path = codegen_path + "include/" + file_name + ".h";
    else output_path = codegen_path + "src/" + file_name + ".cpp";
    
    cout << "output path is : " << output_path << endl;

    ofstream output_file(output_path, std::ios::out | std::ios::trunc);

    if (output_file.is_open()) {
        output_file << gen_code.str();
        output_file.close();
        cout << "Model successfully written to: " << output_path << endl;
    } else {
        cerr << "Unable to open output file: " << output_path << endl;
        cerr << "Error code: " << strerror(errno) << endl;
    }

}





// void CodeGenerator::addLayer(const std::string& layerType, const std::vector<int>& params) {
//     layers.push_back({ layerType, params });
// }

// void CodeGenerator::generateCode(const std::string& outputPath) {
//     generate_code = \
//     "#include <iostream>\n#include <vector>\n\n";

//     for (const auto& layer : layers) {
//         if (layer.type == "Conv2D") {
//             generateConv2D(layer);
//         } else if (layer.type == "ReLU") {
//             generateReLU(layer);
//         } else if (layer.type == "MaxPooling") {
//             generateMaxPooling(layer);
//         } else if (layer.type == "Dense") {
//             generateDense(layer);
//         }
//     }

//     // Ensure the directory exists
//     std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

//     std::ofstream outFile(outputPath);
//     if (outFile.is_open()) {
//         outFile << generate_code;
//         outFile.close();
//     } else {
//         std::cerr << "Unable to open output file: " << outputPath << std::endl;
//         std::cerr << "Error code: " << strerror(errno) << std::endl;
//     }
// }

// void CodeGenerator::optimizePipeline() {
//     for (size_t i = 0; i < layers.size() - 1; ++i) {
//         if (layers[i].type == "Conv2D" && layers[i + 1].type == "ReLU") {
//             layers[i].type = "Conv2DReLU";
//             layers.erase(layers.begin() + i + 1);
//         }
//     }
// }

// void CodeGenerator::generateConv2D(const Layer& layer) {
//     std::stringstream ss;
//     ss << "void conv2d_" << layer.params[0] << "() {\n";
//     ss << "    // Conv2D implementation\n";
//     ss << "}\n\n";
//     generate_code += ss.str();
// }

// void CodeGenerator::generateReLU(const Layer& layer) {
//     std::stringstream ss;
//     ss << "void relu_" << layer.params[0] << "() {\n";
//     ss << "    // ReLU implementation\n";
//     ss << "}\n\n";
//     generate_code += ss.str();
// }

// void CodeGenerator::generateMaxPooling(const Layer& layer) {
//     std::stringstream ss;
//     ss << "void maxpool_" << layer.params[0] << "() {\n";
//     ss << "    // MaxPooling implementation\n";
//     ss << "}\n\n";
//     generate_code += ss.str();
// }

// void CodeGenerator::generateDense(const Layer& layer) {
//     std::stringstream ss;
//     ss << "void dense_" << layer.params[0] << "() {\n";
//     ss << "    // Dense implementation\n";
//     ss << "}\n\n";
//     generate_code += ss.str();
// }

// void CodeGenerator::applyLoopUnrolling(std::string& code) {
//     size_t pos = code.find("for (int i = 0; i < 4; ++i)");
//     if (pos != std::string::npos) {
//         code.replace(pos, 24, "for (int i = 0; i < 4; i += 2)");
//     }
// }
