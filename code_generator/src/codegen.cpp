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
                    auto padding = conv_options->padding();

                    op_built_operator += "_stride_";
                    op_built_operator += std::to_string(conv_options->stride_w());
                    op_built_operator += '_';
                    op_built_operator += std::to_string(conv_options->stride_h());
                    op_built_operator += "_pad_";
                    op_built_operator += (padding == tflite::Padding_SAME ? "same" : "valid");

                    genConv2D(op_built_operator, conv_options);

                    std::cout << op_built_operator << std::endl;
                    func_count++;
                } else {
                    cout << "conv2d 가 문제야 문제 " << endl;
                }
                break;
            
            case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
                if (op->builtin_options_type() == tflite::BuiltinOptions_DepthwiseConv2DOptions) {
                    const tflite::DepthwiseConv2DOptions* conv_options = op->builtin_options_as_DepthwiseConv2DOptions();
                    auto padding = conv_options->padding();

                    op_built_operator += "_stride_";
                    op_built_operator += std::to_string(conv_options->stride_w());
                    op_built_operator += '_';
                    op_built_operator += std::to_string(conv_options->stride_h());
                    op_built_operator += "_pad_";
                    op_built_operator += (padding == tflite::Padding_SAME ? "same" : "valid");

                    genDWConv2D(op_built_operator, conv_options);

                    std::cout << op_built_operator << std::endl;
                    func_count++;
                } else {
                    cout << "depth conv2d 가 문제야 문제 " << endl;
                }
                break;

            case tflite::BuiltinOperator_ADD:
                std::cout << op_built_operator << std::endl;
                genADD(op_built_operator);
                func_count++;                
                break;

            case tflite::BuiltinOperator_PAD:
                std::cout << op_built_operator << std::endl;
                genPAD(op_built_operator);
                func_count++;                
                break;

            case tflite::BuiltinOperator_MEAN:
                std::cout << op_built_operator << std::endl;
                genMean(op_built_operator);
                func_count++;                
                break;

            case tflite::BuiltinOperator_FULLY_CONNECTED:
                if (op->builtin_options_type() == tflite::BuiltinOptions_FullyConnectedOptions) {  
                    std::cout << op_built_operator << std::endl;
                    genFullyConnectedLayer(op_built_operator);
                    func_count++;
                } 
                break;

            case tflite::BuiltinOperator_LOGISTIC:
                std::cout << op_built_operator << std::endl;
                genLogisticRegression(op_built_operator);
                func_count++;                
                break;
            

            default:
                std::cout << "이거 채워야됌  : " << op_built_operator << std::endl;
                func_count++;
                break;
        }
    }
    std::cout << "all function : "  << subgraph->operators()->size() << std::endl;
    std::cout << "function call :  " <<  func_count << std::endl;
}

void CodeGenerator::genHeaders() 
{
    gen_code = \
        "#include <iostream>\n#include <vector>\n\n";
    
}


void CodeGenerator::genConv2D(const string& op_func, const tflite::Conv2DOptions* conv_options) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";

    ss << "}\n";
    gen_code += ss.str();
}

void CodeGenerator::genDWConv2D(const string& op_func, const tflite::DepthwiseConv2DOptions* conv_options) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code += ss.str();
}

void CodeGenerator::genADD(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code += ss.str();
}

void CodeGenerator::genPAD(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code += ss.str();
}

void CodeGenerator::genMean(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code += ss.str();
}

void CodeGenerator::genFullyConnectedLayer(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code += ss.str();
}

void CodeGenerator::genLogisticRegression(const string& op_func) 
{
    std::stringstream ss;
    ss << "void " << op_func << "() { \n";
    
    ss << "}\n";
    gen_code += ss.str();
}



void CodeGenerator::genCppModel(const string& output_path)
{
    try {
        filesystem::path dir = filesystem::path(output_path).parent_path();
        if(!filesystem::exists(dir)) {
            filesystem::create_directories(dir);
        }

        ofstream output_file(output_path);
        if (output_file.is_open()) {
            output_file << gen_code;
            output_file.close();
            cout << "Model successfully written to: " << output_path << endl;
        } else {
            cerr << "Unable to open output file: " << output_path << endl;
            cerr << "Error code: " << strerror(errno) << endl;
        }
    } catch (const filesystem::filesystem_error &e) {
        cerr << "File System error: " << e.what() << endl;
    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
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
