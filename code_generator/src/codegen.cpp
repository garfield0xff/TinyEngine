#include "codegen.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cerrno>
#include <cstring>

CodeGenerator::CodeGenerator() {}
CodeGenerator::~CodeGenerator() {}

void CodeGenerator::addLayer(const std::string& layerType, const std::vector<int>& params) {
    layers.push_back({ layerType, params });
}

void CodeGenerator::generateCode(const std::string& outputPath) {
    generate_code = "#include <iostream>\n#include <vector>\n\n";

    for (const auto& layer : layers) {
        if (layer.type == "Conv2D") {
            generateConv2D(layer);
        } else if (layer.type == "ReLU") {
            generateReLU(layer);
        } else if (layer.type == "MaxPooling") {
            generateMaxPooling(layer);
        } else if (layer.type == "Dense") {
            generateDense(layer);
        }
    }

    // Ensure the directory exists
    std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

    std::ofstream outFile(outputPath);
    if (outFile.is_open()) {
        outFile << generate_code;
        outFile.close();
    } else {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        std::cerr << "Error code: " << strerror(errno) << std::endl;
    }
}

void CodeGenerator::optimizePipeline() {
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        if (layers[i].type == "Conv2D" && layers[i + 1].type == "ReLU") {
            layers[i].type = "Conv2DReLU";
            layers.erase(layers.begin() + i + 1);
        }
    }
}

void CodeGenerator::generateConv2D(const Layer& layer) {
    std::stringstream ss;
    ss << "void conv2d_" << layer.params[0] << "() {\n";
    ss << "    // Conv2D implementation\n";
    ss << "}\n\n";
    generate_code += ss.str();
}

void CodeGenerator::generateReLU(const Layer& layer) {
    std::stringstream ss;
    ss << "void relu_" << layer.params[0] << "() {\n";
    ss << "    // ReLU implementation\n";
    ss << "}\n\n";
    generate_code += ss.str();
}

void CodeGenerator::generateMaxPooling(const Layer& layer) {
    std::stringstream ss;
    ss << "void maxpool_" << layer.params[0] << "() {\n";
    ss << "    // MaxPooling implementation\n";
    ss << "}\n\n";
    generate_code += ss.str();
}

void CodeGenerator::generateDense(const Layer& layer) {
    std::stringstream ss;
    ss << "void dense_" << layer.params[0] << "() {\n";
    ss << "    // Dense implementation\n";
    ss << "}\n\n";
    generate_code += ss.str();
}

void CodeGenerator::applyLoopUnrolling(std::string& code) {
    size_t pos = code.find("for (int i = 0; i < 4; ++i)");
    if (pos != std::string::npos) {
        code.replace(pos, 24, "for (int i = 0; i < 4; i += 2)");
    }
}
