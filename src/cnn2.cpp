#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdint.h>

class NeuralNetwork {
    public:
        int layers;
        int units_per_layer;
        float accuracy;
        size_t max_memory_usage;
        double latency;

        NeuralNetwork() : layers(0), units_per_layer(0), accuracy(0.0f), max_memory_usage(0), latency(0.0) {}

        NeuralNetwork(int layers, int units_per_layer) : layers(layers), units_per_layer(units_per_layer), accuracy(0.0f), max_memory_usage(0), latency(0.0) {}

        void evaluate() {
            accuracy = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

            max_memory_usage = (rand() % 1024 + 1) * 1024 * 1024;
            latency = static_cast<double>(rand() % 100 + 1);
        }

        void print() const {
            std::cout << "Model - Layers:" << layers
                      << ", Units per Layer: " << units_per_layer
                      << ", Accuracy: " << accuracy
                      << ", Max memory Usage: " << max_memory_usage / ( 1024 * 1024 ) << "MB"
                      << ", Latency: " << latency << " ms" << std::endl;
        }

};

class Profiler {
public:
    void profile(NeuralNetwork& model) {
        model.evaluate();
        if(model.max_memory_usage > 1024 * 1024* 1024 ) {
            std::cerr << "Error : Model exceeds max memory limits of 1GB" << std::endl;
            model.accuracy = 0.0f;
        }
    }
};

class NAS {
public:
    std::vector<NeuralNetwork> population;

    NAS(int population_size){
        for (int i = 0; i < population_size; ++i) {
            int layers = rand() % 10 + 1;
            int units = rand() % 100 + 1;
            population.emplace_back(layers, units);
        }
    }

    void search(int iterations, Profiler& profiler) {
        for(int i = 0; i < iterations; ++i) {
            for(auto& model : population) {
                profiler.profile(model);
            }

            //정확도를 기준으로 모델 정렬
            std::sort(population.begin(), population.end(),[](const NeuralNetwork& a, const NeuralNetwork& b) {
                return a.accuracy > b.accuracy;
            });

            // 상위 50% 모델 유지, 하위 50% 모델 제거 및 새로운 모델 생성
            size_t current_size = population.size();
            population.resize(current_size / 2);
            for(size_t j = 0; j < current_size / 2; ++j) {
                int layers = rand() % 10 + 1;
                int units = rand() % 100 + 1;
                population.emplace_back(layers, units);
            }

        }
    }

    void print_best_model() const {
        if(!population.empty()) {
            const auto& best_model = population.front();
            std::cout << "Bestk Model - ";
            best_model.print();
        }
    }
};


// 메모리 사용량을 측정하는 함수
size_t getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // 최대 RSS (Resident Set Size) 반환
}

void im2col(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output, int kernel_size, int stride) {
    int input_height = input.size();
    int input_width = input[0].size();
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;

    for(int y =0; y < output_height; ++y) {
        for(int x = 0; x < output_width; ++x) {
            std::vector<float> patch;
            for(int ky = 0; ky < kernel_size; ++ky) {
                for(int kx = 0; kx < kernel_size; ++kx) {
                    patch.push_back(input[y * stride + ky][x * stride + kx]);
                }
            }
            output.push_back(patch);
        }
    }
}

void fusedBatchNormReLU(std::vector<float>& input, float mean, float variance, float epsilon, float gamma, float beta) {
    for (auto& x : input) {

        //Batch Norm
        x = gamma * ( x - mean ) / std::sqrt(variance + epsilon) + beta;

        //ReLU
        x = std::max(0.0f, x);
    }
}


void addVectors(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result) {
    size_t n = a.size();
    size_t i = 0;

    for(; i + 4 <= n; i +=4 ) {
        result[i] = a[i] + b[i];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
    }

    for(; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

void tiledMatrixMultiply(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C) {
    int n = A.size();
    const int tile_size = 2;
    for(int i = 0; i < n; i+= tile_size) {
        for(int j = 0; j < n; j+= tile_size) {
            
        }
    }
}



int main() {
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    std::vector<std::vector<float>> output;
    im2col(input, output, 3, 1);

    for (const auto& row : output) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }


    return 0;

}