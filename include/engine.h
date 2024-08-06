#ifndef TINY_ENGINE_H
#define TINY_ENGINE_H

#include <stdint.h>
#include <vector>

using namespace std;

typedef int8_t q7_t;
typedef uint8_t q8_t;
typedef int16_t q15_t;
typedef uint16_t q16_t;
typedef int32_t q31_t;
typedef int32_t q32_t;

typedef vector<vector<vector<q7_t>>> rgb7_t;
typedef vector<vector<vector<q8_t>>> rgb8_t;
typedef vector<vector<vector<q15_t>>> rgb15_t;
typedef vector<vector<vector<q16_t>>> rgb16_t;
typedef vector<vector<vector<q31_t>>> rgb31_t;
typedef vector<vector<vector<q32_t>>> rgb32_t;

class TinyEngine {

    
};

#endif // TINY_ENGINE_H