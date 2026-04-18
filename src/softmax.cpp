// softmax.cpp
#include "softmax.h"
#include <cmath>
#include <algorithm>


/*Принцип: Превращает числа в вероятности, чтобы сумма = 1. 
Большие числа → большие вероятности, отрицательные → маленькие.
Аналогия: У тебя три числа: 5, 1, -2. 
Softmax сделает их примерно: 0.98, 0.02, 0.00. Они же в сумме дают 1.*/
std::vector<double> softmax(const std::vector<double>& scores) {
    std::vector<double> res(scores.size());
    double max_val = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;
    for (size_t i = 0; i < scores.size(); i++) {
        res[i] = std::exp(scores[i] - max_val);
        sum += res[i];
    }
    for (size_t i = 0; i < scores.size(); i++) {
        res[i] /= sum;
    }
    return res;
}