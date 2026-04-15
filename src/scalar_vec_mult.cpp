#include "scalar_vec_mult.h"

// ”множени€ скал€ра на вектор
std::vector<double> scalar_vec_mult(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = scalar * vec[i];
    }
    return result;
}