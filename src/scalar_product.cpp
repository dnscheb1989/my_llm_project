#include "scalar_product.h"

// ѕолучаем скал€рное произведение двух векторов
double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}