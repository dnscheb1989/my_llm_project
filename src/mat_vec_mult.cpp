#include "mat_vec_mult.h"

// Умножаем вектор на матрицу
std::vector<double> mat_vec_mult(const std::vector<double>& vec,
    const std::vector<std::vector<double>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<double> res(rows, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res[i] += vec[j] * mat[i][j];
        }
    }
    return res;
}