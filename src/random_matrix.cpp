#include "random_matrix.h"
#include <cstdlib>

void fill_matrix_random(std::vector<std::vector<double>>& M, int rows, int cols) {
    M.assign(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i][j] = (rand() % 1000) / 1000.0 - 0.5;
        }
    }
}