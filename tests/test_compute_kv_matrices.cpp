// tests/test_compute_kv_matrices
//Вычисляет матрицы ключей K_matrix и значений V_matrix для всех токенов 
//последовательности.Каждая строка — результат умножения эмбеддинга токена на W_K и W_V.
#include "LLM.h"
#include <iostream>
#include <vector>

void test_compute_kv_matrices() {
    int vocabSize = 4, dModel = 4, seqLen = 4;
    LLM llm(vocabSize, dModel, seqLen);

    std::vector<int> tokens = { 0, 1, 2, 3 };
    std::vector<std::vector<double>> K_matrix, V_matrix;

    llm.compute_kv_matrices(tokens, K_matrix, V_matrix);

    std::cout << "K_matrix size: " << K_matrix.size() << "x" << K_matrix[0].size() << std::endl;
    std::cout << "V_matrix size: " << V_matrix.size() << "x" << V_matrix[0].size() << std::endl;
}

/*что должно прийти что должно выдать? кратко
Должно выдать:

K_matrix size: 4x4
V_matrix size: 4x4
Это значит, что матрицы созданы правильно. Если значения не вывелись — ошибка.
*/