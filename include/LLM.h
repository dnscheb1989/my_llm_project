#pragma once
#include "MLP.h"
#include <vector>
#include <string>

class LLM {
public:
    //Технически: Принимает три целых числа (int).
    //vocabSize — количество уникальных токенов в словаре.
    //dModel — размерность эмбеддинга (длина вектора токена).
    //seqLen — максимальная длина входной последовательности.
    LLM(int vocabSize, int dModel, int seqLen);
    //Это собственный train LLM. Он будет:
    //Преобразовывать tokens (индексы) в эмбеддинги.
    //Вычислять attention (Q, K, V, score, Z).
    //Вызывать mlp.train(Z, target_embedding), где target_embedding — one-hot вектор из targets.
    void train(const std::vector<int>& tokens, const std::vector<int>& targets);
    std::vector<double> predict(const std::vector<int>& tokens);

    //Вычисляет матрицы ключей K_matrix и значений V_matrix для всех токенов 
    //последовательности.Каждая строка — результат умножения эмбеддинга токена на W_K и W_V.
    void compute_kv_matrices(const std::vector<int>& tokens,
        std::vector<std::vector<double>>& K_matrix,
        std::vector<std::vector<double>>& V_matrix);

private:
    int vocabSize, dModel, seqLen;
    std::vector<std::vector<double>> embedding;  // vocabSize x dModel
    std::vector<std::vector<double>> W_Q, W_K, W_V; // каждая dModel x dModel
    MLP mlp;  // композиция: вход dModel, скрытый dModel*4, выход vocabSize
    
};