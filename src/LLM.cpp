#include "LLM.h"
#include "MLP.h"
#include "random_matrix.h"
#include "mat_vec_mult.h"
#include "scalar_product.h"
#include "scalar_vec_mult.h"
#include <vector>

LLM::LLM(int vocabSize, int dModel, int seqLen)
    //member initializer list)
    : vocabSize(vocabSize), dModel(dModel), seqLen(seqLen), 
        mlp(dModel, dModel * 4, vocabSize) {
    // Инициализация эмбеддингов
    fill_matrix_random(embedding, vocabSize, dModel);

    // Инициализация матриц внимания
    fill_matrix_random(W_Q, dModel, dModel);
    fill_matrix_random(W_K, dModel, dModel);
    fill_matrix_random(W_V, dModel, dModel);
}

void LLM::train(const std::vector<int>& tokens, const std::vector<int>& targets) {
    // Предполагаем, что tokens.size() == seqLen, targets.size() == seqLen
    for (int i = 0; i < seqLen; i++) {
        // Берём эмбеддинг текущего токена
        std::vector<double> x = embedding[tokens[i]];

        // Вычисляем Q, K, V
        std::vector<double> Q = mat_vec_mult(x, W_Q);
        std::vector<double> K = mat_vec_mult(x, W_K);
        std::vector<double> V = mat_vec_mult(x, W_V);

        // Вычисляем score и Z
        double score = scalar_product(Q, K);
        std::vector<double> Z = scalar_vec_mult(score, V);

        // Целевой one-hot вектор для текущего токена
        std::vector<double> target_one_hot(vocabSize, 0.0);
        target_one_hot[targets[i]] = 1.0;

        // Обучаем MLP
        mlp.train(Z, target_one_hot);
    }
}

std::vector<double> LLM::predict(const std::vector<int>& tokens) {
    // Предполагаем, что tokens.size() == seqLen
    std::vector<double> last_Z; // сохраним последний Z

    for (int i = 0; i < seqLen; i++) {
        std::vector<double> x = embedding[tokens[i]];

        std::vector<double> Q = mat_vec_mult(x, W_Q);
        std::vector<double> K = mat_vec_mult(x, W_K);
        std::vector<double> V = mat_vec_mult(x, W_V);

        double score = scalar_product(Q, K);
        std::vector<double> Z = scalar_vec_mult(score, V);

        if (i == seqLen - 1) {
            last_Z = Z; // запоминаем выход последнего токена
        }
    }

    // Предсказание MLP на последнем Z
    return mlp.predict(last_Z);
}