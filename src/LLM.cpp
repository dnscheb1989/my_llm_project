#include "backward.h"
#include "LLM.h"
#include "MLP.h"
#include "softmax.h"
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


//Тренирует MLP предсказывать следующий токен, используя Z (контекстный вектор)
void LLM::train(const std::vector<int>& tokens, const std::vector<int>& targets) {
    std::vector<std::vector<double>> K_matrix(seqLen, std::vector<double>(dModel));
    std::vector<std::vector<double>> V_matrix(seqLen, std::vector<double>(dModel));
    for (int t = 0; t < seqLen; t++) {
        std::vector<double> x = embedding[tokens[t]];
        K_matrix[t] = mat_vec_mult(x, W_K);
        V_matrix[t] = mat_vec_mult(x, W_V);
    }

    // Предполагаем, что tokens.size() == seqLen, targets.size() == seqLen
    for (int i = 0; i < seqLen; i++) {
        // Берём эмбеддинг текущего токена
        std::vector<double> x = embedding[tokens[i]];

        // Вычисляем Q, K, V
        std::vector<double> Q = mat_vec_mult(x, W_Q);
        //std::vector<double> K = mat_vec_mult(x, W_K);
        //std::vector<double> V = mat_vec_mult(x, W_V);

        /*Что: Сравниваем текущее слово со всеми словами в предложении. 
        Получаем список чисел — насколько каждое слово важно.
        Зачем: Чтобы потом взвешенно сложить их значения.
        это основа внимания*/
        std::vector<double> scores(seqLen);
        for (int j = 0; j < seqLen; j++) {
            scores[j] = scalar_product(Q, K_matrix[j]);
        }

        //  Нормализуем score с помощью softmax scores в weights
        std::vector<double> weights = softmax(scores);


        std::vector<double> Z(dModel, 0.0);
        for (int j = 0; j < seqLen; j++) {
            std::vector<double> weighted = scalar_vec_mult(weights[j], V_matrix[j]);
            for (int k = 0; k < dModel; k++) {
                Z[k] += weighted[k];
            }
        }

        // Целевой one-hot вектор для текущего токена
        std::vector<double> target_one_hot(vocabSize, 0.0);
        target_one_hot[targets[i]] = 1.0;

        // Обучаем MLP
        mlp.train(Z, target_one_hot);
    }
}

std::vector<double> LLM::predict(const std::vector<int>& tokens) {
    // Шаг 5: Собираем матрицы K и V для всех токенов
    std::vector<std::vector<double>> K_matrix(seqLen, std::vector<double>(dModel));
    std::vector<std::vector<double>> V_matrix(seqLen, std::vector<double>(dModel));
    for (int t = 0; t < seqLen; t++) {
        std::vector<double> x = embedding[tokens[t]];
        K_matrix[t] = mat_vec_mult(x, W_K);
        V_matrix[t] = mat_vec_mult(x, W_V);
    }

    std::vector<double> last_Z; // сохранит выход последнего токена

    for (int i = 0; i < seqLen; i++) {
        // Берём эмбеддинг текущего токена
        std::vector<double> x = embedding[tokens[i]];

        // Вычисляем Q для текущего токена
        std::vector<double> Q = mat_vec_mult(x, W_Q);

        // Шаг 6: Вычисляем scores (скалярные произведения Q с каждым K)
        std::vector<double> scores(seqLen);
        for (int j = 0; j < seqLen; j++) {
            scores[j] = scalar_product(Q, K_matrix[j]);
        }

        // Шаг 7: Нормализуем scores через softmax -> weights
        std::vector<double> weights = softmax(scores);

        // Шаг 8: Вычисляем Z как взвешенную сумму V_matrix
        std::vector<double> Z(dModel, 0.0);
        for (int j = 0; j < seqLen; j++) {
            std::vector<double> weighted = scalar_vec_mult(weights[j], V_matrix[j]);
            for (int k = 0; k < dModel; k++) {
                Z[k] += weighted[k];
            }
        }

        // Запоминаем Z последнего токена
        if (i == seqLen - 1) last_Z = Z;
    }

    // Шаг 9: Подаём Z в MLP для предсказания
    return mlp.predict(last_Z);
}

// train - 1 Вычисляет матрицы ключей K_matrix и значений V_matrix для всех токенов 
//последовательности.Каждая строка — результат умножения эмбеддинга токена на W_K и W_V.
void LLM::compute_kv_matrices(const std::vector<int>& tokens,
    std::vector<std::vector<double>>& K_matrix,
    std::vector<std::vector<double>>& V_matrix) {
    K_matrix.assign(seqLen, std::vector<double>(dModel));
    V_matrix.assign(seqLen, std::vector<double>(dModel));
    for (int t = 0; t < seqLen; t++) {
        std::vector<double> x = embedding[tokens[t]];
        K_matrix[t] = mat_vec_mult(x, W_K);
        V_matrix[t] = mat_vec_mult(x, W_V);
    }
}