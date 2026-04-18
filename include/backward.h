/*√радиенты, полученные дл€ одного токена, 
обновл€ют общие матрицы W_Q, W_K, W_V, а также эмбеддинги всех токенов 
(через сумму вкладов).*/

#pragma once
#include <vector>

void backward_attention(
    const std::vector<double>& loss_grad,      // градиент от MLP (dLoss/dZ)
    const std::vector<double>& Q,              // Q текущего токена
    const std::vector<std::vector<double>>& K_matrix, // все K
    const std::vector<std::vector<double>>& V_matrix, // все V
    const std::vector<double>& weights,        // weights = softmax(scores)
    std::vector<std::vector<double>>& grad_W_Q, // сюда запишем градиент дл€ W_Q
    std::vector<std::vector<double>>& grad_W_K, // дл€ W_K
    std::vector<std::vector<double>>& grad_W_V, // дл€ W_V
    std::vector<double>& grad_embedding        // градиент дл€ эмбеддинга текущего токена
);