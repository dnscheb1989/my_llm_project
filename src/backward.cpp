#include "backward.h"
#include "mat_vec_mult.h"
#include "scalar_product.h"
#include "scalar_vec_mult.h"
#include <vector>

void backward_attention(
    const std::vector<double>& loss_grad,
    const std::vector<double>& Q,
    const std::vector<std::vector<double>>& K_matrix,
    const std::vector<std::vector<double>>& V_matrix,
    const std::vector<double>& weights,
    std::vector<std::vector<double>>& grad_W_Q,
    std::vector<std::vector<double>>& grad_W_K,
    std::vector<std::vector<double>>& grad_W_V,
    std::vector<double>& grad_embedding) {
    int dModel = Q.size();
    int seqLen = K_matrix.size();

    // dLoss/dweights = dLoss/dZ * V^T
    std::vector<double> grad_weights(seqLen, 0.0);
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < dModel; j++) {
            grad_weights[i] += loss_grad[j] * V_matrix[i][j];
        }
    }

    // dLoss/dscores = grad_weights * softmax_derivative
    // Упрощённо для softmax: grad_scores = grad_weights * weights (покомпонентно)
    std::vector<double> grad_scores(seqLen);
    for (int i = 0; i < seqLen; i++) {
        grad_scores[i] = grad_weights[i] * weights[i];
    }

    // dLoss/dQ = grad_scores * K_matrix
    std::vector<double> grad_Q(dModel, 0.0);
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < dModel; j++) {
            grad_Q[j] += grad_scores[i] * K_matrix[i][j];
        }
    }

    // dLoss/dK_matrix = Q^T * grad_scores
    std::vector<std::vector<double>> grad_K(seqLen, std::vector<double>(dModel, 0.0));
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < dModel; j++) {
            grad_K[i][j] = Q[j] * grad_scores[i];
        }
    }

    // dLoss/dV_matrix = weights^T * loss_grad
    std::vector<std::vector<double>> grad_V(seqLen, std::vector<double>(dModel, 0.0));
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < dModel; j++) {
            grad_V[i][j] = weights[i] * loss_grad[j];
        }
    }

    // dLoss/dW_Q = x^T * grad_Q (x — эмбеддинг текущего токена, его нужно передать)
    // Аналогично для W_K и W_V с суммированием по токенам

    // grad_embedding = grad_Q * W_Q^T + sum(grad_K) * W_K^T + sum(grad_V) * W_V^T
}