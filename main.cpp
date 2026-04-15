#define TEST_MLP

#ifdef TEST_MLP
#include "MLP.h"
#include <iostream>
#include <vector>

int main() {
    MLP::testMLP();
}
#else

#include "random_matrix.h"
#include "mat_vec_mult.h"
#include "scalar_product.h"
#include "scalar_vec_mult.h"
#include "create_one_hots.h"
#include "MLP.h"
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

int main() {
    srand(time(0));

    vector<string> dict = { "привет", "как", "дела", "?" };
    vector<vector<double>> target = create_one_hots(dict);

    vector<vector<double>> embedding_matrix(4, vector<double>(4));
    fill_matrix_random(embedding_matrix, 4, 4);

    vector<vector<double>> W_Q(4, vector<double>(4));
    fill_matrix_random(W_Q, 4, 4);
    vector<vector<double>> W_K(4, vector<double>(4));
    fill_matrix_random(W_K, 4, 4);
    vector<vector<double>> W_V(4, vector<double>(4));
    fill_matrix_random(W_V, 4, 4);

    MLP mlp(4, 16, 4);

    // Обучение
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int i = 0; i < dict.size(); i++) {
            vector<double> Q = mat_vec_mult(embedding_matrix[i], W_Q);
            vector<double> K = mat_vec_mult(embedding_matrix[i], W_K);
            vector<double> V = mat_vec_mult(embedding_matrix[i], W_V);
            double score = scalar_product(Q, K);
            vector<double> Z = scalar_vec_mult(score, V);
            int next_idx = (i + 1) % dict.size();
            mlp.train(Z, target[next_idx]);
        }
    }

    // Вывод результатов
    for (int i = 0; i < dict.size(); i++) {
        vector<double> Q = mat_vec_mult(embedding_matrix[i], W_Q);
        vector<double> K = mat_vec_mult(embedding_matrix[i], W_K);
        vector<double> V = mat_vec_mult(embedding_matrix[i], W_V);
        double score = scalar_product(Q, K);
        vector<double> Z = scalar_vec_mult(score, V);
        vector<double> output = mlp.predict(Z);
        for (double v : output) cout << v << " ";
        cout << endl;
    }

    return 0;
}
#endif
