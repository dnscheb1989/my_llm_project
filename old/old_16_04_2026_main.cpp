#define TEST_MLP

#ifdef TEST_MLP
#include "MLP.h"
#include <iostream>
#include <vector>

int main() {
    MLP::testMLP();
}
#else

#include "LLM.h"
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

int main() {
    srand(time(0));

    vector<string> dict = { "привет", "как", "дела", "?" };
    int vocabSize = dict.size();
    int dModel = 4;
    int seqLen = 4;

    LLM llm(vocabSize, dModel, seqLen);

    // Преобразуем слова в индексы
    vector<int> tokens = { 0, 1, 2, 3 };
    vector<int> targets = { 1, 2, 3, 0 };  // следующий токен (по кругу)

    // Обучение
    for (int epoch = 0; epoch < 1000; epoch++) {
        llm.train(tokens, targets);
    }

    // Вывод результатов
    vector<double> output = llm.predict(tokens);
    for (double v : output) cout << v << " ";
    cout << endl;

    return 0;
}
#endif
