#pragma once
#include <vector>

class MLP {
public:
    MLP(int inputSize, int hiddenSize, int outputSize);
    void train(const std::vector<double>& input, const std::vector<double>& target);
    std::vector<double> predict(const std::vector<double>& input);  // public
    static void testMLP();  // добавь static
private:
    int inputNodes, hiddenNodes, outputNodes;
    std::vector<std::vector<double>> wih, who;
    double sigmoid(double x);
    void feedForward(const std::vector<double>& input, std::vector<double>& hidden, std::vector<double>& output);
};