#include "MLP.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

MLP::MLP(int inputSize, int hiddenSize, int outputSize)
    : inputNodes(inputSize), hiddenNodes(hiddenSize), outputNodes(outputSize) {
    wih.assign(inputNodes, std::vector<double>(hiddenNodes));
    who.assign(hiddenNodes, std::vector<double>(outputNodes));
    for (int i = 0; i < inputNodes; i++)
        for (int j = 0; j < hiddenNodes; j++)
            wih[i][j] = (rand() % 1000) / 1000.0 - 0.5;
    for (int j = 0; j < hiddenNodes; j++)
        for (int k = 0; k < outputNodes; k++)
            who[j][k] = (rand() % 1000) / 1000.0 - 0.5;
}

double MLP::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void MLP::feedForward(const std::vector<double>& input, std::vector<double>& hidden, std::vector<double>& output) {
    for (int j = 0; j < hiddenNodes; j++) {
        hidden[j] = 0;
        for (int i = 0; i < inputNodes; i++)
            hidden[j] += input[i] * wih[i][j];
        hidden[j] = sigmoid(hidden[j]);
    }
    for (int k = 0; k < outputNodes; k++) {
        output[k] = 0;
        for (int j = 0; j < hiddenNodes; j++)
            output[k] += hidden[j] * who[j][k];
        output[k] = sigmoid(output[k]);
    }
}

void MLP::train(const std::vector<double>& input, const std::vector<double>& target) {
    std::vector<double> hidden(hiddenNodes), output(outputNodes);
    feedForward(input, hidden, output);
    double lr = 0.3;
    std::vector<double> output_errors(outputNodes);
    for (int k = 0; k < outputNodes; k++)
        output_errors[k] = target[k] - output[k];
    for (int j = 0; j < hiddenNodes; j++)
        for (int k = 0; k < outputNodes; k++)
            who[j][k] += lr * output_errors[k] * output[k] * (1 - output[k]) * hidden[j];
    std::vector<double> hidden_errors(hiddenNodes);
    for (int j = 0; j < hiddenNodes; j++) {
        hidden_errors[j] = 0;
        for (int k = 0; k < outputNodes; k++)
            hidden_errors[j] += output_errors[k] * who[j][k];
    }
    for (int i = 0; i < inputNodes; i++)
        for (int j = 0; j < hiddenNodes; j++)
            wih[i][j] += lr * hidden_errors[j] * hidden[j] * (1 - hidden[j]) * input[i];
}

std::vector<double> MLP::predict(const std::vector<double>& input) {
    std::vector<double> hidden(hiddenNodes), output(outputNodes);
    feedForward(input, hidden, output);
    return output;
}

void MLP::testMLP() {
    MLP mlp(10, 16, 10);
    std::vector<double> input = { 1,0,0,0,0,0,0,0,0,0 };
    std::vector<double> target = { 0,1,1,0,0,0,0,0,0,1 };
    for (int epoch = 0; epoch < 10000; epoch++) {
        mlp.train(input, target);
    }
    auto output = mlp.predict(input);
    for (double v : output) std::cout << v << " ";
    std::cout << std::endl;
}