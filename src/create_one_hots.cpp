#include "create_one_hots.h"

/*Технически: Создаёт квадратную матрицу size × size,
заполняет её нулями, затем ставит 1.0 на главной диагонали.

Логически: Превращает индекс токена в one-hot вектор
(единица на позиции индекса, нули elsewhere).
Используется для создания целевых меток (target) при обучении.*/
std::vector<std::vector<double>> create_one_hots(const std::vector<std::string>& dict) {
	int size = dict.size();
	std::vector<std::vector<double>> one_hots(size, std::vector<double>(size, 0.0));
	for (int i = 0; i < size; i++) {
		one_hots[i][i] = 1.0;
	}
	return one_hots;
}