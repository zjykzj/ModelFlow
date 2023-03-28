//
// Created by zj on 23-3-28.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

bool DescendSort(std::pair<int, int> a, std::pair<int, int> b) {
	return a.first > b.first;
}

bool AscendSort(std::pair<int, int> a, std::pair<int, int> b) {
	return a.first < b.first;
}

int Demo1() {
	// 创建一个包含10个随机数的vector
	std::vector<int> vec(10);
	for (int i = 0; i < vec.size(); i++) {
		vec[i] = rand() % 100;
	}

	// 输出排序前的vector
	std::cout << "排序前：";
	for (int i = 0; i < vec.size(); i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;

	// 使用sort函数对vector进行排序
	std::sort(vec.begin(), vec.end());

	// 输出排序后的vector
	std::cout << "排序后：";
	for (int i = 0; i < vec.size(); i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}

int Demo2() {
	// 创建一个包含10个随机数的vector
	std::vector<std::pair<int, int>> vec(10);
	for (int i = 0; i < vec.size(); i++) {
		vec[i] = std::make_pair(rand() % 100, i);
	}

	// 输出排序前的vector
	std::cout << "排序前：";
	for (int i = 0; i < vec.size(); i++) {
		std::cout << "[" << vec[i].first << " " << vec[i].second << "] ";
	}
	std::cout << std::endl;

	// 使用sort函数对vector进行排序
//	sort(vec.begin(), vec.end(), AscendSort);
	sort(vec.begin(), vec.end(), DescendSort);

	// 输出排序后的vector
	std::cout << "排序后：";
	for (int i = 0; i < vec.size(); i++) {
		std::cout << "[" << vec[i].first << " " << vec[i].second << "] ";
	}
	std::cout << std::endl;
}

bool cmp(int i, int j) {
	return i > j;
}

int Demo3() {
//	std::vector<int> arr = {3, 1, 4, 1, 5, 9, 2, 6};
	const float bb[8] = {3, 1, 4, 1, 5, 9, 2, 6};
	std::vector<int> arr;
	for (auto &b : bb) {
		arr.push_back(b);
	}
	std::vector<int> idx(arr.size());
	std::iota(idx.begin(), idx.end(), 0);  // 初始化 idx 数组

	std::cout << "排序前下标：";
	for (auto i : idx) {
		std::cout << i << " ";
	}
	std::cout << std::endl;
	std::cout << "排序前数值：";
	for (auto i : idx) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;

//	std::sort(idx.begin(), idx.end(), [&](int i, int j) { return arr[i] < arr[j]; });
	std::sort(idx.begin(), idx.end(), [&](int i, int j) { return cmp(arr[i], arr[j]); });

	std::cout << "排序后下标：";
	for (auto i : idx) {
		std::cout << i << " ";  // 输出排序后的下标
	}
	std::cout << std::endl;
	std::cout << "排序后数值：";
	for (auto i : idx) {
		std::cout << arr[i] << " ";  // 输出排序后的下标
	}
	std::cout << std::endl;

	return 0;
}

int main() {
	std::cout << " => 调用sort函数，从小到大进行排序" << std::endl;
	Demo1();
	std::cout << " => 增加自定义比较算子，从大到小进行排序" << std::endl;
	Demo2();
	std::cout << " => 返回从大到小排序后数值下标" << std::endl;
	Demo3();

	return 0;
}