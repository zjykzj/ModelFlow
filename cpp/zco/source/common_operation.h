//
// Created by zj on 2021/8/17.
//

#ifndef ZCM_COMMON_OPERATION_H
#define ZCM_COMMON_OPERATION_H

#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <complex>

/**
 * 对向量中元素执行累乘操作
 * @tparam T
 * @param v
 * @return
 */
template<typename T>
T vector_product(const std::vector<T> &v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * 计算分类概率
 * @tparam T
 * @param input
 */
template<typename T>
void softmax(T &input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

/**
 * 输入向量，进行降序排序，返回排序下标
 * 参考[C++ 数组排序返回下标](https://blog.csdn.net/wuqingshan2010/article/details/108508499)
 * @tparam T
 * @param input
 */
void get_top_n(const std::vector<float> &input, std::vector<size_t> &output_sorted_idxes);

/**
 * 获取数组最大值下标
 * @param input
 * @return
 */
size_t get_max_idx(const std::vector<float> &input);

/**
 * 读取文本文件每行内容，保存到结果向量
 * @param file_name
 * @param result
 * @return
 */
bool read_txt_file(const std::string &file_name, std::vector<std::string> *result);


#endif //ZCM_COMMON_OPERATION_H
