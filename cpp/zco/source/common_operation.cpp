//
// Created by zj on 2021/8/17.
//

#include "common_operation.h"

bool read_txt_file(const std::string &file_name, std::vector<std::string> *result) {
    std::ifstream file(file_name);
    if (!file) {
        std::cerr << "TXT file " << file_name << " not found\n";
        return false;
    }
    result->clear();
    std::string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    return true;
}

void get_top_n(const std::vector<float> &input, std::vector<size_t> &output_sorted_idxes) {
    std::iota(output_sorted_idxes.begin(), output_sorted_idxes.end(), 0);
    std::sort(output_sorted_idxes.begin(), output_sorted_idxes.end(),
              [&input](size_t index_1, size_t index_2) {
                  return input[index_1] > input[index_2];
              }
    );
}

size_t get_max_idx(const std::vector<float> &input) {
    size_t result_ = std::distance(input.begin(), std::max_element(input.begin(), input.end()));
    return result_;
}
