//
// Created by zj on 2022/10/22.
//

#include "gtest/gtest.h"

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}