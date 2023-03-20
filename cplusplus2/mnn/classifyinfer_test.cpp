//
// Created by zj on 23-3-20.
//

#include <opencv2/opencv.hpp>

#include "classifyinfer.h"

#include "effnetlite0_v3161.h"

int main(int argc, char *argv[]) {
	auto model = ClassifyInfer();

	float mean[3] = {103.94f, 116.78f, 123.68f};
	float normals[3] = {0.017f, 0.017f, 0.017f};
	model.Init(2, 224, 224, 3, mean, normals, effnetlite0_v3161_mnn, effnetlite0_v3161_mnn_len);

	return 0;
}