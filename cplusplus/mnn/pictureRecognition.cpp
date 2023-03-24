//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;

class MNNInfer {
 public:
  MNNInfer() = default;
  ~MNNInfer();

  int Init(const std::string &model_path, int batch_size);

  int Input(const std::vector<std::string> &img_paths);

  int Infer();

  std::shared_ptr<Tensor> Output();

  void UpdateCache();

  int Release();

 private:
  std::shared_ptr<MNN::Interpreter> interpreter_ = nullptr;
  MNN::Session *session_ = nullptr;
  std::shared_ptr<Tensor> outputUser = nullptr;
};
int MNNInfer::Init(const std::string &model_path, int batch_size) {
	if (nullptr == this->interpreter_) {
		this->interpreter_ =
			std::shared_ptr<MNN::Interpreter>(Interpreter::createFromFile(model_path.c_str()), Interpreter::destroy);

		this->interpreter_->setCacheFile(".cachefile");
		this->interpreter_->setSessionMode(MNN::Interpreter::Session_Backend_Auto);
		this->interpreter_->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);

		ScheduleConfig config;
		config.type = MNN_FORWARD_AUTO;
		this->session_ = this->interpreter_->createSession(config);
	}

	auto input = this->interpreter_->getSessionInput(this->session_, NULL);
	auto shape = input->shape();
	// Set Batch Size
	shape[0] = batch_size;
	this->interpreter_->resizeTensor(input, shape);
	this->interpreter_->resizeSession(this->session_);

	float memoryUsage = 0.0f;
	this->interpreter_->getSessionInfo(this->session_, MNN::Interpreter::MEMORY, &memoryUsage);
	float flops = 0.0f;
	this->interpreter_->getSessionInfo(this->session_, MNN::Interpreter::FLOPS, &flops);
	int backendType[2];
	this->interpreter_->getSessionInfo(this->session_, MNN::Interpreter::BACKENDS, backendType);
	MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d, batch size = %d\n",
			  memoryUsage,
			  flops,
			  backendType[0],
			  batch_size);

	auto output = this->interpreter_->getSessionOutput(this->session_, NULL);
	if (nullptr == output || output->elementSize() == 0) {
		MNN_ERROR("Resize error, the model can't run batch: %d\n", shape[0]);
		return -1;
	}

	auto dimType = output->getDimensionType();
	if (output->getType().code != halide_type_float) {
		dimType = Tensor::TENSORFLOW;
	}
	this->outputUser = std::make_shared<Tensor>(output, dimType);

	return 0;
}
int MNNInfer::Release() {
	if (nullptr != this->session_) {
		this->interpreter_->releaseModel();
		this->interpreter_->releaseSession(this->session_);
		this->session_ = nullptr;
		this->interpreter_ = nullptr;
	}
	return 0;
}
int MNNInfer::Input(const std::vector<std::string> &img_paths) {
	auto input = this->interpreter_->getSessionInput(this->session_, NULL);
	std::shared_ptr<Tensor> inputUser(new Tensor(input, Tensor::TENSORFLOW));
	auto bpp = inputUser->channel();
	auto size_h = inputUser->height();
	auto size_w = inputUser->width();
	MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

	int batch_size = input->shape()[0];
	for (int batch = 0; batch < batch_size; ++batch) {
		const std::string &inputPatch = img_paths.at(batch);
		int width, height, channel;
		auto inputImage = stbi_load(inputPatch.c_str(), &width, &height, &channel, 4);
		if (nullptr == inputImage) {
			MNN_ERROR("Can't open %s\n", inputPatch.c_str());
			return 0;
		}
		MNN_PRINT("origin size: %d, %d\n", width, height);

		Matrix trans;
		// Set transform, from dst scale to src, the ways below are both ok
#ifdef USE_MAP_POINT
		float srcPoints[] = {
			0.0f, 0.0f,
			0.0f, (float)(height-1),
			(float)(width-1), 0.0f,
			(float)(width-1), (float)(height-1),
		};
		float dstPoints[] = {
			0.0f, 0.0f,
			0.0f, (float)(size_h-1),
			(float)(size_w-1), 0.0f,
			(float)(size_w-1), (float)(size_h-1),
		};
		trans.setPolyToPoly((Point*)dstPoints, (Point*)srcPoints, 4);
#else
		trans.setScale((float)(width - 1) / (size_w - 1), (float)(height - 1) / (size_h - 1));
#endif
		ImageProcess::Config config;
		config.filterType = BILINEAR;
		float mean[3] = {103.94f, 116.78f, 123.68f};
		float normals[3] = {0.017f, 0.017f, 0.017f};
		// float mean[3]     = {127.5f, 127.5f, 127.5f};
		// float normals[3] = {0.00785f, 0.00785f, 0.00785f};
		::memcpy(config.mean, mean, sizeof(mean));
		::memcpy(config.normal, normals, sizeof(normals));
		config.sourceFormat = RGBA;
		config.destFormat = BGR;

		std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config), ImageProcess::destroy);
		pretreat->setMatrix(trans);
		pretreat->convert((uint8_t *)inputImage,
						  width,
						  height,
						  0,
						  inputUser->host<uint8_t>() + inputUser->stride(0) * batch * inputUser->getType().bytes(),
						  size_w,
						  size_h,
						  bpp,
						  0,
						  inputUser->getType());
		stbi_image_free(inputImage);
	}
	input->copyFromHostTensor(inputUser.get());

	return 0;
}
int MNNInfer::Infer() {
	return this->interpreter_->runSession(this->session_);
}
std::shared_ptr<Tensor> MNNInfer::Output() {
	auto output = this->interpreter_->getSessionOutput(this->session_, NULL);
	output->copyToHostTensor(this->outputUser.get());
	return this->outputUser;
}
void MNNInfer::UpdateCache() {
	if (nullptr != this->session_ and nullptr != this->interpreter_)
		this->interpreter_->updateCacheFile(this->session_);
}
MNNInfer::~MNNInfer() {
	this->Release();
}

int main(int argc, const char *argv[]) {
	if (argc < 3) {
		MNN_PRINT("Usage: ./pictureRecognition.out model.mnn input0.jpg input1.jpg input2.jpg ... \n");
		return 0;
	}
	int batch_size = argc - 2;
	MNNInfer mnn_infer;
	mnn_infer.Init(argv[1], batch_size);

	std::vector<std::string> img_paths;
	for (int i = 0; i < batch_size; i++) {
		img_paths.push_back(argv[i + 2]);
	}
	mnn_infer.Input(img_paths);
	mnn_infer.Infer();

	std::shared_ptr<Tensor> outputUser = mnn_infer.Output();
	auto type = outputUser->getType();
	for (int batch = 0; batch < batch_size; ++batch) {
		MNN_PRINT("For Image: %s\n", argv[batch + 2]);
		auto size = outputUser->stride(0);
		std::vector<std::pair<int, float>> tempValues(size);
		if (type.code == halide_type_float) {
			auto values = outputUser->host<float>() + batch * outputUser->stride(0);
			for (int i = 0; i < size; ++i) {
				tempValues[i] = std::make_pair(i, values[i]);
			}
		}
		if (type.code == halide_type_uint && type.bytes() == 1) {
			auto values = outputUser->host<uint8_t>() + batch * outputUser->stride(0);
			for (int i = 0; i < size; ++i) {
				tempValues[i] = std::make_pair(i, values[i]);
			}
		}
		if (type.code == halide_type_int && type.bytes() == 1) {
			auto values = outputUser->host<int8_t>() + batch * outputUser->stride(0);
			for (int i = 0; i < size; ++i) {
				tempValues[i] = std::make_pair(i, values[i]);
			}
		}
		// Find Max
		std::sort(tempValues.begin(), tempValues.end(),
				  [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });

		int length = size > 10 ? 10 : size;
		for (int i = 0; i < length; ++i) {
			MNN_PRINT("%d, %f\n", tempValues[i].first, tempValues[i].second);
		}
	}
	mnn_infer.UpdateCache();
	mnn_infer.Release();
	return 0;
}
