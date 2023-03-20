//
// Created by zj on 23-3-20.
//

#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>

#include "classifyinfer.h"

int ClassifyInfer::Init(int num_thread,
						int img_w,
						int img_h,
						int img_step,
						const float *means,
						const float *normals,
						const unsigned char *model_buffer,
						unsigned int model_len) {
	this->interpreter_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(model_buffer, model_len));

	MNN::ScheduleConfig net_config;
	net_config.type = MNN_FORWARD_OPENCL;
	net_config.backupType = MNN_FORWARD_CPU;
	printf("numThread: %d\n", num_thread);
	net_config.numThread = num_thread;
	this->session_ = this->interpreter_->createSession(net_config);

	auto input = this->interpreter_->getSessionInput(this->session_, nullptr);
	auto shape = input->shape();
	// Set Batch Size
	shape[0] = 1;
	this->interpreter_->resizeTensor(input, shape);
	this->interpreter_->resizeSession(this->session_);
	MNN_PRINT("Model Info: batch size = %d, input channels = %d, input height = %d, input width = %d\n",
			  input->batch(),
			  input->channel(),
			  input->height(),
			  input->width());

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
			  1);

	// Image Process
	this->img_w_ = img_w;
	this->img_h_ = img_h;
	this->img_step_ = img_step;
	this->input_size_ = img_w * img_h * img_step;
	this->input_data_ = new unsigned char[this->input_size_];

	this->pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
		MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, means, 3, normals, 3));

	return 0;
}

unsigned char *ClassifyInfer::GetCharInputPtr() { return this->input_data_; }

int ClassifyInfer::SetInput() {
	auto input = this->interpreter_->getSessionInput(this->session_, nullptr);;
	this->pretreat->convert(this->input_data_, input->width(), input->height(), input->channel(), input);

	return 0;
}

int ClassifyInfer::ModelInfer() {
	MNN::ErrorCode error_code = this->interpreter_->runSession(this->session_);
	return int(error_code);
}

int ClassifyInfer::GetInputNum() { return this->input_size_; }

float *ClassifyInfer::GetOutPtr() {
	auto output = this->interpreter_->getSessionOutput(this->session_, "output");
	return output->host<float>();
}

int ClassifyInfer::GetOutNum() {
	auto output = this->interpreter_->getSessionOutput(this->session_, "output");
	return output->elementSize();
}

int ClassifyInfer::Release() {
	if (nullptr != this->session_) {
		this->interpreter_->releaseModel();
		this->interpreter_->releaseSession(this->session_);
		this->session_ = nullptr;
	}

	if (nullptr != this->input_data_) {
		delete[] this->input_data_;
		this->input_data_ = nullptr;
	}

	return 0;
}