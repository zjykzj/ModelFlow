//
// Created by zj on 23-3-20.
//

#ifndef MNN__CLASSIFYINFER_H_
#define MNN__CLASSIFYINFER_H_

#include <MNN/Interpreter.hpp>

class ClassifyInfer {

 public:
  ClassifyInfer() = default;
  ~ClassifyInfer() = default;

  int Init(int num_thread, int img_w, int img_h, int img_step,
		   const float *means, const float *normals,
		   const unsigned char *model_buffer, unsigned int model_len);

  unsigned char *GetCharInputPtr();

  int SetInput();

  int ModelInfer();

  int GetInputNum();

  float *GetOutPtr();

  int GetOutNum();

  int Release();

 private:
  int img_w_ = 224;
  int img_h_ = 224;
  int img_step_ = 3;
  uint32_t input_size_ = 0;
  unsigned char *input_data_ = nullptr;

  std::unique_ptr<MNN::Interpreter> interpreter_;
  MNN::Session *session_ = nullptr;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat = nullptr;

};

#endif //MNN__CLASSIFYINFER_H_
