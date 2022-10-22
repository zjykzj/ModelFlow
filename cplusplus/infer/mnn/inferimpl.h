//
// Created by zj on 2022/10/22.
//

#ifndef ONNX_INFER_MNN_INFERIMPL_H_
#define ONNX_INFER_MNN_INFERIMPL_H_

#include "../inferinterface.h"

class InferImpl : public InferInterface {
   public:
    InferImpl() = default;
    ~InferImpl() = default;

    int Init(int width, int height) override;

   private:
    int width_ = 0;
    int height_ = 0;
};

#endif  // ONNX_INFER_MNN_INFERIMPL_H_
