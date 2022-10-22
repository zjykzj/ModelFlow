//
// Created by zj on 2022/10/22.
//

#ifndef ONNX_INFER_INFERINTERFACE_H_
#define ONNX_INFER_INFERINTERFACE_H_

class InferInterface {
   public:
    virtual int Init(int width, int height) = 0;
};

#endif  // ONNX_INFER_INFERINTERFACE_H_
