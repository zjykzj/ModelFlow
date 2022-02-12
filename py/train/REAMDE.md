# MNIST

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

I use following commands:

```bash
python main.py --epochs 5 --save-model
```

After the training, you can see `mnist_cnn.pt` in `outputs/`

## Reference

* [examples/mnist](https://github.com/pytorch/examples/tree/master/mnist)