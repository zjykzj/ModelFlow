
# Results

## Linear Probe + LR Classifier

```text
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_linear_probe_cifar.py --dataset cifar10 --classifier logistic
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR10 training set...
📂 Loading CIFAR10 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:25<00:00, 30.45it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 30.02it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Training Logistic Regression classifier...
✅ Logistic classifier training completed.
🧪 Evaluating on test set...

======================================================================
🎯 OpenCLIP Frozen Feature Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR10
   Classifier:  logistic
   Train Size:  50000
   Test Size:   10000
   Accuracy:    94.8500%  (9485/10000)
======================================================================
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_linear_probe_cifar.py --dataset cifar100 --classifier logistic
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR100 training set...
📂 Loading CIFAR100 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:24<00:00, 31.77it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 27.15it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Training Logistic Regression classifier...
✅ Logistic classifier training completed.
🧪 Evaluating on test set...

======================================================================
🎯 OpenCLIP Frozen Feature Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR100
   Classifier:  logistic
   Train Size:  50000
   Test Size:   10000
   Accuracy:    78.7000%  (7870/10000)
======================================================================
```

## Linear Probe + KNN Classifier (k=1)

```text
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python openclip_linear_probe_cifar.py --dataset cifar10 --classifier knn --k 1
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR10 training set...
📂 Loading CIFAR10 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:24<00:00, 31.84it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:04<00:00, 31.44it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Setting up KNN classifier with k=1 (metric=cosine, since features are normalized)...
✅ KNN classifier ready (no training needed).
🧪 Evaluating on test set...

======================================================================
🎯 OpenCLIP Frozen Feature Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR10
   Classifier:  knn
   k (top-N):   1
   Train Size:  50000
   Test Size:   10000
   Accuracy:    91.9100%  (9191/10000)
======================================================================
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python openclip_linear_probe_cifar.py --dataset cifar100 --classifier knn --k 1
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR100 training set...
📂 Loading CIFAR100 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:25<00:00, 30.91it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 29.65it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Setting up KNN classifier with k=1 (metric=cosine, since features are normalized)...
✅ KNN classifier ready (no training needed).
🧪 Evaluating on test set...

======================================================================
🎯 OpenCLIP Frozen Feature Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR100
   Classifier:  knn
   k (top-N):   1
   Train Size:  50000
   Test Size:   10000
   Accuracy:    70.8300%  (7083/10000)
======================================================================
```

## Linear Probe + KNN Classifier (k=5)

```text
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python openclip_linear_probe_cifar.py --dataset cifar10 --classifier knn --k 5
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR10 training set...
📂 Loading CIFAR10 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:24<00:00, 31.84it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 28.54it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Setting up KNN classifier with k=5 (metric=cosine, since features are normalized)...
✅ KNN classifier ready (no training needed).
🧪 Evaluating on test set...

======================================================================
🎯 OpenCLIP Frozen Feature Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR10
   Classifier:  knn
   k (top-N):   5
   Train Size:  50000
   Test Size:   10000
   Accuracy:    93.5600%  (9356/10000)
======================================================================
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
(base) root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python openclip_linear_probe_cifar.py --dataset cifar100 --classifier knn --k 5
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR100 training set...
📂 Loading CIFAR100 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:24<00:00, 31.46it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 31.19it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Setting up KNN classifier with k=5 (metric=cosine, since features are normalized)...
✅ KNN classifier ready (no training needed).
🧪 Evaluating on test set...

======================================================================
🎯 OpenCLIP Frozen Feature Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR100
   Classifier:  knn
   k (top-N):   5
   Train Size:  50000
   Test Size:   10000
   Accuracy:    73.3200%  (7332/10000)
======================================================================
```