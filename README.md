# 🖊️ MNIST Digit Recognition

This is my notebook where I trained a Convolutional Neural Network (CNN) on the MNIST dataset (handwritten digits from 0 to 9).
I’m still learning, so this is my attempt to understand how CNNs work and how different layers affect the performance.

## 📌 About the Dataset

Dataset: MNIST

Images: 28x28 grayscale handwritten digits (0–9).

Train set: 60,000 images

Test set: 10,000 images

Every image looks like this:

28 x 28
pixel values (0–255)
grayscale

## Data Augmentation

To make the model generalize better, I applied the following transforms to the training data:

CenterCrop(22) (applied randomly with probability 0.1)

Resize back to (28, 28)

Random rotation between -15° and +15°

Convert to tensor

Normalize with mean = 0.1307 and std = 0.3081

For test data:

Only tensor conversion + normalization

## 🏗️ Model Architecture

I built a custom CNN with multiple convolution blocks. Here’s the structure:
```
Input: 1 x 28 x 28 # (grayscale image)

Block 1:
  Conv2d(1 → 8, kernel=3, padding=1)
  ReLU
  MaxPool2d(2x2)

Block 2:
  Conv2d(8 → 16, kernel=3, padding=1)
  ReLU
  MaxPool2d(2x2)
  Conv2d(16 → 8, kernel=1)   # channel reduction

Block 3:
  Conv2d(8 → 16, kernel=3, padding=1)
  ReLU
  Conv2d(16 → 16, kernel=1)  # refinement

Block 4:
  Conv2d(16 → 32, kernel=3, padding=1)
  ReLU
  Conv2d(32 → 32, kernel=1)  # refinement

Fully Connected:
  Flatten
  Linear(32×7×7 → 10)
  LogSoftmax
```



### 🧮 Total parameters: under 25k, so it’s a fairly lightweight model.

## ⚙️ Training Setup

Optimizer: Adam (lr=0.002)

Scheduler: StepLR (decay every 15 steps, gamma=0.1)

Loss Function: CrossEntropyLoss

Batch Size: 64

Epochs: 5

Device: CUDA if available

## 📊 Training Logs

Here’s a snapshot of how training went (sample logs):

Epoch 1
Train: Loss=0.0457 Batch_id=937 Accuracy=90.51
Test set: Average loss: 0.0015, Accuracy: 58235/60000 (**97.06%**)

**Achieved Test accuracy of 97% at 1st epoch itself!**

Epoch 5
Train: Accuracy = 98.23%
Test set: Accuracy 58893/60000 (98.16%)



## 🚀 Results

Final Test Accuracy: ~98.16%

Model trains fast because it’s small (<25k params).

Using 1x1 convolutions helped reduce parameters while keeping accuracy high.

<img width="1222" height="836" alt="image" src="https://github.com/user-attachments/assets/bb79c7d0-a64a-460e-a9d7-1f12dc1a4443" />

## 🔍 Training Summary:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
         MaxPool2d-2            [-1, 8, 14, 14]               0
            Conv2d-3           [-1, 16, 14, 14]           1,168
         MaxPool2d-4             [-1, 16, 7, 7]               0
            Conv2d-5              [-1, 8, 7, 7]             136
            Conv2d-6             [-1, 16, 7, 7]           1,168
            Conv2d-7             [-1, 16, 7, 7]             272
            Conv2d-8             [-1, 32, 7, 7]           4,640
            Conv2d-9             [-1, 32, 7, 7]           1,056
           Linear-10                   [-1, 10]          15,690
================================================================
Total params: 24,210
Trainable params: 24,210
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.13
Params size (MB): 0.09
Estimated Total Size (MB): 0.22
----------------------------------------------------------------
```

## 🌱 What I Learned

How to design a CNN with multiple blocks.

The role of 1x1 convolutions (they compress or refine feature maps without losing spatial size).

Data augmentation really helps prevent overfitting.

Difference between training loss and test loss (and why overfitting happens).

## 🧩 Next Steps

Try adding BatchNorm and Dropout to improve stability.

Experiment with different optimizers (SGD, RMSprop).

Try data augmentation (rotation, shift, etc.) for better generalization.

✨ This is a learning project, so I kept things simple. Feedback is welcome!

Would you like me to also add graphs (loss vs accuracy curves) into the README so that it looks even more beginner-friendly and visually clear?
