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

## 🏗️ Model Architecture

I built a custom CNN with multiple convolution blocks. Here’s the structure:

Input: [1, 28, 28]  (grayscale image)

Block 1:
  Conv2d: 1 → 8 channels, kernel=3, padding=1
  ReLU
  MaxPool2d: 2x2

Block 2:
  Conv2d: 8 → 16 channels, kernel=3, padding=1
  ReLU
  MaxPool2d: 2x2
  Conv2d (1x1): 16 → 8 channels (shrink)
  ReLU

Block 3:
  Conv2d: 8 → 16 channels, kernel=3, padding=1
  ReLU
  Conv2d (1x1): 16 → 16 channels (refine)
  ReLU

Block 4:
  Conv2d: 16 → 32 channels, kernel=3, padding=1
  ReLU
  Conv2d (1x1): 32 → 32 channels (refine)
  ReLU

Fully Connected:
  Flatten → Linear(32*7*7 → 10)

Output:
  LogSoftmax across 10 classes (digits 0–9)


### 🧮 Total parameters: under 25k, so it’s a fairly lightweight model.

## ⚙️ Training Setup

Optimizer: SGD with lr=0.01, momentum=0.9

Loss function: CrossEntropyLoss

Normalization: (mean=0.1307, std=0.3081) for MNIST images

Epochs: 10

## 📊 Training Logs

Here’s a snapshot of how training went (sample logs):

Epoch 1
Train: Loss=0.5895 Batch_id=117 Accuracy=46.65
Test set: Average loss: 0.0012, Accuracy: 82.10%

Epoch 5
Train: Accuracy ~96%
Test set: Accuracy ~97.5%

Epoch 10
Train: Accuracy ~99%
Test set: Accuracy ~98.5%

## 🚀 Results

Final Test Accuracy: ~98.5%

Model trains fast because it’s small (<25k params).

Using 1x1 convolutions helped reduce parameters while keeping accuracy high.

## 🌱 What I Learned

How to design a CNN with multiple blocks.

The role of 1x1 convolutions (they compress or refine feature maps without losing spatial size).

Why normalization is important (transforms.Normalize).

Difference between training loss and test loss (and why overfitting happens).

## 🧩 Next Steps

Try adding BatchNorm and Dropout to improve stability.

Experiment with different optimizers (SGD, RMSprop).

Try data augmentation (rotation, shift, etc.) for better generalization.

✨ This is a learning project, so I kept things simple. Feedback is welcome!

Would you like me to also add graphs (loss vs accuracy curves) into the README so that it looks even more beginner-friendly and visually clear?
