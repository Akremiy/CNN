Here's the updated README with the correct file name:

# CNN for CIFAR-10 Classification

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The CNN is trained to recognize 10 different classes of objects, including planes, cars, birds, and more.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)

## Installation

To get started with this project, make sure you have Python and the required libraries installed. You can install the necessary packages using pip:

```bash
pip install torch torchvision matplotlib
```

## Usage

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. Run the script:
    ```bash
    python main.py
    ```

## Model Architecture

The CNN consists of:
- Two convolutional layers followed by ReLU activations and max pooling.
- Two fully connected layers for classification.

### Architecture Breakdown:
- **Conv Layer 1:** 3 input channels, 32 output channels, kernel size 3x3, padding 1
- **Max Pooling:** 2x2
- **Conv Layer 2:** 32 input channels, 64 output channels, kernel size 3x3, padding 1
- **Fully Connected Layer 1:** 512 units
- **Fully Connected Layer 2:** 10 output units (one for each class)

## Training

The model is trained using:
- **Loss Function:** Cross Entropy Loss
- **Optimizer:** Adam with a learning rate of 0.001
- **Number of Epochs:** Configurable, default is 10

### Training Function:
```python
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
```

## Evaluation

The model is evaluated on the test set, providing the overall accuracy and accuracy for each class. 

### Evaluation Function:
```python
evaluate_model(model, test_loader)
```

### Class-wise Accuracy:
```python
evaluate_model_with_class_percentages(model, test_loader)
```

## Visualization

Predictions can be visualized using the `visualize_predictions` function, which shows the model's predictions alongside the true labels for a sample of test images.

```python
visualize_predictions(model, test_loader)
```

---
