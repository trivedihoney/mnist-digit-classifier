# MNIST Digit Classifier

## Description
Running this code will create a model using PyTorch that can detect hand-written digits with approximately 97.5% accuracy.

## Project Structure
- `pytorch_modules/`: Helper functions to make the codebase modular.
- `scripts/`: Python scripts for models and model training.
- `models/`: Output path for trained models.
- `scripts/models`: Path for `nn.Module` classes that create the model.
- `scripts/train_mnist.py`: Running this python file trains the model.

## Setup Instructions

Note : UV is required for installing necessary dependancies automatically.

1. Clone the repository:
   ```sh
   git clone https://github.com/trivedihoney/mnist-digit-classifier.git
   cd mnist-digit-classifier
   uv sync

2. Run main.py

## Current Setup
`MNISTModelv0` consists of 3 linear layers along with ReLU activation. This model achieves approximately 95% accuracy on test set.

`MNISTConvModelv0` consists of 2 Convolutional layers having kernel size 3 and padding 1. This model achieves approximately 97.5% accuracy on test set.

Hyperparameters can be configured in main.py

### `MNISTConvModelv0` Hyperparameters

   epochs = 20

   batch_size = 32

   learning_rate = 0.01

   optimizer = "SGD"

   loss = "CrossEntropyLoss"
