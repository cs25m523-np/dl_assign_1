# dl_assign_1
Deep learning assignment 1

# VGG6 on CIFAR-10
This project trains a VGG6 convolutional neural network on CIFAR-10 and uses Weights & Biases (W&B) to track experiment results.
The script compares:
- Five activation functions: ReLU, Sigmoid, Tanh, SiLU, and GELU. These runs use Adam.
- Six optimizers: SGD, Nesterov SGD, Adam, Adagrad, RMSprop, and NAdam. These runs use ReLU.

These are 11 separate experiments, not all activation–optimizer pairing is done due to time contraints.

For training, it uses random crop, random horizontal flip, CIFAR-10 normalization, and Cutout. It reserves 5,000 training images for validation and evaluates on CIFAR-10's test set.

# Install dependencies
Install Python and the required packages.

Then install the other packages:
```bash
python -m pip install wandb numpy pillow matplotlib
