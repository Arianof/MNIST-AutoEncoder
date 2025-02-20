# MNIST Autoencoder

This repository contains a simple convolutional autoencoder implemented in PyTorch for reconstructing images from the MNIST dataset. The model learns to compress and reconstruct handwritten digit images using convolutional and transposed convolutional layers.

## Features
- Loads the MNIST dataset.
- Implements an **Encoder** to reduce image dimensionality.
- Implements a **Decoder** to reconstruct images.
- Trains using **Mean Squared Error (MSE)** loss and the **Adam optimizer**.
- Displays sample reconstructions before and after training.

## Model Architecture
### Encoder
The encoder consists of a series of convolutional layers that progressively downsample the input image:
- Conv2D (1 → 8 channels, kernel=15, stride=1, padding=1) → Output: 8×16×16
- Conv2D (8 → 16 channels, kernel=3, stride=2, padding=1) → Output: 16×8×8
- Conv2D (16 → 32 channels, kernel=3, stride=2, padding=1) → Output: 32×4×4
- Conv2D (32 → 64 channels, kernel=5, stride=2, padding=1) → Output: 64×1×1

### Decoder
The decoder reverses the encoding process using transposed convolutions:
- ConvTranspose2D (64 → 32 channels, kernel=4, stride=1, padding=0) → Output: 32×4×4
- ConvTranspose2D (32 → 16 channels, kernel=2, stride=2, padding=0) → Output: 16×8×8
- ConvTranspose2D (16 → 8 channels, kernel=2, stride=2, padding=0) → Output: 8×16×16
- ConvTranspose2D (8 → 1 channel, kernel=4, stride=2, padding=3) → Output: 1×28×28

## Training
- The model is trained for **15 epochs** using a batch size of **32**.
- Uses **Mean Squared Error (MSE)** loss function.
- Optimized using the **Adam optimizer** with a learning rate of **1e-2**.
- Training loop iterates over batches, performing forward and backward propagation to minimize reconstruction error.

## Results
### Before Training
The untrained model produces blurry or random reconstructions of MNIST digits.

### After Training
The trained autoencoder successfully reconstructs MNIST images with high fidelity.

## How to Use
### Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mnist-autoencoder.git
   cd mnist-autoencoder
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```
3. Run the script to train and visualize the results:
   ```bash
   python autoencoder.py
   ```

## Future Improvements
- Add batch normalization and dropout for better generalization.
- Experiment with different architectures (e.g., Variational Autoencoders).
- Extend to color images and other datasets.
