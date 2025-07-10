# app/model.py

# Step 1
# This piece of code does the following:
# Import necessary libaries
# Downloads MNIST dataset and preprocess the data (conversion to tensors and normalization)
    # Creates train, val, test dataset and applies transforms
    # Creates train, val, test dataloaders and shuffles for train dataloader
    # Note: No 'collate_fn' is used as input pipeline is simple and standardized i.e., data format is consistent
# Builds a simple CNN model architecture
    # Model architecture:
    # Input (grayscale, channel=1): Image shape (1, 28, 28)
    # First Conv: Conv1(1, 32, 3x3) + ReLU -> intermediate featuremap: [32×26×26]
    # Second Conv: Conv2(32, 64, 3x3) + ReLU -> intermediate featuremap: [64×24×24]
    # Max Pooling: MaxPool(2x2) -> intermediate featuremap: [64×12×12]
    # Dropout for Conv (2D): 25%. Lower rate because model is still learning spatial features.
    # Flatten: [64×12×12 = 9216]. Flattens all dimensions except batch.
    # FC Layer + ReLU: (9216 to 128)
    # Dropout for FC (1D): 50%. Higher rate because dense layers tend to overfit more.
    # Final layer: (128 to 10)
    # LogSoftmax
        # Note for ReLU, Max Pooling, Dropout, and LogSoftmax:
        # ReLU: Learns non-linear patterns like edges, curves, shapes, fights vanishing gradients.
        # MaxPool: For downsampling, translation invariance.
        # Dropout: Prevents overfitting. Used after pooling. Only during training. Disable at inference time.
        # LogSoftmax: Softmax for turning logits into probabilities and Log for log-probabilities for classification and numerical stability.
# Trains the CNN model on MNIST dataset with validation monitoring and early stopping
# Saves the trained model (.pth) for inference

import torch                                                                                          # Pytorch library: Supports core functionalities such as tensors (multi-dimensional arrays), automatic differentiation (backpropagation)
import torch.nn as nn                                                                                 # This module supports predefines layers (nn.Layer, nn.Conv2d, nn.ReLU, nn.Dropout) to build neural networks and model architecture
import torch.nn.functional as F                                                                       # Provides functional interfaces for layers and activation (nn.ReLU -> F.reLU (x)). Advantage: More control and no need to store parameters (example:dropout/activation)
import torch.optim as optim                                                                           # Optimization parameters (SGD, Adam) to update model parameters during training
from torchvision import datasets                                                                      # torchvision.dataset: Provides access to popular datasets like MNIST, CIFAR-10 with built-in downloads/load utilities
from torchvision import transforms                                                                    # torchvision.transforms: To preprocess and normalize images (e.g., resizing, converting to tensors, normalizing pixel values)
from torch.utils.data import DataLoader                                                               # Dataloader class: Loads data in mini-batches, shuffles it, multiprocessing for efficiency
from torch.utils.data import random_split                                                             # Splits train dataset into train and val
import os                                                                                             # File and directory management: Interacts with operating system to handle paths, check files, save model checkpoints


os.makedirs("app", exist_ok=True)
MODEL_PATH = "app/mnist_cnn.pth"                                                                      # Stores the model's learned parameters (weight), .pth: naming convention for saving PyTorch models (.pt, .bin also work)

# Download and preprocess the dataset
def get_data_loaders(batch_size=64):                                                                  # Function: Loads the MNIST dataset, applies necessary preprocessing, and returns dataloader objects for training, val and testing. Default batch size = 64 (approx. 859 batches per epoch)
    transform = transforms.Compose([                                                                  # Compose: Creates a pipeline of image transformations
        transforms.ToTensor(),                                                                        # ToTensor: Converts PIL image (28x28 grayscale) to PyTorch tensor [1, 28, 28] and scales pixel values from [0, 255] to [0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))                                                    # Normalize image tensor: Expects tuples of floats for mean & std (1-element tuples): Substracts the mean (0.1307) and divides by std (0.3081). Note: These values are specific to MNIST dataset to help model learn faster and better.
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)          # Loads training set of MNIST (downloads it if necessary) and applies transform to every image. datasets.MNIST: built-in PyTorch dataset that loads images with corresponding labels and returns tuple (image_tensor, label)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)                         # Loads test split of MNIST (No re-download, train=False switches to test set)

    train_set, val_set = random_split(train_dataset, [55000, 5000])                                   # Splits 55,000 for train, 5,000 for validation

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)                         # Wraps train set in a DataLoader (loads data in batches, shuffles to randomize order of samples each epoch and reduce overfitting)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)                          # Wraps val set in a DataLoader (loads data in batches, no shuffle to keep the data in order for consistent evaluation)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)                      # Wraps test dataset in a DataLoader
    return train_loader, val_loader, test_loader                                                      # Returns train, val, and test data loaders for training and evaluating the model

# Model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()                                                             # Calls nn.Module.__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    # How data (input tensor 'x') flows through the model/network
    def forward(self, x):
        x = F.relu(self.conv1(x))                                                                     # Intermediate featuremap: [32×26×26]
        x = F.relu(self.conv2(x))                                                                     # Intermediate featuremap: [64×24×24]
        x = F.max_pool2d(x, 2)                                                                        # Intermediate featuremap: [64×12×12]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)                                                                       # Flattens from dimension 1 onward: [64×12×12 = 9216]
        x = F.relu(self.fc1(x))                                                                       # FC: 9216 to 128
        x = self.dropout2(x)
        x = self.fc2(x)                                                                               # FC: 128 to 10
        return F.log_softmax(x, dim=1)                                                                # x shape (batch size, num_classes), dim=1 applies softmax across class dimension

# This function is for training and saving model (supports validation loss monitoring + early stopping)
def train_model(epochs=10, patience=3):
    model = SimpleCNN()
    train_loader, val_loader, _ = get_data_loaders()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()                                                                                # Important for layers like dropout, batchnorm
        for batch_idx, (data, target) in enumerate(train_loader):                                    # Loops through each batch. data: [batch_size,1,28,28] target: [batch_size=64], values 0 to 9, batch_idx: running index (0 to 858)
            optimizer.zero_grad()                                                                    # Clears old gradient from the last step. Note: Do this before '.backward()' to avoid accumulating gradients
            output = model(data)                                                                     # Forward pass of the batch through CNN. Output shape: [64,10]
            loss = F.nll_loss(output, target)                                                        # Computes negative log-likelihood loss between model output and true labels
            loss.backward()                                                                          # Backpropagation: Computes gradient of loss wrt model weights
            optimizer.step()                                                                         # Updates model weights based on computed gradients

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

        # Validation after each epoch
        model.eval()                                                                                 # Evaluation mode: Disables dropout and batchnorm randomness
        val_loss = 0                                                                                 # Stores total loss over all validation samples
        correct = 0                                                                                  # Counts how many predictions were correct
        with torch.no_grad():                                                                        # Disables gradient computation for validation
            for data, target in val_loader:
                output = model(data)                                                                 # Output shape: [batch_size=64, 10]
                val_loss += F.nll_loss(output, target, reduction='sum').item()                       # Computes loss for current batch. 'sum': Summation of loss for each sample (not average) to compute true average at the end. '.item()': Converts tensor scalar loss to Python float
                pred = output.argmax(dim=1)                                                          # Predicted class index with highest log-probability
                correct += pred.eq(target).sum().item()                                              # '.eq': Element-wise comparison between pred and output, '.sum': Counts how many correct, '.item()': Converts the result into Python int

        # Compute val loss and val accuracy (%)
        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)                                          # Float conversion to make accuracy comparison safe
        print(f"\nVal set: Avg loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)")

        # Early stopping and model saving
        if val_loss < best_val_loss:                                                                 # Updates save model if val loss is less than best loss
            best_val_loss = val_loss
            epochs_no_improve = 0                                                                    # Resets 'epochs_no_improve'. This restarts the patience count from 0
            torch.save(model.state_dict(), MODEL_PATH)                                               # Saves model's parameters to disk (not full model architecture)
            print("Saved new best model!\n")
        else:
            epochs_no_improve += 1                                                                   # If val loss does not improve, increment 'epochs_no_improve'
            print(f"No improvement. Patience: {epochs_no_improve}/{patience}\n")
            if epochs_no_improve >= patience:                                                        # If 'epochs_no_improve' >= patience
                print("Early stopping triggered.")
                break                                                                                # Exits the training loop and use the last best saved model to disk

# This function loads the previously trained model's parameters from disk
def load_model():
    model = SimpleCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))              # Loads saved weight data from disk to cpu (even if training is done on GPU)
        model.eval()                                                                                 # Consistent, deterministic inference
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError("Trained model not found. Please run train_model() first.")
    return model

# To train the model directly
if __name__ == "__main__":
    train_model()
