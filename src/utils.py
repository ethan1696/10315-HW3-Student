import numpy as np
import gzip
import matplotlib.pyplot as plt

class BatchedMNIST:
    def __init__(self, dataset="training", batch_size=32, randomize=True):
        """
        Initialize the BatchedMNIST object.
        
        Parameters:
            dataset (str): "training" or "validation"
            batch_size (int): Number of samples per batch
            randomize (bool): Whether to randomize on shuffle (for autograder consistency)
        """
        if dataset == "training":
            images_path = 'data/MNIST/raw/train-images-idx3-ubyte.gz'
            labels_path = 'data/MNIST/raw/train-labels-idx1-ubyte.gz'
        elif dataset == "validation":
            images_path = 'data/MNIST/raw/t10k-images-idx3-ubyte.gz'
            labels_path = 'data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
        else:
            raise ValueError("Dataset must be 'training' or 'validation'")
        
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)
        self.batch_size = batch_size
        self.indices = np.arange(len(self.images))
        self.randomize = randomize
        self.shuffle()

    def load_images(self, filename):
        """
        Load the images from the idx3-ubyte file format.
        """
        with gzip.open(filename, 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')  # Magic number
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            buffer = f.read(num_images * rows * cols)
            data = np.frombuffer(buffer, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
        return data

    def load_labels(self, filename):
        """
        Load the labels from the idx1-ubyte file format.
        """
        with gzip.open(filename, 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')  # Magic number
            num_labels = int.from_bytes(f.read(4), 'big')
            buffer = f.read(num_labels)
            labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels
    
    def shuffle(self):
        """
        Shuffle the dataset and re-batch it.
        """
        if self.randomize:
            np.random.shuffle(self.indices)
        self.batches = [
            (self.images[self.indices[i:i + self.batch_size]],
            self.labels[self.indices[i:i + self.batch_size]])
            for i in range(0, len(self.images), self.batch_size)
        ]

    def __getitem__(self, index):
        """
        Get the batch at the given index.
        
        Parameters:
            index (int): The batch index
        
        Returns:
            tuple: (image batch, label batch)
        """
        return self.batches[index]

    def __len__(self):
        """
        Return the total number of batches.
        """
        return len(self.batches)

def visualize_batch(image_batch, label_batch, cols=8):
    """
    Visualize a batch of MNIST images with their corresponding labels.

    Parameters:
    image_batch (numpy array): Batch of images of shape (batch_size, 28, 28).
    label_batch (numpy array): Batch of labels of shape (batch_size,).
    cols (int): Number of columns in the grid. Default is 8.
    """
    batch_size = len(image_batch)
    rows = (batch_size // cols) + int(batch_size % cols != 0)  # Calculate the number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    for i in range(batch_size):
        img = image_batch[i]
        label = label_batch[i]
        
        # Plot image
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')  # Hide axes

    # Turn off remaining empty subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def initialize_random_array(shape):
    """
    Generates a random array with the specified shape using a fixed random seed for consistent results.
    
    Parameters:
    shape (tuple of int): Desired dimensions of the output array
    
    Returns:
    numpy array: Randomly initialized array of the specified shape, containing values from a standard normal distribution (mean 0, variance 1).
    """
    # Set the random seed for reproducibility DO NOT CHANGE THIS !!
    np.random.seed(10315)

    random_array = np.random.randn(*shape)

    np.random.seed(None)
    
    return random_array

def clip_str(string, length):
    """
    Clips string to length 
    """

    return (str(string) + (" " * length))[:length]

def get_accuracy(nn_output, y):
    """
    Gets the average accuracy given the neural network output and true labels.

    Params:
        nn_output (numpy array): output class probabilities from neural network with dimensions (batch_size, num_classes)
        y (numpy array): true output labels in one-hot vector format with dimensions (batch_size, num_classes)
    
    Returns:
        float: average accuracy of the nn_output
    """
    predicted_labels = np.argmax(nn_output, axis=1)
    true_labels = np.argmax(y, axis=1)
    correct_predictions = np.sum(predicted_labels == true_labels)
    accuracy = correct_predictions / y.shape[0]

    return accuracy

def to_one_hot(y_labels, num_classes):
    """
    Converts an array of class labels into one-hot encoded vectors.

    Params:
        y_labels (numpy array): Array of class labels with shape (batch_size,).
                                Each label is an integer between 0 and num_classes-1.
        num_classes (int): The number of classes. This determines the size of the one-hot vectors.
    
    Returns:
        numpy array: A one-hot encoded array of shape (batch_size, num_classes).
                     Each row corresponds to the one-hot encoding of a label.
    """
    one_hot = np.zeros((y_labels.shape[0], num_classes))    
    one_hot[np.arange(y_labels.shape[0]), y_labels] = 1
    
    return one_hot

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the training and validation losses and accuracies over epochs.

    Args:
        train_losses (list of float): List of training loss values for each epoch.
        val_losses (list of float): List of validation loss values for each epoch.
        train_accuracies (list of float): List of training accuracy values for each epoch.
        val_accuracies (list of float): List of validation accuracy values for each epoch.

    Returns:
        None: Displays two plots, one for the loss and one for the accuracy, 
        with legends differentiating between training and validation metrics.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Losses
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()