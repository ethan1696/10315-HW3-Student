import numpy as np
from .utils import *

class Transform:
    def __init__(self):
        """
        For initializing parameters/configurations for the model layer
        """
        pass
    def forward(self, x):
        """
        Forward pass for the layer, returns the output of the forward pass
        The input can be overridden 
        """
        pass
    def backward(self, grad_wrt_out):
        """
        Backward pass for the layer - calculates/records gradients for the input and the parameters
        Returns the gradients of the input
        """
        pass
    def step(self, lr):
        """
        Updates the parameters using the gradients calculated in the backward pass
        """
        pass

class Sigmoid(Transform):
    def __init__(self):
        """
        The sigmoid activation function as no parameters or configurations. Nothing to do here!
        """
        super().__init__()
    
    def forward(self, x):
        """
        Computes the sigmoid for x

        Parameters:
            x (numpy array): input to the sigmoid, with dimensions (N, M), where N is the batch size
        
        Returns:
            numpy array: output of the sigmoid, with dimensions (N, M), where N is the batch size
        """
        # STUDENT SOLUTION START
        raise NotImplementedError()
        # STUDENT SOLUTION END

    def backward(self, grad_wrt_out):
        """
        Computes gradient of the input of the forward pass, given the gradient of loss w.r.t. the output
        Note - in the forward pass, you may need to store some values in the object to implement the backward pass

        Parameters:
            grad_wrt_out (numpy array): gradient of the loss w.r.t. output, with dimensions (N, M), where N is the batch size
        
        Returns:
            numpy array: gradient of the input w.r.t. loss, with dimensions (N, M), where N is the batch size
        """
        # STUDENT SOLUTION START
        raise NotImplementedError()
        # STUDENT SOLUTION END
    
    # No step function in this layer, since there are no parameters to update

class Linear(Transform):
    def __init__(self, in_dim, out_dim):
        """
        Initializes the parameters for the model. Note that the parameters for a linear layer are the weights matrix and 
        the bias vector

        Parameters:
            in_dim (int): the dimension of the input vector to the linear layer
            out_dim (int): the dimension of the output vector to the linear layer

        IMPORTANT -- use initialize_random_array from src/utils.py to randomly initialize your arrays. 
                     You need to do this to pass the autograder.
        """
        super().__init__()
        self.W = None # Initialize this in your solution!
        self.b = None # Initialize this in your solution!
        self.grad_wrt_W = None # This will be set in the backward pass (the __init__ function shouldn't touch this)
        self.grad_wrt_b = None # This will be set in the backward pass (the __init__ function shouldn't touch this)
        # STUDENT SOLUTION START
        # Note: b should be a one-dimensional array (not 2 dimensional!)
        raise NotImplementedError()
        # STUDENT SOLUTION END
    
    def forward(self, X):
        """
        Forward pass for the linear layer

        Params:
            X (numpy array): Input to the linear layer with dimensions (N, in_dim), where N is the batch size

        Returns:
        numpy array: Output of the linear layer with dimensions (N, out_dim), where N is the batch size
        """
        # STUDENT SOLUTION START
        raise NotImplementedError()
        # STUDENT SOLUTION END
    
    def backward(self, grad_wrt_out):
        """
        Backward pass for the linear layer and stored gradients of the loss w.r.t parameters
        Note - in the forward pass, you may need to store some values in the object to implement the backward pass

        Params:
            grad_wrt_out (numpy array): gradient of the loss w.r.t. output, with dimensions (N, out_dim), where N is the batch size

        Returns:
            py array: gradient of the loss w.r.t. input, with dimensions (N, in_dim) where N is the batch size

        Also:
        Store the gradient w.r.t. W in self.grad_wrt_W, and the gradient w.r.t. b in self.grad_wrt_b
        You should store the AVERAGE of the gradients across the batch.
        """
        # STUDENT SOLUTION START
        # Note: self.grad_wrt_W and self.grad_wrt_b should have the same dimensions as self.W and self.b respectively
        raise NotImplementedError()
        # STUDENT SOLUTION END
    
    def step(self, lr):
        """
        Updates the parameters after the backward pass using gradient descent. 

        Params:
            lr (float): the learning rate for gradient descent
        """
        # STUDENT SOLUTION START
        raise NotImplementedError()
        # STUDENT SOLUTION END

    def setParams(self, W, b):
        """
        For testing and autograder -- DO NOT EDIT!!
        """
        self.W = W
        self.b = b
    
    def getParams(self):
        """
        For testing and autograder -- DO NOT EDIT!!
        """
        return self.W, self.b
    
    def getGradients(self):
        """
        For testing and autograder -- DO NOT EDIT!!
        """
        return self.grad_wrt_W, self.grad_wrt_b

class SoftmaxLoss(Transform):
    def __init__(self):
        """
        The softmax activation function and loss function have no parameters or configurations. Nothing to do here!
        """
        super().__init__()
    
    def forward(self, z, training, y=None):
        """
        Forward pass for SoftmaxLoss
        There are two modes to this function - training and validation (when the training argument is False)
            if training == True - take the softmax of z and compute the cross entropy loss (for each element of the batch) 
                and return it. That is, return L(y, Softmax(z))
            else - Take the softmax of z and return the result. That is, return Softmax(z)

        This is because when we train, we are interested in the losses of the model, but when validating, we want the model
            to output predictions (i.e. the probability distributions for the output classes)
        
        Params:
            z (numpy array): The output logits from the neural network with dimensions (N, M), where N is the batch size
            y (numpy array): A batch of one-hot vector representations of the true labels with dimensions (N, num_classes), 
                where N is the batch size. This parameter is None when the training paramter is False. 
            training (bool): Whether to run the forward pass for training or validation (details above)

        Returns:
            numpy array: if training is True, then a batch of losses with dimensions (N,), 
                otherwise a batch of probability distributions for the output classes with dimensions (N, num_classes), 
                where N is the batch size
        """
        ### STUDENT SOLUTION START
        raise NotImplementedError()
        ### STUDENT SOLUTION END
    
    def backward(self):
        """
        Backward pass for softmax loss
        Note - in the forward pass, you may need to store some values in the object to implement the backward pass
        The backward function should never be called outside of training

        Returns:
            numpy array: gradient of loss w.r.t. the input z
        """
        ### STUDENT SOLUTION START
        raise NotImplementedError()
        ### STUDENT SOLUTION END
    
    # No step function in this layer, since there are no parameters to update

class NeuralNetwork_2HL(Transform):
    def __init__(self):
        """
        Instantiate all your layers here! 
        """
        ### STUDENT SOLUTION START
        raise NotImplementedError()
        ### STUDENT SOLUTION END
    
    def forward(self, X, training, y=None):
        """
        Forward pass for the neural network

        Params: 
            X (numpy array): A batch of inputs (in this case, it would be 784 dimensional MNIST images) with dimensions (N, 784), 
                where N is the batch size
            training (bool): Whether to run the forward pass for training or validation - this controls whether the output is a batch 
                of losses or a batch of class probabilities (see the doc string for the forward function of SoftmaxLoss for more details)
            y (numpy array): A batch of one-hot vector representations of the true labels with dimensions (N, num_classes), 
                where N is the batch size. This parameter is None when the training paramter is False. 
        Returns:
            numpy array: if training is True, then a batch of losses with dimensions (N,), 
                otherwise a batch of probability distributions for the output classes with dimensions (N, num_classes), 
                where N is the batch size
        """
        ### STUDENT SOLUTION START
        raise NotImplementedError()
        ### STUDENT SOLUTION END

    def backward(self):
        """
        Backward pass for the neural network 
        
        Returns:
            numpy array: gradient of the loss w.r.t. input, with dimensions (N, 784) where N is the batch size
        """
        ### STUDENT SOLUTION START
        raise NotImplementedError()
        ### STUDENT SOLUTION END
    
    def step(self, lr):
        """
        Updates the parameters after the backward pass using gradient descent. 

        Params:
            lr (float): the learning rate for gradient descent
        """
        ### STUDENT SOLUTION START
        raise NotImplementedError()
        ### STUDENT SOLUTION END

def train_NN(neural_network, lr, batch_size, num_epochs, train_batches_per_epoch=150, val_batches_per_epoch=20, randomize=True, verbose=True):
    """
    Trains the neural network and records training metrics

    Params:
        neural_network (Transform): untrained neural network. This network will be trained in-place
        lr (float): learning rate
        batch_size (int): batch size
        num_epochs (int): number of epochs to train for
        train_batches_per_epoch (int): number of training batches to use per epoch
        val_batches_per_epoch (int): number of validation batches to use per epoch
        randomize (bool): whether to randomize when shuffling (this is only set to False for auto-grader consistency; 
            you do not need to worry about this field)

    Returns:
        float list: training losses - array of average training losses for each epoch
        float list: validation losses - array of average validation losses for each epoch
        float list: training accuracies - array of average training accuracy
        float list: validation accuracies - array of average validation accuracy
    """

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    # Initialize datasets (these are custom-defined for you for grading purposes)
    batched_dataset_training = BatchedMNIST(dataset="training", batch_size=batch_size, randomize=randomize)
    batched_dataset_validation = BatchedMNIST(dataset="validation", batch_size=batch_size, randomize=randomize)
    for epoch_num in range(num_epochs):
        batched_dataset_training.shuffle()

        # For training
        total_training_loss = 0
        total_training_accuracy = 0
        for X_image, y_labels in batched_dataset_training[:train_batches_per_epoch]:
            X = X_image.reshape(batch_size, 784)
            y = to_one_hot(y_labels, 10)
            avg_training_loss = None # Average training loss across this batch -- SET THIS IN YOUR SOLUTION!
            avg_training_accuracy = None # Average training accuracy accross this batch -- SET THIS IN YOUR SOLUTION!

            # - In this section, run the forward pass, backward pass, and step function for the neural network on the training batch
            # - Store the average training loss and average training accuracy in the variables defined above. 
            # - You may need to run the forward pass again to compute the training accuracies
            # - You may use the get_accuracy function in src/utils.py to compute the accuracy of your neural network accuracy
            # - The input batch and one-hot-vector labels are in the X and y variables, respectively
            # - Note: Make sure you compute the accuracy AFTER updating the parameters on the neural network for this batch
            ### STUDENT SOLUTION START
            raise NotImplementedError()
            ### STUDENT SOLUTION END

            total_training_loss += avg_training_loss
            total_training_accuracy += avg_training_accuracy
        training_losses.append(total_training_loss / train_batches_per_epoch)
        training_accuracies.append(total_training_accuracy / train_batches_per_epoch)

        # For validation
        total_validation_loss = 0
        total_validation_accuracy = 0
        for X_image, y_labels in batched_dataset_validation[:val_batches_per_epoch]:
            X = X_image.reshape(batch_size, 784)
            y = to_one_hot(y_labels, 10)
            avg_validation_loss = None # Average validation loss across this batch -- SET THIS IN YOUR SOLUTION!
            avg_validation_accuracy = None # Average validation accuracy accross this batch -- SET THIS IN YOUR SOLUTION!

            # - In this section, compute the averate validation loss and average validation accuracy on the validation set
            # - Store the average validation loss and average validation accuracy in the variables defined above. 
            # - Note that you should NOT be updating parameters with the validation data here
            # - You may use the get_accuracy function in src/utils.py to get the accuracy of your neural network accuracy
            # - The input batch and one-hot-vector labels are in the X and y variables, respectively
            ### STUDENT SOLUTION START
            raise NotImplementedError()
            ### STUDENT SOLUTION END

            total_validation_loss += avg_validation_loss
            total_validation_accuracy += avg_validation_accuracy
        
        validation_losses.append(total_validation_loss / val_batches_per_epoch)
        validation_accuracies.append(total_validation_accuracy / val_batches_per_epoch)
        if verbose:
            print(
                f"Epoch {clip_str(epoch_num, 3)} | "
                f"Training loss {clip_str(training_losses[-1], 6)} | "
                f"Validation loss {clip_str(validation_losses[-1], 6)} | "
                f"Training accuracy {clip_str(training_accuracies[-1], 6)} | "
                f"Validation accuracy {clip_str(validation_accuracies[-1], 6)}"
            )
        
    return training_losses, validation_losses, training_accuracies, validation_accuracies






