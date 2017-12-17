"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    
    def __init__(self, input_dim=(3, 32, 32), num_filters=16, kernel_size=3,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        #######################################kernel_size#################################
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(int(channels)),
            nn.Conv2d(int(channels), int(num_filters), kernel_size=int(kernel_size), padding=int((kernel_size-1)/2), stride=int(stride_conv)),
            
            nn.ReLU())
            #nn.MaxPool2d(pool,stride=int(stride_pool)))
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(int(num_filters)),
            nn.Conv2d(int(num_filters), int(np.ceil(num_filters*2)), kernel_size=5, padding=int((kernel_size-1)/2), stride=int(stride_conv)),
            
            nn.ReLU(),
            nn.MaxPool2d(pool,stride=int(stride_pool)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(int(np.ceil(num_filters)), int(np.ceil(num_filters/2)), kernel_size=int(kernel_size), padding=int((kernel_size-1)/2), stride=int(stride_conv)),
            #nn.BatchNorm2d(int(num_filters)),
            nn.ReLU(),
            nn.MaxPool2d(pool,stride=int(stride_pool)))
        self.layer4 = nn.Sequential(
          torch.nn.Linear(7200, int(hidden_dim)),
          torch.nn.Dropout(dropout),
          torch.nn.ReLU(),
          #torch.nn.Linear(int(5*hidden_dim), int(hidden_dim)),
          torch.nn.Linear(hidden_dim, num_classes))


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        x = self.layer1(x)
        #print("layer1")
        #print(x.shape)
        x = self.layer2(x)
        #print("layer2")
        #print(x.shape)
        #x = self.layer3(x)
        #print("layer3")
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print("layer4")
        #print(x.shape)
        x = self.layer4(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
