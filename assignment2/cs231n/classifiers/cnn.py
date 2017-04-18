import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(num_filters * (input_dim[1] / 2) * (input_dim[2] / 2), hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    crp, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    ar, ar_cache = affine_relu_forward(crp, W2, b2)
    scores, scores_cache = affine_forward(ar, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
    loss += 0.5 * self.reg * np.sum(self.params['W2'] ** 2)
    loss += 0.5 * self.reg * np.sum(self.params['W3'] ** 2)
    dar, dW3, db3 = affine_backward(dscores, scores_cache)
    dcrp, dW2, db2 = affine_relu_backward(dar, ar_cache)
    dX, dW1, db1 = conv_relu_pool_backward(dcrp, crp_cache)
    dW1 += self.reg * self.params['W1']
    dW2 += self.reg * self.params['W2']
    dW3 += self.reg * self.params['W3']
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass


from cs231n.classifiers.fc_net import *

class ThreeLayerConvNetBatchNorm(object):
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(num_filters * (input_dim[1] / 2) * (input_dim[2] / 2), hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    
    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    mode = 'test' if y is None else 'train'
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
    bn_param = [{'mode': mode} for i in 1, 2]

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    c, c_cache = conv_forward_fast(X, W1, b1, conv_param)
    sbn, sbn_cache = spatial_batchnorm_forward(c, gamma1, beta1, bn_param[0])
    r, r_cache = relu_forward(sbn)
    mp, mp_cache = max_pool_forward_fast(r, pool_param)
    abnr, abnr_cache = affine_batchnorm_relu_forward(mp, W2, b2, gamma2, beta2, bn_param[1])
    scores, scores_cache = affine_forward(abnr, W3, b3)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(self.params['W1'] ** 2)
    loss += 0.5 * self.reg * np.sum(self.params['W2'] ** 2)
    loss += 0.5 * self.reg * np.sum(self.params['W3'] ** 2)
    dabnr, dW3, db3 = affine_backward(dscores, scores_cache)
    dmp, dW2, db2, dgamma2, dbeta2 = affine_batchnorm_relu_backward(dabnr, abnr_cache)
    dr = max_pool_backward_fast(dmp, mp_cache)
    dsbn = relu_backward(dr, r_cache)
    dc, dgamma1, dbeta1 = spatial_batchnorm_backward(dsbn, sbn_cache)
    dX, dW1, db1 = conv_backward_fast(dc, c_cache)
    dW1 += self.reg * self.params['W1']
    dW2 += self.reg * self.params['W2']
    dW3 += self.reg * self.params['W3']
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['gamma1'] = dgamma1
    grads['beta1'] = dbeta1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['gamma2'] = dgamma2
    grads['beta2'] = dbeta2
    grads['W3'] = dW3
    grads['b3'] = db3
    
    return loss, grads
  
  
pass


