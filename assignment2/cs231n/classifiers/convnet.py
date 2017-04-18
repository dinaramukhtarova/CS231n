import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ThreeLayerConvNet1(object):
    '''
    [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax]
    '''
    def __init__(self, num_filters, hidden_dims, filter_size=7, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.N = len(num_filters) - 1
        self.M = len(hidden_dims) + 1
        
        N, M = self.N, self.M
        self.params['W1'] = np.random.randn(num_filters[0], input_dim[0], filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters[0])
        for i in xrange(N):
            self.params['W'+str(i+2)] = np.random.randn(num_filters[i+1], num_filters[i], filter_size, filter_size) * weight_scale
            self.params['b'+str(i+2)] = np.zeros(num_filters[i+1])
        self.params['W'+str(N+2)] = np.random.randn(num_filters[N] * (input_dim[1] / 2 ** N) * (input_dim[2] / 2 ** N), hidden_dims[0]) * weight_scale
        self.params['b'+str(N+2)] = np.zeros(hidden_dims[0])
        for i in xrange(M - 2):
            self.params['W'+str(i+N+3)] = np.random.randn(hidden_dims[i], hidden_dims[i+1]) * weight_scale
            self.params['b'+str(i+N+3)] = np.zeros(hidden_dims[i+1])
        self.params['W'+str(M+N+1)] = np.random.randn(hidden_dims[M-2], num_classes) * weight_scale
        self.params['b'+str(M+N+1)] = np.zeros(num_classes)
        
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
            
            
            
    def loss(self, X, y=None):
    
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
        N, M = self.N, self.M

        scores = None
        cache = []
        out = X
        for i in xrange(N):
            out, cache_cur = conv_relu_pool_forward(out, self.params['W'+str(i+1)], self.params['b'+str(i+1)], conv_param, pool_param)
            cache.append(cache_cur)
        out, cache_cur = conv_relu_forward(out, self.params['W'+str(N+1)], self.params['b'+str(N+1)], conv_param)
        cache.append(cache_cur)
        for i in range(M):
            out, cache_cur = affine_forward(out, self.params['W'+str(i+N+2)], self.params['b'+str(i+N+2)])
            cache.append(cache_cur)
        scores = out
    
        if y is None:
          return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        for i in range(M + N + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)] ** 2)
        dout = dscores
        for i in range(M):
            dout, grads['W'+str(M+N+1-i)], grads['b'+str(M+N+1-i)] = affine_backward(dout, cache.pop())
        dout, grads['W'+str(N+1)], grads['b'+str(N+1)] = conv_relu_backward(dout, cache.pop())
        for i in range(N):
            dout, grads['W'+str(N-i)], grads['b'+str(N-i)] = conv_relu_pool_backward(dout, cache.pop())
        for i in range(M + N + 1):
            grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
    
        return loss, grads
    
    
    
class ThreeLayerConvNet2(object):
    '''
    [conv-relu-pool]XN - [affine]XM - [softmax]
    '''
    def __init__(self, num_filters, hidden_dims, filter_size=7, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.N = len(num_filters)
        self.M = len(hidden_dims) + 1
        
        N, M = self.N, self.M
        self.params['W1'] = np.random.randn(num_filters[0], input_dim[0], filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters[0])
        for i in xrange(N - 1):
            self.params['W'+str(i+2)] = np.random.randn(num_filters[i+1], num_filters[i], filter_size, filter_size) * weight_scale
            self.params['b'+str(i+2)] = np.zeros(num_filters[i+1])
        self.params['W'+str(N+1)] = np.random.randn(num_filters[N-1] * (input_dim[1] / 2 ** N) * (input_dim[2] / 2 ** N), hidden_dims[0]) * weight_scale
        self.params['b'+str(N+1)] = np.zeros(hidden_dims[0])
        for i in xrange(M - 2):
            self.params['W'+str(i+N+2)] = np.random.randn(hidden_dims[i], hidden_dims[i+1]) * weight_scale
            self.params['b'+str(i+N+2)] = np.zeros(hidden_dims[i+1])
        self.params['W'+str(M+N)] = np.random.randn(hidden_dims[M-2], num_classes) * weight_scale
        self.params['b'+str(M+N)] = np.zeros(num_classes)
        
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
            
            
    def loss(self, X, y=None):
    
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
        N, M = self.N, self.M

        scores = None
        cache = []
        out = X
        for i in xrange(N):
            out, cache_cur = conv_relu_pool_forward(out, self.params['W'+str(i+1)], self.params['b'+str(i+1)], conv_param, pool_param)
            cache.append(cache_cur)
        for i in range(M):
            out, cache_cur = affine_forward(out, self.params['W'+str(i+N+1)], self.params['b'+str(i+N+1)])
            cache.append(cache_cur)
        scores = out
    
        if y is None:
          return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        for i in range(M + N):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)] ** 2)
        dout = dscores
        for i in range(M):
            dout, grads['W'+str(M+N-i)], grads['b'+str(M+N-i)] = affine_backward(dout, cache.pop())
        for i in range(N):
            dout, grads['W'+str(N-i)], grads['b'+str(N-i)] = conv_relu_pool_backward(dout, cache.pop())
        for i in range(M + N):
            grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
    
        return loss, grads
    
    
    
class ThreeLayerConvNet3(object):
    '''
    [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax]
    '''
    def __init__(self, num_filters, hidden_dims, filter_size=7, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.N = len(num_filters) / 2
        self.M = len(hidden_dims) + 1
        
        N, M = self.N, self.M
        self.params['W1'] = np.random.randn(num_filters[0], input_dim[0], filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters[0])
        for i in xrange(2, 2*N+1):
            self.params['W'+str(i)] = np.random.randn(num_filters[i-1], num_filters[i-2], filter_size, filter_size) * weight_scale
            self.params['b'+str(i)] = np.zeros(num_filters[i-1])
        self.params['W'+str(2*N+1)] = np.random.randn(num_filters[2*N-1] * (input_dim[1] / 2 ** N) * (input_dim[2] / 2 ** N), hidden_dims[0]) * weight_scale
        self.params['b'+str(2*N+1)] = np.zeros(hidden_dims[0])
        for i in xrange(2, M):
            self.params['W'+str(i+2*N)] = np.random.randn(hidden_dims[i-2], hidden_dims[i-1]) * weight_scale
            self.params['b'+str(i+2*N)] = np.zeros(hidden_dims[i-1])
        self.params['W'+str(M+2*N)] = np.random.randn(hidden_dims[M-2], num_classes) * weight_scale
        self.params['b'+str(M+2*N)] = np.zeros(num_classes)
        
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
            
    def loss(self, X, y=None):
    
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
        N, M = self.N, self.M

        scores = None
        cache = []
        out = X
        for i in xrange(1, N+1):
            out, cache_cur = conv_relu_forward(out, self.params['W'+str(2*i-1)], self.params['b'+str(2*i-1)], conv_param)
            cache.append(cache_cur)
            out, cache_cur = conv_relu_pool_forward(out, self.params['W'+str(2*i)], self.params['b'+str(2*i)], conv_param, pool_param)
            cache.append(cache_cur)
        for i in xrange(1, M+1):
            out, cache_cur = affine_forward(out, self.params['W'+str(i+2*N)], self.params['b'+str(i+2*N)])
            cache.append(cache_cur)
        scores = out
    
        if y is None:
          return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        for i in xrange(1, M+2*N+1):
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i)] ** 2)
        dout = dscores
        for i in xrange(M):
            dout, grads['W'+str(M+2*N-i)], grads['b'+str(M+2*N-i)] = affine_backward(dout, cache.pop())
        for i in xrange(N):
            dout, grads['W'+str(2*N-2*i)], grads['b'+str(2*N-2*i)] = conv_relu_pool_backward(dout, cache.pop())
            dout, grads['W'+str(2*N-2*i-1)], grads['b'+str(2*N-2*i-1)] = conv_relu_backward(dout, cache.pop())
        for i in xrange(1, M+2*N+1):
            grads['W'+str(i)] += self.reg * self.params['W'+str(i)]
    
        return loss, grads