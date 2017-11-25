import tensorflow as tf
import numpy as np


class Model:
    
    def __init__(self,
                 n_variables,
                 learning_rate):
        
        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, n_variables])
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        
    
        # Create model
    def build_net(self,
                  learning_rate,
                  convolution_levels,
                  weights_shape,
                  bias_shape):

        
        
        
        weights = {}
        bias = {}
        
        out = self.X
        for i in range(convolution_levels):
            
            w = 'wfc' + str(i)
            weights[w] = tf.get_variable(w, 
                                         shape = weights_shape[i], 
                                         dtype = tf.float32, 
                                         initializer = tf.contrib.layers.xavier_initializer())    
        
            b = 'bfc' + str(i)
            bias[b] = tf.get_variable(b, 
                                   shape = bias_shape[i], 
                                   dtype = tf.float32, 
                                   initializer = tf.constant_initializer(0.0))
            
            linear_out = tf.matmul(out, weights[w]) + bias[b]
            
            if i < convolution_levels -1:
                out = tf.nn.relu(linear_out)
            else:
                out = tf.nn.sigmoid(linear_out)
            
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y,
                                                                               logits = linear_out))
        
        self.training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
                                                
        tf.summary.scalar('cost', self.cost)
             
        self.out = out
