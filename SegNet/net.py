#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:59:53 2018

@author: zhanglei
"""
import tensorflow as tf
import numpy as np
FLAGS = tf.flags.FLAGS




################################net############################    
def conv_bias_with_bn(tensor,shape,relu,name):
    with tf.variable_scope(name):
        kernel=tf.get_variable('kernel',shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())##############tf.contrib.layers.variance_scaling_initializer()########
        conv=tf.nn.conv2d(tensor,kernel,strides=[1,1,1,1],padding='SAME')
        bias=tf.Variable(initial_value=tf.constant(0.0,shape=[shape[-1]]),dtype=tf.float32,name='bias')
        bias_out=tf.nn.bias_add(conv,bias)
        
        ########train  or test
        if relu==True:
            return tf.nn.relu(tf.contrib.layers.batch_norm(bias_out,is_training=FLAGS.train,center=False,decay=FLAGS.moving_average_decay))
        if relu==False:
            return tf.contrib.layers.batch_norm(bias_out,is_training=FLAGS.train,center=False,decay=FLAGS.moving_average_decay)


#########################    
def unpool_with_argmax(pool, ind, output_shape,name = None, ksize=[1, 2, 2, 1]):

    '''
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the SAME as for the pool
       Return:
           unpool:    unpooling tensor
    '''    

    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()        
        flat_input_size = np.prod(input_shape)
        ind_=tf.cast(tf.reshape(ind,[flat_input_size,1]),tf.int32)
        pool_=tf.reshape(pool,[flat_input_size])
        flat_output_shape =tf.constant([ output_shape[0]*output_shape[1] * output_shape[2] * output_shape[3]])
        ret= tf.scatter_nd(ind_, pool_, flat_output_shape)
        ret = tf.reshape(ret, output_shape)

    return ret

def net(images,labels):    
    norm1=tf.nn.lrn(images,alpha=0.0001,beta=0.5,name='norm1')
    conv1=conv_bias_with_bn(norm1,[7,7,3,64],True,'conv1')
    pool1,indice1=tf.nn.max_pool_with_argmax(conv1,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool1')
    
    conv2=conv_bias_with_bn(pool1,[7,7,64,64],True,'conv2')
    pool2,indice2=tf.nn.max_pool_with_argmax(conv2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool2')
    
    
    conv3=conv_bias_with_bn(pool2,[7,7,64,64],True,'conv3')
    pool3,indice3=tf.nn.max_pool_with_argmax(conv3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool3')
    
    
    conv4=conv_bias_with_bn(pool3,[7,7,64,64],True,'conv4')
    pool4,indice4=tf.nn.max_pool_with_argmax(conv4,[1,2,2,1],[1,2,2,1],padding='SAME',name='pool4')
    
    
    
    
    upsample4=unpool_with_argmax(pool4, indice4,conv4.get_shape().as_list(), name ='upsample4', ksize=[1, 2, 2, 1])
    #upsample4=conv4
    deconv4=conv_bias_with_bn(upsample4,[7,7,64,64],False,'deconv4')
    
    upsample3=unpool_with_argmax(deconv4, indice3,conv3.get_shape().as_list(), name ='upsample3', ksize=[1, 2, 2, 1])
    #upsample3=conv3
    deconv3=conv_bias_with_bn(upsample3,[7,7,64,64],False,'deconv3')
    
    upsample2=unpool_with_argmax(deconv3, indice2,conv2.get_shape().as_list(), name ='upsample2', ksize=[1, 2, 2, 1])
    #upsample2=conv2
    deconv2=conv_bias_with_bn(upsample2,[7,7,64,64],False,'deconv2')
    
    
    upsample1=unpool_with_argmax(deconv2, indice1,conv1.get_shape().as_list(), name ='upsample1', ksize=[1, 2, 2, 1])
    #upsample1=conv1
    deconv1=conv_bias_with_bn(upsample1,[7,7,64,64],False,'deconv1')
    
    with tf.variable_scope('classifier'):
        kernel=tf.get_variable('kernel',shape=[1, 1, 64, FLAGS.num_class],dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())##############tf.contrib.layers.variance_scaling_initializer()########
        conv=tf.nn.conv2d(deconv1,kernel,strides=[1,1,1,1],padding='SAME')
        bias=tf.Variable(initial_value=tf.constant(0.0,shape=[FLAGS.num_class]),dtype=tf.float32,name='bias')
        bias_out=tf.nn.bias_add(conv,bias)
    
    
    pre=tf.arg_max(bias_out,3)
    pre=tf.cast(pre,dtype=tf.float32)
    pre=tf.expand_dims(pre,-1)
    
    tf.summary.image('pre',pre)
    tf.summary.image('gt',tf.cast(labels,tf.uint8))
    tf.summary.image('img',images)
    return pre,bias_out