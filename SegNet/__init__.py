#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:50:59 2018

@author: zhanglei
"""

import tensorflow as tf
from .data_prepare import get_file_names,get_data_label_batch
from .net import net
from .predict_eval import predict_eval

FLAGS=tf.flags.FLAGS


tf.flags.DEFINE_bool('train','True','If True train , else test ')
tf.flags.DEFINE_bool('finetune','False','If True reuse weight , else init ')
tf.flags.DEFINE_float('moving_average_decay', '0.99','The decay to use for the moving average')
tf.flags.DEFINE_integer('num_class','12','num of classes')
tf.flags.DEFINE_integer('batch_size','2','batchsize during training ')
tf.flags.DEFINE_integer('num_epoches','36700','num of epoches')
tf.flags.DEFINE_integer('img_height','360','height of imgs')
tf.flags.DEFINE_integer('img_width','480','width of imgs')
tf.flags.DEFINE_string('checkpoint_dir','checkpoint/','the dir in which model is saved ')
tf.flags.DEFINE_string('log_dir','log','summary saved')
tf.flags.DEFINE_string('data_dir','data','dataset dir')
tf.flags.DEFINE_string('pre_dir','test_result','pre result')
tf.app.flags.DEFINE_integer('num_examples_epoch_train', '367','num examples per epoch for train')
tf.app.flags.DEFINE_integer('num_examples_epoch_test', '233','num examples per epoch for test')
tf.app.flags.DEFINE_integer('num_examples_epoch_val', '101','num examples per epoch for test')
tf.app.flags.DEFINE_float('fraction_of_examples_in_queue', '0.1','Fraction of examples from datasat to put in queue. Large datasets need smaller value, otherwise memory gets full. ')