#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:02:10 2018

@author: zhanglei
"""

from scipy import misc
import  os
import numpy as np
from sklearn.metrics import confusion_matrix
from operator import truediv





def predict_eval(predictions, label_tensor):
    pre_imgs=predictions.astype('uint8')
    gt_imgs=label_tensor.astype('uint8')
    
    
    precision=np.mean(pre_imgs==gt_imgs)
    print('global precision is:',precision)


    flat_pre_imgs=pre_imgs.flatten()
    flat_gt_imgs=gt_imgs.flatten()
    labels=[x for x in range(12)]

    confusion=confusion_matrix(flat_gt_imgs,flat_pre_imgs,labels)

    list_diag=np.diag(confusion)

    list_raw_sum=np.sum(confusion,axis=1)

    each_acc=np.nan_to_num(truediv(list_diag,list_raw_sum))
    
    print('each_acc is:',each_acc)