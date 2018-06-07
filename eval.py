#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:20:29 2018

@author: zhanglei
"""

from scipy import misc
import  os
import numpy as np
from sklearn.metrics import confusion_matrix
from operator import truediv


predict_file_names=os.listdir('test_result/pre/')
predict_file_names=['test_result/pre/'+x for x in predict_file_names]

gt_file_names=os.listdir('test_result/gt/')
gt_file_names=['test_result/gt/'+x for x in gt_file_names]

size=len(predict_file_names)
img_height=360
img_width=480

pre_imgs=np.zeros((size,img_height,img_width))
gt_imgs=np.zeros((size,img_height,img_width))


for i in range(size):
    temp_pre=misc.imread(predict_file_names[i])
    temp_gt=misc.imread(gt_file_names[i])
    pre_imgs[i]=temp_pre
    gt_imgs[i]=temp_gt


pre_imgs=pre_imgs.astype('uint8')
gt_imgs=gt_imgs.astype('uint8')

precision=np.mean(pre_imgs==gt_imgs)
print('global precision is:',precision)


flat_pre_imgs=pre_imgs.flatten()
flat_gt_imgs=gt_imgs.flatten()
labels=[x for x in range(12)]

confusion=confusion_matrix(flat_gt_imgs,flat_pre_imgs,labels)

list_diag=np.diag(confusion)

list_raw_sum=np.sum(confusion,axis=1)

each_acc=np.nan_to_num(truediv(list_diag,list_raw_sum))





