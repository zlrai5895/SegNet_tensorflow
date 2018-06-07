#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:50:33 2018

@author: zhanglei
"""

import tensorflow as tf
import numpy as np
import logging
import datetime
from scipy import misc
import SegNet







FLAGS = tf.flags.FLAGS



def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level= logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
  
  
initLogging()





##################data_pre####################
train_imgs_dir,train_labels_dir=SegNet.get_file_names(FLAGS.data_dir,'train')
val_imgs_dir,val_labels_dir=SegNet.get_file_names(FLAGS.data_dir,'val')
test_imgs_dir,test_labels_dir=SegNet.get_file_names(FLAGS.data_dir,'test')


##########################train#########################################    
def train():
    train_imgs,train_labels=SegNet.get_data_label_batch(train_imgs_dir,train_labels_dir)
    val_imgs,val_labels=SegNet.get_data_label_batch(val_imgs_dir,val_labels_dir)
    
    imgs=tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,FLAGS.img_height,FLAGS.img_width,3],name='images')
    labels=tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.img_height,FLAGS.img_width,1],name='labels')
    pre,logits=SegNet.net(imgs,labels)
    
    #precision=tf.reduce_mean(tf.to_int32(tf.equal(tf.cast(pre,tf.int32),labels)))############
    #tf.summary.scalar('precision',precision)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels,squeeze_dims=3),logits=logits))
        tf.summary.scalar('loss',loss)
        merge_op=tf.summary.merge_all()
        train_op=tf.train.AdamOptimizer(0.001).minimize(loss)
    with tf.Session() as sess:
        train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train')
        val_writer=tf.summary.FileWriter(FLAGS.log_dir+'/val')
        
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        saver=tf.train.Saver()
        if FLAGS.finetune==True:
            ckpt=tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
    
    
    
        try:
            for ep in range(5350,FLAGS.num_epoches):
                train_imgs_batch,train_labels_batch=sess.run([train_imgs,train_labels])
    
                _,summary,train_loss,train_predict_value=sess.run([train_op,merge_op,loss,pre],feed_dict={imgs:train_imgs_batch,labels:train_labels_batch})
                train_precision=np.mean(train_predict_value==train_labels_batch)
                if ep%300==0:
                    print('\r|'+'>'*40+'|'+'%d/%d' %(300,300)+'  current loss is=%f,precision=%f' %(train_loss,train_precision),end=' ')
                else:
                    trained_part=int((ep%300)/300*40)
                    rest_part=40-int((ep%300)/300*40)
                    #print(np.unique(train_predict_value))
                    print('\r|'+'>'*trained_part+' '*rest_part+'|'+'%d/%d' %(ep%300,300)+'  current loss is=%f,precision=%f' %(train_loss,train_precision),end=' ')
                if ep%10==0:
                    train_writer.add_summary(summary,ep)
                    train_writer.flush()
                    
                if ep%300==0:
                    val_imgs_batch,val_labels_batch=sess.run([val_imgs,val_labels])
                    _,summary,val_loss,val_predict_value=sess.run([train_op,merge_op,loss,pre],feed_dict={imgs:val_imgs_batch,labels:val_labels_batch})
                    val_precision=np.mean(val_predict_value==val_labels_batch)
                    val_writer.add_summary(summary,global_step=ep)
                    val_writer.flush()
                    #print(np.unique(val_predict_value))
                    logging.info('step %d:  the val_loss is %f, precision is %f' % (ep,val_loss,val_precision))
                    logging.info('>>%s Saving in %s' % (datetime.datetime.now(), FLAGS.checkpoint_dir))
                    saver.save(sess,FLAGS.checkpoint_dir,global_step=ep)
                    SegNet.predict_eval(val_predict_value,val_labels_batch)
                    
                if ep%1000==99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merge_op, train_op],feed_dict={imgs:train_imgs_batch,labels:train_labels_batch},options=run_options,run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % ep)    
    
            train_writer.close()
            val_writer.close()
            coord.request_stop()
            coord.join(threads)
        except KeyboardInterrupt :
                print('Being interrupted')
                logging.info('>>%s Saving in %s' % (datetime.datetime.now(), FLAGS.checkpoint_dir))
                saver.save(sess,FLAGS.checkpoint_dir+'/model.ckpt',ep)
        finally:
                train_writer.close()
                val_writer.close()
                coord.request_stop()
                coord.join(threads)
    

def save_result(pre_batch,gt_batch,ep):
    img_shape=gt_batch.shape
    for i in range(img_shape[0]):
        temp_pre=pre_batch[i].reshape([FLAGS.img_height,FLAGS.img_width])
        temp_gt=gt_batch[i].reshape([FLAGS.img_height,FLAGS.img_width])
        misc.imsave(FLAGS.pre_dir+'/pre/pre'+str(ep*FLAGS.batch_size+i)+'.jpg',temp_pre)
        misc.imsave(FLAGS.pre_dir+'/gt/gt'+str(ep*FLAGS.batch_size+i)+'.jpg',temp_gt)
        temp_pre,temp_gt=gray_to_rgb(temp_pre,temp_gt)
        misc.imsave(FLAGS.pre_dir+'/visual_pre/pre'+str(ep*FLAGS.batch_size+i)+'.jpg',temp_pre)
        misc.imsave(FLAGS.pre_dir+'/visual_gt/gt'+str(ep*FLAGS.batch_size+i)+'.jpg',temp_gt)


def gray_to_rgb(temp_pre,temp_gt):
    r=np.zeros_like(temp_pre)
    g=np.zeros_like(temp_pre)
    b=np.zeros_like(temp_pre)
    
    r_gt=np.zeros_like(temp_pre)
    g_gt=np.zeros_like(temp_pre)
    b_gt=np.zeros_like(temp_pre)
    
    
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    
    
    
    
    
    
    label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[temp_pre==l] = label_colours[l,0]
        g[temp_pre==l] = label_colours[l,1]
        b[temp_pre==l] = label_colours[l,2]
        r_gt[temp_gt==l] = label_colours[l,0]
        g_gt[temp_gt==l] = label_colours[l,1]
        b_gt[temp_gt==l] = label_colours[l,2]
    
    rgb = np.zeros((temp_pre.shape[0], temp_pre.shape[1], 3))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    rgb_gt = np.zeros((temp_pre.shape[0], temp_pre.shape[1], 3))
    rgb_gt[:,:,0] = r_gt
    rgb_gt[:,:,1] = g_gt
    rgb_gt[:,:,2] = b_gt
    
    return rgb,rgb_gt
    
   
    
def test():
    test_imgs,test_labels=SegNet.get_data_label_batch(test_imgs_dir,test_labels_dir)
    
    imgs=tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,FLAGS.img_height,FLAGS.img_width,3],name='images')
    labels=tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.img_height,FLAGS.img_width,1],name='labels')
    pre,logits=SegNet.net(imgs,labels)
    
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels,squeeze_dims=3),logits=logits))
        test_op=tf.train.AdamOptimizer(0.001).minimize(loss)
        
        
    with tf.Session() as sess:        
        saver=tf.train.Saver()
#        ckpt=tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#        saver.restore(sess,ckpt.model_checkpoint_path)
        saver.restore(sess, 'checkpoint/model.ckpt-5400')
        
        
        coord=tf.train.Coordinator()   #创建一个协调器，管理线程
        threads=tf.train.start_queue_runners(sess=sess,coord=coord) #启动QueueRunner, 此时文件名队列已经进队
        test_precison=0
        for ep in range(int(FLAGS.num_examples_epoch_test/FLAGS.batch_size)):
            test_imgs_batch,test_labels_batch=sess.run([test_imgs,test_labels])
            _,test_loss,test_predict_value=sess.run([test_op,loss,pre],feed_dict={imgs:test_imgs_batch,labels:test_labels_batch})
            save_result(test_predict_value,test_labels_batch,ep)
            temp_test_precision = np.mean(test_predict_value==test_labels_batch)
            test_precison=(test_precison*ep+temp_test_precision)/(ep+1)
            print('temp_precision:',temp_test_precision)
            print('global precision is : ',test_precison)
            print('------------',ep*FLAGS.batch_size,'-------------')
            
        coord.request_stop()
        coord.join(threads)



    
    
if __name__=='__main__':
    if FLAGS.train==True:
        train()
    else:
        test()