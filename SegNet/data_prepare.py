from tensorflow.python.framework import ops
import tensorflow as tf
import os

FLAGS = tf.flags.FLAGS

#####################data prepare#############################
def get_file_names(file_dir,name):
    imgs=os.listdir(file_dir+'/'+name)
    labels=os.listdir(file_dir+'/'+name+'annot')
    img_names=[file_dir+'/'+name+'/'+x  for x in imgs]
    label_names=[file_dir+'/'+name+'annot'+'/'+x for x in labels]
    return sorted(img_names),sorted(label_names)



##########################################################
def get_data_label_batch(imgs_dir,labels_dir):
    imgs_tensor=ops.convert_to_tensor(imgs_dir,dtype=tf.string)
    labels_tensor=ops.convert_to_tensor(labels_dir,dtype=tf.string)
    filename_queue=tf.train.slice_input_producer([imgs_tensor,labels_tensor])
    
    image_filename = filename_queue[0]
    label_filename = filename_queue[1]
    
    imgs_values=tf.read_file(image_filename)
    label_values=tf.read_file(label_filename)
    
    imgs_decorded=tf.image.decode_png(imgs_values)
    labels_decorded=tf.image.decode_png(label_values)
    
    imgs_reshaped=tf.reshape(imgs_decorded,[FLAGS.img_height,FLAGS.img_width,3])
    labels_reshaped=tf.reshape(labels_decorded,[FLAGS.img_height,FLAGS.img_width,1])
    
    imgs_reshaped = tf.cast(imgs_reshaped, tf.float32)
    
    
    min_fraction_of_examples_in_queue = FLAGS.fraction_of_examples_in_queue
    min_queue_examples = int(FLAGS.num_examples_epoch_train *min_fraction_of_examples_in_queue)
    
    print ('Filling queue with %d input images before starting to train.This may take some time.' % min_queue_examples)
    
    if FLAGS.train==True:
        images_batch, labels_batch = tf.train.shuffle_batch([imgs_reshaped,labels_reshaped],
                                                           batch_size=FLAGS.batch_size,
                                                           num_threads=6,
                                                           capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                                           min_after_dequeue=min_queue_examples)
    if FLAGS.train==False:
        images_batch, labels_batch = tf.train.batch([imgs_reshaped, labels_reshaped],
                                                    batch_size=FLAGS.batch_size,
                                                    num_threads=6,
                                                    capacity=min_queue_examples + 3 * FLAGS.batch_size)
    
    
    return images_batch, labels_batch 