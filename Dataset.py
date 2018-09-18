#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:40:52 2018

@author: wsw
"""

# make train or test dataset

import tensorflow as tf
import os

tf.reset_default_graph()

def parse_tfrecord_example(serialized_example):
  
  features = {'feature':tf.FixedLenFeature([],tf.string),
              'label':tf.FixedLenFeature([],tf.float32)
              }
  example = tf.parse_single_example(serialized_example,features=features)
  # decode img
  image = tf.decode_raw(example['feature'],out_type=tf.uint8)
  image = tf.reshape(image,shape=[28,28,1])
  # simple normalization
  image = tf.image.convert_image_dtype(image,dtype=tf.float32)
  # decode label
  label = tf.cast(example['label'],dtype=tf.int64)
  return image,label


def make_dataset(dataType='train',epoch=20,batchsize=128,buffer=10000):
  # make train dataset
  if dataType=='train':
    tfrecords_path = os.path.join('../mnist','train.tfrecords')
  # make test dataset
  elif dataType=='test':
    tfrecords_path = os.path.join('../mnist','test.tfrecords')
  else:
    raise Exception('Dataset type unkown,must be train or test')
  # num_parallel_reads resprents read file number in parallel
  # num_parallel_reads=1 reading files sequentially.
  dataset = tf.data.TFRecordDataset([tfrecords_path],num_parallel_reads=1)
  dataset = dataset.map(parse_tfrecord_example,num_parallel_calls=8)
  dataset = dataset.shuffle(buffer).repeat(epoch).batch(batch_size=batchsize)
  return dataset

def test_dataset():
  dataset = make_dataset(epoch=1)
  train_iter = dataset.make_one_shot_iterator()
  # get batch
  img_batch,lab_batch = train_iter.get_next()
  sess = tf.Session()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess,coord)
  try:
    while not coord.should_stop():
      imgs,labs = sess.run([img_batch,lab_batch])
      print(imgs.shape,labs.shape)
      import matplotlib.pyplot as plt
      fig,ax = plt.subplots(1,3)
      ax[0].imshow(imgs[0][...,0])
      ax[0].set_title(str(labs[0]))
      ax[1].imshow(imgs[1][...,0])
      ax[1].set_title(str(labs[1]))
      ax[2].imshow(imgs[2][...,0])
      ax[2].set_title(str(labs[2]))
      plt.show()
      break
  except tf.errors.OutOfRangeError:
    coord.request_stop()
  coord.join(threads)
      
if __name__ == '__main__':
  test_dataset()


  