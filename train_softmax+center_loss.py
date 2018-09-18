#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:01:32 2018

@author: wsw
"""

# train softamx+center_loss

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataset import make_dataset
import os
import time
import numpy as np
from center_loss import get_center_loss
from sklearn.decomposition import PCA
slim = tf.contrib.slim

tf.reset_default_graph()


def build_model(xs,num_classes=10,is_training=True,print_info=True):
  
  
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      kernel_size=[3,3],
                      ):
    net = slim.conv2d(xs,num_outputs=32,scope='conv1')
    net = slim.max_pool2d(net,kernel_size=[2,2],scope='pool1')
    net = slim.conv2d(net,num_outputs=64,scope='conv2')
    net = slim.max_pool2d(net,kernel_size=[2,2],scope='pool2')
    net = slim.flatten(net,scope='flatten')
    net = slim.fully_connected(net,num_outputs=512,scope='fc1')
    # extract feature output to display
    net = slim.fully_connected(net,num_outputs=128,scope='fc2')
    logits = slim.fully_connected(net,num_outputs=10,activation_fn=None,scope='fc3')
    if print_info:
      varNums,varBytes = slim.model_analyzer.analyze_vars(tf.trainable_variables(),
                                                          print_info=False)
      print('Trainable variable Nums:%d Bytes:%d'%(varNums,varBytes))
    return net,logits


def compute_test_accuracy(logits,targets):
  predict_labs = np.argmax(np.array(logits),axis=-1)
  targets = np.array(targets)
  accuracy = np.mean(np.equal(predict_labs,targets))
  print('\033[1;31mTest Accuracy:%.3f\033[0m'%accuracy)
  

def display_fig(features,targets,epoch):
  if not os.path.exists('./center_loss_images'):
    os.makedirs('./center_loss_images')
  features = np.array(features)
  fig = plt.figure()
  ax = Axes3D(fig)
  pca = PCA(n_components=3)
  features = pca.fit_transform(features)
  ax.scatter(features[:,0],features[:,1],features[:,2],c=targets)
  fig.savefig('./center_loss_images/%d.png'%epoch)
  plt.close(fig)
  
def train():
  
  # make train dataset
  train_dataset = make_dataset(dataType='train',epoch=20,batchsize=128)
  train_iter = train_dataset.make_one_shot_iterator()
  train_img_batch,train_lab_batch = train_iter.get_next()
  
  # make test dataset
  test_dataset = make_dataset(dataType='test',epoch=None,batchsize=128)
  test_iter = test_dataset.make_one_shot_iterator()
  test_img_batch,test_lab_batch = test_iter.get_next()

  with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
    train_features,logits = build_model(train_img_batch,print_info=True)
    
    test_features,test_logits = build_model(test_img_batch,
                                            is_training=False,
                                            print_info=False)
    
  with tf.name_scope('softmax_loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=train_lab_batch,
                                                  logits=logits)
    # compute center loss
    center_loss,centers,center_update_op = get_center_loss(train_features,train_lab_batch,alpha=0.5)
    
  with tf.name_scope('optimizer'):
    global_step = tf.train.create_global_step()
    optimizer = tf.train.MomentumOptimizer(0.01,0.9)
    # first update center vector
    with tf.control_dependencies([center_update_op]):
      train_op = optimizer.minimize(loss+center_loss,global_step)
  
  with tf.name_scope('compute_accu'):
    predict_lab = tf.argmax(logits,axis=-1)
    accu_op = tf.reduce_mean(tf.cast(tf.equal(predict_lab,train_lab_batch),dtype=tf.float32))
  
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    try:
      epoch = 1
      while not coord.should_stop():
        start = time.time()
        # train_imgs,train_labs = sess.run([train_img_batch,train_lab_batch])
        loss_value,c_loss_value,accu,_ = sess.run([loss,center_loss,accu_op,train_op])
        end = time.time()
        step = global_step.eval()
        fmt = 'Epoch[{:02d}]-Step:{:05d}-SoftmaxLoss:{:.3f}-CenterLoss:{:.3f}-Accu:{:.3f}-Time:{:.3f}(Sec)'.\
        format(epoch,step,loss_value,c_loss_value,accu,(end-start))
        if step%100 == 0:
          print(fmt)
        if step%469 == 0:
          epoch += 1
          # note cant sess.run() a test batch 
          # test_labs = sess.run(test_lab_batch)
          testNums = 10000
          outputs = []
          labels = []
          feats = []
          for i in range(testNums//128+1):
            # here sess.run a batch data
            batch_features,batch_logits,test_labs = sess.run([test_features,test_logits,test_lab_batch])
            outputs.extend(batch_logits.tolist())
            labels.extend(test_labs.tolist())
            feats.extend(batch_features.tolist())
          compute_test_accuracy(outputs,labels)
          display_fig(feats,labels,epoch-1)
    except tf.errors.OutOfRangeError:
      print('Train Finished!!!')
      coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  train()