#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:03:48 2018

@author: wsw
"""

# angular loss
import tensorflow as tf

def get_angular_loss(logits,labels,s=64.0,m=0.5):
  """
  logits:last layer output without activation fuction
         weights and feature vector have been l2-normalizerd
  labels:target label without one-hot encoding
  m:angular margin default=0.5
  s:hypershpere radius default=64.0
  """
  target_margin = tf.one_hot(labels,depth=10,on_value=m)
  # get target angular
  # arccos(logits)
  target_angular = tf.acos(logits)
  # get target logits
  target_logits = tf.cos(target_angular+target_margin)
  # compute angular loss
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=target_logits)
  return loss