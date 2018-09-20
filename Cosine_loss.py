#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:18:22 2018

@author: wsw
"""

# Cosine-loss

import tensorflow as tf

def get_cosine_loss(logits,labels,s=64.0,m=0.35):
  """
  logits:last layer output without activation fuction
         weights and feature vector have been l2-normalizerd
  labels:target label without one-hot encoding
  m:cosine margin default=0.35
  s:hypershpere radius default=64.0
  """
  target_margin = tf.one_hot(labels,depth=10,on_value=m)
  # print(logits.shape,target_margin.shape)
  # logits substract target margin
  target_logits = s*(logits - target_margin)
  # compute cosine loss
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=target_logits)
  return loss