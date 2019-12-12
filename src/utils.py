"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import tensorflow as tf
import math
import numpy as np
import copy
import h5py

def load_test_data(idx, filetest=None, dataset="test_dataset"):

    img = filetest[dataset][idx]
    img = np.array(img).astype(np.float32)
    img = img[None, :, :, None]

    return img

def load_train_data(idx, is_testing=False, batch_size=1, fileA=None, fileB=None, dataset="train_dataset"):

    img_A = fileA[dataset][idx*batch_size:(idx+1)*batch_size]
    img_A = np.array(img_A).astype(np.float32)[:,:,:,None]

    img_B = fileB[dataset][idx*batch_size:(idx+1)*batch_size]
    img_B = np.array(img_B).astype(np.float32)[:,:,:,None]

    if not is_testing:
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    img_AB = np.concatenate((img_A, img_B), axis=3)

    return img_AB


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)