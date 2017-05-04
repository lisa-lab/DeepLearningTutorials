
# coding: utf-8

#!/usr/bin/env python2


import os


import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree
import pickle

import numpy as np
import random
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params
from lasagne.objectives import categorical_crossentropy

import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import gridspec

from data_loader.cortical_layers import Cortical6LayersDataset
from fcn1D import build_model



def accuracy_metric(y_pred, y_true, void_labels, one_hot=False):

    assert (y_pred.ndim == 2) or (y_pred.ndim == 1)

    # y_pred to indices
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=1)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Compute accuracy
    acc = T.eq(y_pred, y_true).astype(_FLOATX)

    # Create mask
    mask = T.ones_like(y_true, dtype=_FLOATX)
    for el in void_labels:
        indices = T.eq(y_true, el).nonzero()
        if any(indices):
            mask = T.set_subtensor(mask[indices], 0.)

    # Apply mask
    acc *= mask
    acc = T.sum(acc) / T.sum(mask)

    return acc

# In[2]:

_FLOATX = config.floatX


def jaccard(y_pred, y_true, n_classes, one_hot=False):

    assert (y_pred.ndim == 2) or (y_pred.ndim == 1)

    # y_pred to indices
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=1)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Compute confusion matrix
    cm = T.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cm = T.set_subtensor(
                cm[i, j], T.sum(T.eq(y_pred, i) * T.eq(y_true, j)))

    # Compute Jaccard Index
    TP_perclass = T.cast(cm.diagonal(), _FLOATX)
    FP_perclass = cm.sum(1) - TP_perclass
    FN_perclass = cm.sum(0) - TP_perclass

    num = TP_perclass
    denom = TP_perclass + FP_perclass + FN_perclass

    return T.stack([num, denom], axis=0)

SAVEPATH = 'save_models/'
LOADPATH = SAVEPATH
WEIGHTS_PATH = SAVEPATH



# In[22]:

#Model hyperparameters
n_filters = 64
filter_size = 25
depth  = 8
block = 'bn_relu_conv'

#Training loop hyperparameters
weight_decay=0.001
num_epochs=500
max_patience=25
resume=False
learning_rate_value = 0.0005 #learning rate is defined below as a theano variable.

#Hyperparameters for the dataset loader
batch_size=[1000,1000,1]
smooth_or_raw = 'both' #use both input channels
shuffle_at_each_epoch = True
n_layers = 6 #use the 6layer dataset


#
# Prepare load/save directories
#

savepath=SAVEPATH
loadpath=LOADPATH

exp_name = 'fcn1D'
exp_name += '_lrate=' + str(learning_rate_value)
exp_name += '_fil=' + str(n_filters)
exp_name += '_fsizes=' + str(filter_size)
exp_name += '_depth=' + str(depth)
exp_name += '_data=' + smooth_or_raw
exp_name += '_decay=' + str(weight_decay)
exp_name += '_pat=' + str(max_patience)


dataset = str(n_layers)+'cortical_layers'
savepath = os.path.join(savepath, dataset, exp_name)
loadpath = os.path.join(loadpath, dataset, exp_name)
print 'Savepath : '
print savepath
print 'Loadpath : '
print loadpath

if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print('\033[93m The following folder already exists {}. '
          'It will be overwritten in a few seconds...\033[0m'.format(
              savepath))

print('Saving directory : ' + savepath)
with open(os.path.join(savepath, "config.txt"), "w") as f:
    for key, value in locals().items():
        f.write('{} = {}\n'.format(key, value))



#
# Define symbolic variables
#
input_var = T.tensor3('input_var') #n_example*nb_in_channels*ray_size
target_var = T.ivector('target_var') #n_example*ray_size

learn_step=  theano.shared(np.array(learning_rate_value, dtype=theano.config.floatX))

#
# Build dataset iterator
#

if smooth_or_raw =='both':
    nb_in_channels = 2
    use_threads = False
else:
    nb_in_channels = 1
    use_threads = True


train_iter = Cortical6LayersDataset(
    which_set='train',
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[0],
    data_augm_kwargs={},
    shuffle_at_each_epoch = True,
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads)

val_iter = Cortical6LayersDataset(
    which_set='valid',
    smooth_or_raw = smooth_or_raw,
    batch_size=batch_size[1],
    shuffle_at_each_epoch = True,
    return_one_hot=False,
    return_01c=False,
    return_list=False,
    use_threads=use_threads)

test_iter = None



n_batches_train = train_iter.nbatches
n_batches_val = val_iter.nbatches
n_batches_test = test_iter.nbatches if test_iter is not None else 0
n_classes = train_iter.non_void_nclasses
void_labels = train_iter.void_labels


#
# Build network
#
simple_net_output, net = build_model(input_var,
                    filter_size = filter_size,
                    n_filters = n_filters,
                    depth = depth,
                    block= block,
                    nb_in_channels = nb_in_channels,
                    n_classes = n_classes)


#
# Define and compile theano functions
#
print "Defining and compiling training functions"

prediction = lasagne.layers.get_output(simple_net_output[0])
loss = categorical_crossentropy(prediction, target_var)
loss = loss.mean()

if weight_decay > 0:
    weightsl2 = regularize_network_params(
        simple_net_output, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

train_acc = accuracy_metric(prediction, target_var, void_labels)

params = lasagne.layers.get_all_params(simple_net_output, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)

print "Done"


print "Defining and compiling valid functions"
valid_prediction = lasagne.layers.get_output(simple_net_output[0],deterministic=True)
valid_loss = categorical_crossentropy(valid_prediction, target_var).mean()
valid_acc  = accuracy_metric(valid_prediction, target_var, void_labels)
valid_jacc = jaccard(valid_prediction, target_var, n_classes)

valid_fn = theano.function([input_var, target_var], [valid_loss, valid_acc,valid_jacc])
print "Done"

#
# Train loop
#
err_train = []
acc_train = []

err_valid = []
acc_valid = []
jacc_valid = []
patience = 0




# Training main loop
print "Start training"

for epoch in range(num_epochs):
    learn_step.set_value((learn_step.get_value()*0.99).astype(theano.config.floatX))

    # Single epoch training and validation
    start_time = time.time()
    #Cost train and acc train for this epoch
    cost_train_epoch = 0
    acc_train_epoch = 0


    for i in range(n_batches_train):
        # Get minibatch (comment the next line if only 1 minibatch in training)
        train_batch = train_iter.next()
        X_train_batch, L_train_batch, idx_train_batch = train_batch['data'], train_batch['labels'], train_batch['filenames'][0]
        L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))

        # Training step
        cost_train_batch, acc_train_batch = train_fn(X_train_batch, L_train_batch)



        #Update epoch results
        cost_train_epoch += cost_train_batch
        acc_train_epoch += acc_train_batch


    #Add epoch results
    err_train += [cost_train_epoch/n_batches_train]
    acc_train += [acc_train_epoch/n_batches_train]

    # Validation
    cost_val_epoch = 0
    acc_val_epoch = 0
    jacc_val_epoch = np.zeros((2, n_classes))


    for i in range(n_batches_val):

        # Get minibatch (comment the next line if only 1 minibatch in training)
        val_batch = val_iter.next()
        X_val_batch, L_val_batch, idx_val_batch = val_batch['data'], val_batch['labels'], val_batch['filenames'][0]
        L_val_batch = np.reshape(L_val_batch, np.prod(L_val_batch.shape))

        # Validation step
        cost_val_batch, acc_val_batch, jacc_val_batch = valid_fn(X_val_batch, L_val_batch)

        #Update epoch results
        cost_val_epoch += cost_val_batch
        acc_val_epoch += acc_val_batch
        jacc_val_epoch += jacc_val_batch


    #Add epoch results
    err_valid += [cost_val_epoch/n_batches_val]
    acc_valid += [acc_val_epoch/n_batches_val]
    jacc_perclass_valid = jacc_val_epoch[0, :] / jacc_val_epoch[1, :]
    jacc_valid += [np.mean(jacc_perclass_valid)]
    # worse_indices_valid += [worse_indices_val_epoch]


    #Print results (once per epoch)

    out_str = "EPOCH %i: Avg cost train %f, acc train %f"+        ", cost val %f, acc val %f, jacc val %f took %f s"
    out_str = out_str % (epoch, err_train[epoch],
                         acc_train[epoch],
                         err_valid[epoch],
                         acc_valid[epoch],
                         jacc_valid[epoch],
                         time.time()-start_time)
    print out_str



    # Early stopping and saving stuff

    with open(os.path.join(savepath, "fcn1D_output.log"), "a") as f:
        f.write(out_str + "\n")

    if epoch == 0:
        best_jacc_val = jacc_valid[epoch]
    elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
        print('saving best (and last) model')
        best_jacc_val = jacc_valid[epoch]
        patience = 0
        np.savez(os.path.join(savepath, 'new_fcn1D_model_best.npz'),  *lasagne.layers.get_all_param_values(simple_net_output))
        np.savez(os.path.join(savepath , "fcn1D_errors_best.npz"),
                 err_train=err_train, acc_train=acc_train,
                 err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
    else:
        patience += 1
        print('saving last model')

    np.savez(os.path.join(savepath, 'new_fcn1D_model_last.npz'),  *lasagne.layers.get_all_param_values(simple_net_output))
    np.savez(os.path.join(savepath , "fcn1D_errors_last.npz"),
             err_train=err_train, acc_train=acc_train,
             err_valid=err_valid, acc_valid=acc_valid, jacc_valid=jacc_valid)
    # Finish training if patience has expired or max nber of epochs reached

    if patience == max_patience or epoch == num_epochs-1:
        if savepath != loadpath:
            print('Copying model and other training files to {}'.format(loadpath))
            copy_tree(savepath, loadpath)
        break
