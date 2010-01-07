import numpy
import theano
import theano.tensor as T

from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano.compile.sandbox.shared_randomstreams import RandomStreams
from theano.tensor.nnet import sigmoid

class A():

    @execute
    def propup();
        # do symbolic prop
        self.hid = T.dot(

class RBM():

    def __init__(self, input=None, vsize=None, hsize=None, bsize=10, lr=1e-1, seed=123):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), as well
        as for performing CD updates.
        param input: None for standalone RBMs or symbolic variable if RBM is
                     part of a larger graph.
        param vsize: number of visible units
        param hsize: number of hidden units
        param bsize: size of minibatch
        param lr: unsupervised learning rate
        param seed: seed for random number generator
        """
        assert vsize and hsize

        self.vsize = vsize
        self.hsize = hsize
        self.lr = shared(lr, 'lr')
        
        # setup theano random number generator
        self.random = RandomStreams(seed)
       
        #### INITIALIZATION ####

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input if input else T.dmatrix('input')
        # initialize biases
        self.b = shared(numpy.zeros(vsize), 'b')
        self.c = shared(numpy.zeros(hsize), 'c')
        # initialize random weights
        rngseed = numpy.random.RandomState(seed).randint(2**30)
        rng = numpy.random.RandomState(rngseed)
        ubound = 1./numpy.sqrt(max(self.vsize,self.hsize))
        self.w = shared(rng.uniform(low=-ubound, high=ubound, size=(hsize,vsize)), 'w')
      

        #### POSITIVE AND NEGATIVE PHASE ####

        # define graph for positive phase
        ph, ph_s = self.def_propup(self.input)
        # function which computes p(h|v=x) and ~ p(h|v=x)
        self.pos_phase = pfunc([self.input], [ph, ph_s])

        # define graph for negative phase
        nv, nv_s = self.def_propdown(ph_s)
        nh, nh_s = self.def_propup(nv_s)
        # function which computes p(v|h=ph_s), ~ p(v|h=ph_s) and p(h|v=nv_s)
        self.neg_phase = pfunc([ph_s], [nv, nv_s, nh, nh_s])
        
        # calculate CD gradients for each parameter
        db = T.mean(self.input, axis=0) - T.mean(nv, axis=0)
        dc = T.mean(ph, axis=0) - T.mean(nh, axis=0)
        dwp = T.dot(ph.T, self.input)/nv.shape[0]
        dwn = T.dot(nh.T, nv)/nv.shape[0]
        dw = dwp - dwn

        # define dictionary of stochastic gradient update equations
        updates = {self.b: self.b - self.lr * db,
                   self.c: self.c - self.lr * dc,
                   self.w: self.w - self.lr * dw}

        # define private function, which performs one step in direction of CD gradient
        self.cd_step = pfunc([self.input, ph, nv, nh], [], updates=updates)


    def def_propup(self, vis):
        """ Symbolic definition of p(hid|vis) """
        hid_activation = T.dot(vis, self.w.T) + self.c
        hid = sigmoid(hid_activation)
        hid_sample = self.random.binomial(T.shape(hid), 1, hid)*1.0
        return hid, hid_sample
    
    def def_propdown(self, hid):
        """ Symbolic definition of p(vis|hid) """
        vis_activation = T.dot(hid, self.w) + self.b
        vis = sigmoid(vis_activation)
        vis_sample = self.random.binomial(T.shape(vis), 1, vis)*1.0
        return vis, vis_sample

    def cd(self, x, k=1):
        """ Performs actual CD update """
        ph, ph_s = self.pos_phase(x)
        
        nh_s = ph_s
        for ki in range(k):
            nv, nv_s, nh, nh_s = self.neg_phase(nh_s)

        self.cd_step(x, ph, nv_s, nh)



import os
from pylearn.datasets import MNIST

if __name__ == '__main__':

    bsize = 10

    # initialize dataset
    dataset = MNIST.first_1k() 
    # initialize RBM with 784 visible units and 500 hidden units
    r = RBM(vsize=784, hsize=500, bsize=bsize, lr=0.1)

    # for a fixed number of epochs ...
    for e in range(10):

        print '@epoch %i ' % e

        # iterate over all training set mini-batches
        for i in range(len(dataset.train.x)/bsize):

            rng = range(i*bsize,(i+1)*bsize) # index range of subsequent mini-batch
            x = dataset.train.x[rng]         # next mini-batch
            r.cd(x)                          # perform cd update

