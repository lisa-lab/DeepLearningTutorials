"""
This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Description of RBM : TODO
"""


import numpy
import theano
import theano.tensor as T
import time
import gzip
import cPickle

from theano.tensor.shared_randomstreams import RandomStreams

from theano.sandbox.scan import scan


class RBM():
    """Restricted Boltzmann Machine (RBM)
    """
    def __init__(self, input=None, n_visible=784, n_hidden=500, \
        n_Gibbs_steps = 3, shared_W = None, shared_b = None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param n_Gibbs_steps: number of Gibbs steps to do when computing the gradient
        :param shared_W: None for standalone RBMs or symbolic variable to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param shared_b: None for standalone RBMs or symbolic variable to a
        shared bias vector in case RBM is part of a DBN network
        """

        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        
        # setup theano random number generator
        theano_rng = RandomStreams()
        numpy_rng  = numpy.random.RandomState()
        
        # initial values for weights and biases
        if shared_W and shared_b :
            self.W     = shared_W
            self.hbias = shared_b
        else:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
            # the output of uniform if converted using asarray to dtype 
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray( numpy.random.uniform( \
                  low = -numpy.sqrt(6./(n_hidden+n_visible)), \
                high = numpy.sqrt(6./(n_hidden+n_visible)), \
                size = (n_visible, n_hidden)), dtype = theano.config.floatX)
            # initial value of the hidden units bias
            initial_hbias   = numpy.zeros(n_hidden)

            # theano shared variables for weights and biases
            self.W     = theano.shared(value = initial_W    , name = 'W')
            self.hbias = theano.shared(value = initial_hbias, name = 'hbias')


        # initial value of the visible units bias 
        initial_vbias  = numpy.zeros(n_visible)
        self.vbias = theano.shared(value = initial_vbias, name = 'vbias')

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input if input else T.dmatrix('input')

        #### POSITIVE AND NEGATIVE PHASE ####

        # define graph for positive phase
        p_hid_activation = T.dot(self.input, self.W) + self.hbias
        self.p_hid            = T.nnet.sigmoid(p_hid_activation)
        p_hid_sample     = theano_rng.binomial(T.shape(self.p_hid), 1, self.p_hid)*1.0

    
        def oneGibbsStep(vis_km1, hid_km1):
           hid_sample_km1   = theano_rng.binomial(T.shape(hid_km1),1,hid_km1)*1.0
           vis_activation_k = T.dot(hid_sample_km1, self.W.T) + self.vbias
           vis_k            = T.nnet.sigmoid(vis_activation_k)
           vis_sample_k     = theano_rng.binomial(T.shape(vis_k),1,vis_k)*1.0
           hid_activation_k = T.dot(vis_sample_k, self.W) + self.hbias
           hid_k            = T.nnet.sigmoid(hid_activation_k)
           
           return [vis_k, hid_k]
           
        # to compute the negative phase perform k Gibbs step; for this we 
        # use the scan op, that implements a loop

        self.n_vis_values, self.n_hid_values = scan(oneGibbsStep,[],[self.input, self.p_hid],
                            [], n_steps = n_Gibbs_steps) #, mode='DEBUG_MODE')

        
        self.g_vbias = T.mean( self.input - self.n_vis_values[-1], axis = 0)
        self.g_hbias = T.mean( self.p_hid      - self.n_hid_values[-1], axis = 0)

        minibatch_size = self.input.shape[0]
        self.g_W = T.dot(self.p_hid.T           , self.input      )/minibatch_size - \
              T.dot(self.n_hid_values[-1].T, self.n_vis_values[-1])/ minibatch_size

        self.params  = [self.W, self.vbias, self.hbias]
        self.gparams = [self.g_W, self.g_vbias, self.g_hbias]

        # define dictionary of stochastic gradient update equations
        self.updates = zip (self.params, self.gparams)
        self.cost = T.mean(self.input - self.n_vis_values[-1])




def sgd_optimization_mnist( learning_rate=0.1, training_epochs = 20, \
                            dataset='mnist.pkl.gz'):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer 
    perceptron

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used in the finetune stage 
    (factor for the stochastic gradient)

    :param pretraining_epochs: number of epoch to do pretraining

    :param pretrain_lr: learning rate to be used during pre-training

    :param n_iter: maximal number of iterations ot run the optimizer 

    :param dataset: path the the pickled dataset

    """

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)

    batch_size = 20    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y') # the labels are presented as 1D vector of 
                                 # [int] labels




    # construct the RBM class
    rbm_object = RBM( input = x, n_visible=28*28, \
                      n_hidden = 500, n_Gibbs_steps = 3)

    train_rbm = theano.function([index], rbm_object.g_hbias, 
           updates = {},#rbm_object.updates, 
           givens = { 
             x: train_set_x[index*batch_size:(index+1)*batch_size],
             y: train_set_y[index*batch_size:(index+1)*batch_size]}
             )# , mode='DEBUG_MODE')

    start_time = time.clock()  
    # go through training epochs 
    for epoch in xrange(training_epochs):
        # go through the training set
        for batch_index in xrange(n_train_batches):
           c =  train_rbm(batch_index)
           print '---------------------------------------------'
           print c.shape
           print c
        print 'Training epoch %d '%epoch, c
 
    end_time = time.clock()

    print ('Training took %f minutes' %((end_time-start_time)/60.))



if __name__ == '__main__':
    sgd_optimization_mnist()

