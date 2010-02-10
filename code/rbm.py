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
        batch_size = 20, shared_W = None, shared_b = None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param batch_size: [mini]batch size used in the SGD updates

        :param shared_W: None for standalone RBMs or symbolic variable to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param shared_b: None for standalone RBMs or symbolic variable to a
        shared bias vector in case RBM is part of a DBN network
        """

        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        
        # setup theano random number generator
        self.theano_rng = RandomStreams()
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

        # Create the shared variables that will store the negative phase
        # values; these values are usable only in case of PCD
        self.n_vis = theano.shared(value = numpy.zeros((batch_size,n_visible)))
        self.n_hid = theano.shared(value = numpy.zeros((batch_size,n_hidden)))
        self.params     = [self.W, self.vbias, self.hbias]
        self.batch_size = batch_size
 
    def CD_k(self, n_Gibbs_steps = 1, persistent = False):
        '''
        :param n_Gibbs_steps: number of Gibbs steps to do when computing the 
        gradient with CD
        '''

        # define the graph for positive phase
        p_hid_activation = T.dot(self.input, self.W) + self.hbias
        p_hid            = T.nnet.sigmoid(p_hid_activation)
        p_hid_sample     = self.theano_rng.binomial(T.shape(p_hid), 1, p_hid)*1.0
        
        # for negative phase we need to implement k Gibbs steps; for this we 
        # will use the scan op (see theano documentation about it)
    
        # Create a function that builds the graph for one step of Gibbs
        def oneGibbsStep( vis_km1, hid_km1):
           # For clarity we used the following naming conventions : 
           #     var_kmx -> the value of variable ``var`` at step k - x
           #     var_k   -> the value of variable ``var`` at step k
           #     var_kpx -> the value of variable ``var`` at step k + x

           # if you want persistent CD, we have to use the last negative 
           # value generated ( which is stored in ``self.n_hid``
           if persistent : 
                hid_km1 = self.n_hid

           # sample this hidden layer
           hid_sample_km1   = self.theano_rng.binomial(T.shape(hid_km1),1,hid_km1)*1.0
           # compute visible layer values
           vis_activation_k = T.dot(hid_sample_km1, self.W.T) + self.vbias
           vis_k            = T.nnet.sigmoid(vis_activation_k)
           # sample the visible
           vis_sample_k     = self.theano_rng.binomial(T.shape(vis_k),1,vis_k)*1.0
           # compute new hidden layer values
           hid_activation_k = T.dot(vis_sample_k, self.W) + self.hbias
           hid_k            = T.nnet.sigmoid(hid_activation_k)

           # return a list of outputs, plus a dictionary of updates 
           return ([vis_k, hid_k],{ self.n_vis : vis_k, self.n_hid : hid_k})
        
        # to compute the negative phase perform k Gibbs step; for this we 
        # use the scan op, that implements a loop

        # keep_outputs tells scan that we do not care about intermediate values
        # of n_vis and n_hid, and that it should only return the last one
        n_vis, n_hid = scan(oneGibbsStep,[],[self.input, p_hid],\
               [], n_steps = n_Gibbs_steps, keep_outputs = {0:False, 1:False} ) 

        g_vbias = T.mean( self.input - n_vis , axis = 0)
        g_hbias = T.mean( p_hid      - n_hid , axis = 0)
        
        g_W = T.dot(p_hid.T, self.input )/ self.batch_size - \
              T.dot(n_hid.T, n_vis      )/ self.batch_size

        gparams = [g_W.T, g_vbias, g_hbias]
        # define dictionary of stochastic gradient update equations
        cost = T.mean(abs(self.input - n_vis))
        return (gparams, cost)

       


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
                      n_hidden = 500)

    (gparams,cost) = rbm_object.CD_k(3)

    updates = {}
    for param,gparam in zip( rbm_object.params, gparams):
        updates[param] = param + learning_rate* gparam

    train_rbm = theano.function([index], cost, 
           updates = updates, 
           givens = { 
             x: train_set_x[index*batch_size:(index+1)*batch_size],
             y: train_set_y[index*batch_size:(index+1)*batch_size]}
             )

    start_time = time.clock()  
    # go through training epochs 
    for epoch in xrange(training_epochs):
        # go through the training set
        c = []
        for batch_index in xrange(n_train_batches):
           c += [ train_rbm(batch_index) ]
        print 'Training epoch %d '%epoch, numpy.mean(c)
 
    end_time = time.clock()

    print ('Training took %f minutes' %((end_time-start_time)/60.))



if __name__ == '__main__':
    sgd_optimization_mnist()

