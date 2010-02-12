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

class RBM(object):
    """
    *** WRITE THE ENERGY FUNCTION  USE SAME LETTERS AS VARIABLE NAMES IN CODE
    """

    def __init__(self, input=None, n_visible=784, n_hidden=500,
            W=None, hbias=None, vbias=None,
            numpy_rng=None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units (necessary when W or vbias is None)

        :param n_hidden: number of hidden units (necessary when W or hbias is None)

        :param W: weights to use for the RBM.  None means that a shared variable will be
        created with a randomly chosen matrix of size (n_visible, n_hidden).

        :param hbias: ***

        :param vbias: ***

        :param numpy_rng: random number generator (necessary when W is None)

        """
        
        params = []
        if W is None:
            # choose initial values for weight matrix of RBM 
            initial_W = numpy.asarray(
                    numpy_rng.uniform( \
                        low=-numpy.sqrt(6./(n_hidden+n_visible)), \
                        high=numpy.sqrt(6./(n_hidden+n_visible)), \
                        size=(n_visible, n_hidden)), \
                    dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W')
            params.append(W)

        if hbias is None:
            # theano shared variables for hidden biases
            hbias = theano.shared(value=numpy.zeros(n_hidden), name='hbias')
            params.append(hbias)

        if vbias is None:
            # theano shared variables for visible biases
            vbias = theano.shared(value=numpy.zeros(n_visible), name='vbias')
            params.append(vbias)

        if input is None:
            # initialize input layer for standalone RBM or layer0 of DBN
            input = T.dmatrix('input')

        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = Trng = RandomStreams()
        self.params = params
        self.hidden_mean = T.nnet.sigmoid(T.dot(input, W)+hbias)
        self.hidden_sample = Trng.binomial(self.hidden_mean.shape, 1, self.hidden_mean)

    def gibbs_k(self, v_sample, k):
        ''' This function implements k steps of Gibbs sampling '''
 
        # We compute the visible after k steps of Gibbs by iterating 
        # over ``gibs_1`` for k times; this can be done in Theano using
        # the `scan op`. For a more comprehensive description of scan see 
        # http://deeplearning.net/software/theano/library/scan.html .
        
        def gibbs_1(v0_sample, W, hbias, vbias):
            ''' This function implements one Gibbs step '''

            # compute the activation of the hidden units given a sample of the
            # vissibles
            h0_mean = T.nnet.sigmoid(T.dot(v0_sample, W) + hbias)
            # get a sample of the hiddens given their activation
            h0_sample = self.theano_rng.binomial(h0_mean.shape, 1, h0_mean)
            # compute the activation of the visible given the hidden sample
            v1_mean = T.nnet.sigmoid(T.dot(h0_sample, W.T) + vbias)
            # get a sample of the visible given their activation
            v1_act = self.theano_rng.binomial(v1_mean.shape, 1, v1_mean)
            return [v1_act, v1_mean]

       
       
        # Because we require as output two values, namely the mean field
        # approximation of the visible and the sample obtained after k steps, 
        # scan needs to know the shape of those two outputs. Scan takes 
        # this information from the variables containing the initial state
        # of the outputs. Since we do not need a initial state of ``v_mean``
        # we provide a dummy one used only to get the correct shape 
        v_mean = T.zeros_like(v_sample)
        
        # ``outputs_taps`` is an argument of scan which describes at each
        # time step what past values of the outputs the function applied 
        # recursively needs. This is given in the form of a dictionary, 
        # where the keys are outputs indexes, and values are a list of 
        # of the offsets used  by the corresponding outputs
        # In our case the function ``gibbs_1`` applied recursively, requires
        # at time k the past value k-1 for the first output (index 0) and
        # no past value of the second output
        outputs_taps = { 0 : [-1], 1 : [] }

        v_samples, v_means = theano.scan( fn = gibbs_1, 
                                          sequences      = [], 
                                          initial_states = [v_sample, v_mean],
                                          non_sequences  = self.params, 
                                          outputs_taps   = outputs_taps,
                                          n_steps        = k)
        return v_means[-1], v_samples[-1]

    def free_energy(self, v_sample):
        h_mean = T.nnet.sigmoid(T.dot(v_sample, self.W) + self.hbias)
        #TODO: make sure log(sigmoid) is optimized to something stable!
        return -T.sum(T.log(1.0001-h_mean)) - T.sum(T.dot(v_sample, self.vbias))

    def cd(self, visible=None, persistent=None, steps = 1):
        """
        Return a 5-tuple of values related to contrastive divergence: (cost,
        end-state of negative-phase chain, gradient on weights, gradient on
        hidden bias, gradient on visible bias)

        If visible is None, it defaults to self.input
        If persistent is None, it defaults to self.input

        CD aka CD1 - cd()
        CD-10      - cd(steps=10)
        PCD        - cd(persistent=shared(numpy.asarray(initializer)))
        PCD-k      - cd(persistent=shared(numpy.asarray(initializer)),
                        steps=10)
        """
        if visible is None:
            visible = self.input

        if visible is None:
            raise TypeError('visible argument is required when self.input is None')

        if persistent is None:
            chain_start = visible
        else:
            chain_start = persistent
        chain_end_mean, chain_end_sample = self.gibbs_k(chain_start, steps)

        cost = self.free_energy(visible) - self.free_energy(chain_end_sample)
        
        # Compute the gradient of the cost with respect to the parameters
        # Note the use of argument ``consider_constant``. The reason for 
        # using this parameter is because the gradient should not try to 
        # propagate through the gibs chain
        gparams = T.grad(cost, self.params, consider_constant = [chain_end_sample])
       
        cross_entropy_error = T.mean(T.sum( visible*T.log(chain_end_sample) + 
                        (1 - visible)*T.log(1-chain_end_sample), axis = 1))
        return (cross_entropy_error, chain_end_sample,) + tuple(gparams)

    def cd_updates(self, lr, visible=None, persistent=None, steps = 1):
        """
        Return the learning updates for the RBM parameters that are shared variables.

        Also returns an update for the persistent if it is a shared variable.

        These updates are returned as a dictionary.

        :param lr: [scalar] learning rate for contrastive divergence learning
        :param visible: see `cd_grad`
        :param persistent: see `cd_grad`
        :param step: see `cd_grad`

        """

        cost, chain_end, gW, ghbias, gvbias = self.cd(visible, persistent, steps)

        updates = {}
        if self.W in self.params:
            updates[self.W] = self.W - lr * gW
        if self.hbias in self.params:
            updates[self.hbias] = self.hbias - lr * ghbias
        if self.vbias in self.params:
            updates[self.vbias] = self.vbias - lr * gvbias
        if persistent:
            #if persistent is a shared var, then it means we should use
            updates[persistent] = chain_end

        return updates

def test_RBM(learning_rate=0.1, training_epochs = 20, 
        dataset='mnist.pkl.gz'):

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')

    print '... loading data'
    train_set_x, train_set_y = shared_dataset(train_set)

    batch_size = 20    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y') # the labels are presented as 1D vector of 
                                 # [int] labels

    print '... making model'
    # construct the RBM class
    rbm = RBM(input = x, n_visible=28*28, n_hidden=500, numpy_rng=
            numpy.random.RandomState(234234))
    cost = rbm.cd(steps = 10 )[0]

    print '... compiling train function'
    train_rbm = theano.function([index], rbm.cd(steps = 10)[0], 
           updates = rbm.cd_updates(learning_rate, steps = 10), 
           givens = { 
             x: train_set_x[index*batch_size:(index+1)*batch_size],
             y: train_set_y[index*batch_size:(index+1)*batch_size]},
            mode = 'DEBUG_MODE'
             )

    start_time = time.clock()  
    # go through training epochs 
    for epoch in xrange(training_epochs):
        # go through the training set
        c = []
        for batch_index in xrange(n_train_batches):
            c += [ train_rbm(batch_index) ]
            print 'batch ', c[-1]
        print 'Training epoch %d '%epoch, numpy.mean(c)
 
    end_time = time.clock()

    print ('Training took %f minutes' %((end_time-start_time)/60.))

if __name__ == '__main__':
    test_RBM()


