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


        # *** This is not seeded deterministically
        # *** This generator is not even used (??)
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

        # **** WARNING: It is not a good idea to put things in this list other than shared variables
        #      created in this function.
        self.params     = [self.W, self.vbias, self.hbias]
        self.batch_size = batch_size
 
    def CD_k(self, n_Gibbs_steps = 1, persistent = False):
        '''
        :param n_Gibbs_steps: number of Gibbs steps to do when computing the 
        gradient with CD

        *** Document what does this function return?

        *** How about a more flexible way to make this function do PCD: an optional
        visible_0=None parameter to this function where None means self.input.  This variable
        is used to initialize the negative chain.
        '''

        # define the graph for positive phase
        p_hid_activation = T.dot(self.input, self.W) + self.hbias
        p_hid            = T.nnet.sigmoid(p_hid_activation)

        # *** why multiply by 1?  Is this meant to be a cast?
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

           # *** This is not a good feature... scan shouldn't be doing updates like this.  This
           # is a potentially bad bug.
           if persistent : 
                hid_km1 = self.n_hid

           # *** why multiply by 1?  Is this meant to be a cast?
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

           # *** Could we modify scan to accept a friendlier encoding of this information,
           # like: 
           #    dict( outputs=[vis_k, hid_k], updates={...})
           return ([vis_k, hid_k],{ self.n_vis : vis_k, self.n_hid : hid_k})
        
        # to compute the negative phase perform k Gibbs step; for this we 
        # use the scan op, that implements a loop


        # *** Could we use a different variable name here?  n_something usually means the
        # number of somethings.  Like the number of visible or hidden units.

        # keep_outputs tells scan that we do not care about intermediate values
        # of n_vis and n_hid, and that it should only return the last one
        n_vis_vals, n_hid_vals = scan(oneGibbsStep,[],[self.input, p_hid],\
               [], n_steps = n_Gibbs_steps  ) 

        g_vbias = T.mean( self.input - n_vis_vals[-1] , axis = 0)
        g_hbias = T.mean( p_hid      - n_hid_vals[-1] , axis = 0)

        # ***Why are we using mean for the biases but a dot()/size formula for the weights?
        #    It's a minor point, but we're confusing two kinds of terminology.
        #    Better would be using mean & covariance (I think we have a cov() op...)
        #    -or- sum()/batchsize and then dot() / batchsize
        
        g_W = T.dot(p_hid.T         , self.input          )/ self.batch_size - \
              T.dot(n_hid_vals[-1].T, n_vis_vals[-1]      )/ self.batch_size

        gparams = [g_W.T, g_vbias, g_hbias]
        # define dictionary of stochastic gradient update equations

        # *** It is misleading to say that it returns a cost, since usually a cost is the thing
        # we are minimizing.

        cost = T.mean(abs(self.input - n_vis_vals[-1]))
        return (gparams, cost)


# *** rename this function
def sgd_optimization_mnist( learning_rate=0.1, training_epochs = 20, \
                            dataset='mnist.pkl.gz'):
    """
    Demonstrate ***

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


class RBM_option2(object):
    """
    *** WRITE THE ENERGY FUNCTION  USE SAME LETTERS AS VARIABLE NAMES IN CODE
    """

    @classmethod
    def new(cls, input=None, n_visible=784, n_hidden=500,
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

        return cls(input, W, hbias, vbias, params)

    def __init__(self, input, W, hbias, vbias, params):

        # setup theano random number generator
        self.visible = self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = Trng = RandomStreams()
        self.params = params
        self.hidden_mean = T.nnet.sigmoid(T.dot(input, W)+hbias)
        self.hidden_sample = Trng.binomial(self.hidden_mean.shape, 1, self.hidden_mean)

    def gibbs_1(self, v_sample):
        # quick change of names internally: v_sample -> v0_sample
        v0_sample = v_sample; del v_sample

        h0_mean = T.nnet.sigmoid(T.dot(v0_sample, self.W) + self.hbias)
        h0_sample = self.theano_rng.binomial(h0_mean.shape, 1, h0_mean)
        v1_mean = T.nnet.sigmoid(T.dot(h0_sample, self.W.T) + self.vbias)
        v1_act = self.theano_rng.binomial(v1_mean.shape, 1, v1_mean)
        return v1_mean, v1_act

    def gibbs_k(self, k):
        def gibbs_steps(v_sample):
            v0_sample = v_sample; del v_sample
            h0_mean = T.nnet.sigmoid(T.dot(v0_sample, self.W) + self.hbias)
            h0_sample = self.theano_rng.binomial(h0_mean.shape, 1, h0_mean)
            v1_mean = T.nnet.sigmoid(T.dot(h0_sample, self.W.T) + self.vbias)
            v1_act = self.theano_rng.binomial(v1_mean.shape, 1, v1_mean)
 
            def gibbs_step(v_sample_tm1, v_mean_tm1 ):
                h_mean_t   = T.nnet.sigmoid(T.dot(v_sample_tm1, self.W) + self.hbias)
                h_sample_t = self.theano_rng.binomial(h_mean_t.shape, 1, h_mean_t)
                v_mean_t   = T.nnet.sigmoid(T.dot(h_sample_t, self.W.T) + self.vbias)
                v_sample_t = self.theano_rng.binomial(v_mean_t.shape, 1, v_mean_t)
                return v_sample_t, v_mean_t

            v_samples, v_means = scan(gibbs_step, [], [v1_act, v1_mean],[], \
                                                                n_steps = k-1)
            return v_means[-1], v_samples[-1]

    def free_energy(self, v_sample):
        h_mean = T.nnet.sigmoid(T.dot(v_sample, self.W) + self.hbias)
        #TODO: make sure log(sigmoid) is optimized to something stable!
        return -T.sum(T.log(1.0001-h_mean)) - T.sum(T.dot(v_sample, self.vbias))

    def cd(self, visible=None, persistent=None, step = None):
        """
        Return a 5-tuple of values related to contrastive divergence: (cost,
        end-state of negative-phase chain, gradient on weights, gradient on
        hidden bias, gradient on visible bias)

        If visible is None, it defaults to self.input
        If persistent is None, it defaults to self.input

        CD aka CD1 - cd()
        CD-10      - cd(step=gibbs_k(10))
        PCD        - cd(persistent=shared(numpy.asarray(initializer)))
        PCD-k      - cd(persistent=shared(numpy.asarray(initializer)),
                        step=gibbs_k(10))
        """
        if visible is None:
            visible = self.input

        if visible is None:
            raise TypeError('visible argument is required when self.input is None')

        if step is None:
            step = self.gibbs_1

        if persistent is None:
            chain_start = visible
        else:
            chain_start = persistent
        chain_end_mean, chain_end_sample = step(chain_start)

        cost = self.free_energy(visible) - self.free_energy(chain_end_sample)

        return (cost, chain_end_sample,) + tuple(T.grad(cost, [self.W, self.hbias, self.vbias]))

    def cd_updates(self, lr, visible=None, persistent=None, step = None):
        """
        Return the learning updates for the RBM parameters that are shared variables.

        Also returns an update for the persistent if it is a shared variable.

        These updates are returned as a dictionary.

        :param lr: [scalar] learning rate for contrastive divergence learning
        :param visible: see `cd_grad`
        :param persistent: see `cd_grad`
        :param step: see `cd_grad`

        """

        cost, chain_end, gW, ghbias, gvbias = self.cd(visible, persistent, step)

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

def test_RBM_option2(learning_rate=0.1, training_epochs = 20, 
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
    rbm = RBM_option2.new(input = x, n_visible=28*28, n_hidden=500, numpy_rng=
            numpy.random.RandomState(234234))
    step = rbm.gibbs_k(10) 
    cost = rbm.cd(step = step)[0]

    print '... compiling train function'
    train_rbm = theano.function([index], rbm.cd(step = step)[0], 
           updates = rbm.cd_updates(learning_rate, step = step), 
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
            print 'batch ', c[-1]
        print 'Training epoch %d '%epoch, numpy.mean(c)
 
    end_time = time.clock()

    print ('Training took %f minutes' %((end_time-start_time)/60.))

if __name__ == '__main__':
    test_RBM_option2()

