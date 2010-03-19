"""
Draft of DBN, DAA, SDAA, RBM tutorial code

"""
import sys
import numpy 
import theano
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared, function

import gzip
import cPickle
import pylearn.io.image_tiling
import PIL

# NNET STUFF

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W` 
    and bias vector :math:`b`. Classification is done by projecting data 
    points onto a set of hyperplanes, the distance to which is used to 
    determine a class membership probability. 
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :param input: symbolic variable that describes the input of the 
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
                     which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
                      which the labels lie
        """ 

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
        self.W = theano.shared( value=numpy.zeros((n_in,n_out),
                                            dtype = theano.config.floatX) )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared( value=numpy.zeros((n_out,), 
                                            dtype = theano.config.floatX) )
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)
        
        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        # list of parameters for this layer
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
        the learning rate is less dependent on the batch size
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        """
        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class SigmoidalLayer(object):
    def __init__(self, rng, input, n_in, n_out):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        
        Hidden unit activation is given by: sigmoid(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.matrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        """
        self.input = input

        W_values = numpy.asarray( rng.uniform( \
              low = -numpy.sqrt(6./(n_in+n_out)), \
              high = numpy.sqrt(6./(n_in+n_out)), \
              size = (n_in, n_out)), dtype = theano.config.floatX)
        self.W = theano.shared(value = W_values)

        b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
        self.b = theano.shared(value= b_values)

        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]

# PRETRAINING LAYERS

class RBM(object):
    """
    *** WRITE THE ENERGY FUNCTION  USE SAME LETTERS AS VARIABLE NAMES IN CODE
    """

    def __init__(self, input=None, n_visible=None, n_hidden=None,
            W=None, hbias=None, vbias=None,
            numpy_rng=None, theano_rng=None):
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
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                dtype=theano.config.floatX), name='hbias')
            params.append(hbias)

        if vbias is None:
            # theano shared variables for visible biases
            vbias = theano.shared(value=numpy.zeros(n_visible,
                dtype=theano.config.floatX), name='vbias')
            params.append(vbias)

        if input is None:
            # initialize input layer for standalone RBM or layer0 of DBN
            input = T.matrix('input')

        # setup theano random number generator
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        self.visible = self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng 
        self.params = params
        self.hidden_mean = T.nnet.sigmoid(T.dot(input, W)+hbias)
        self.hidden_sample = theano_rng.binomial(self.hidden_mean.shape, 1, self.hidden_mean)

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
            return [v1_mean, v1_act]


        # DEBUGGING TO DO ALL WITHOUT SCAN
        if k == 1:
            return gibbs_1(v_sample, self.W, self.hbias, self.vbias)
       
       
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

        v_means, v_samples = theano.scan( fn = gibbs_1, 
                                          sequences      = [], 
                                          initial_states = [v_sample, v_mean],
                                          non_sequences  = [self.W, self.hbias, self.vbias], 
                                          outputs_taps   = outputs_taps,
                                          n_steps        = k)
        return v_means[-1], v_samples[-1]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.sum(T.dot(v_sample, self.vbias))
        hidden_term = T.sum(T.log(1+T.exp(wx_b)))
        return -hidden_term - vbias_term

    def cd(self, visible = None, persistent = None, steps = 1):
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

        if steps is None:
            steps = self.gibbs_1

        if persistent is None:
            chain_start = visible
        else:
            chain_start = persistent

        chain_end_mean, chain_end_sample = self.gibbs_k(chain_start, steps)

        #print >> sys.stderr, "WARNING: DEBUGGING with wrong FREE ENERGY"
        #free_energy_delta = - self.free_energy(chain_end_sample)
        free_energy_delta = self.free_energy(visible) - self.free_energy(chain_end_sample)

        # we will return all of these regardless of what is in self.params
        all_params = [self.W, self.hbias, self.vbias]

        gparams = T.grad(free_energy_delta, all_params, 
                consider_constant = [chain_end_sample])

        cross_entropy = T.mean(T.sum(
            visible*T.log(chain_end_mean) + (1 - visible)*T.log(1-chain_end_mean),
            axis = 1))

        return (cross_entropy, chain_end_sample,) + tuple(gparams)

    def cd_updates(self, lr, visible = None, persistent = None, steps = 1):
        """
        Return the learning updates for the RBM parameters that are shared variables.

        Also returns an update for the persistent if it is a shared variable.

        These updates are returned as a dictionary.

        :param lr: [scalar] learning rate for contrastive divergence learning
        :param visible: see `cd_grad`
        :param persistent: see `cd_grad`
        :param steps: see `cd_grad`

        """

        cross_entropy, chain_end, gW, ghbias, gvbias = self.cd(visible,
                persistent, steps)

        updates = {}
        if hasattr(self.W, 'value'):
            updates[self.W] = self.W - lr * gW
        if hasattr(self.hbias, 'value'):
            updates[self.hbias] = self.hbias - lr * ghbias
        if hasattr(self.vbias, 'value'):
            updates[self.vbias] = self.vbias - lr * gvbias
        if persistent:
            #if persistent is a shared var, then it means we should use
            updates[persistent] = chain_end

        return updates

# DEEP MODELS 

class DBN(object):
    """
    *** WHAT IS A DBN?
    """

    def __init__(self, input_len, hidden_layers_sizes, n_classes, rng):
        """ This class is made to support a variable number of layers. 

        :param train_set_x: symbolic variable pointing to the training dataset 

        :param train_set_y: symbolic variable pointing to the labels of the
        training dataset

        :param input_len: dimension of the input to the sdA

        :param n_layers_sizes: intermidiate layers size, must contain 
        at least one value

        :param n_classes: dimension of the output of the network

        :param corruption_levels: amount of corruption to use for each 
        layer

        :param rng: numpy random number generator used to draw initial weights

        :param pretrain_lr: learning rate used during pre-trainnig stage

        :param finetune_lr: learning rate used during finetune stage
        """
        
        self.sigmoid_layers     = []
        self.rbm_layers         = []
        self.pretrain_functions = []
        self.params             = []

        theano_rng = RandomStreams(rng.randint(2**30))

        # allocate symbolic variables for the data
        index   = T.lscalar()    # index to a [mini]batch 
        self.x  = T.matrix('x')  # the data is presented as rasterized images
        self.y  = T.ivector('y') # the labels are presented as 1D vector of 
                                 # [int] labels
        input = self.x

        # The SdA is an MLP, for which all weights of intermidiate layers
        # are shared with a different denoising autoencoders 
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a 
        # denoising autoencoder that shares weights with that layer, and 
        # compile a training function for that denoising autoencoder

        for n_hid in hidden_layers_sizes:
            # construct the sigmoidal layer

            sigmoid_layer = SigmoidalLayer(rng, input, input_len, n_hid)
            self.sigmoid_layers.append(sigmoid_layer)

            self.rbm_layers.append(RBM(input=input,
                W=sigmoid_layer.W,
                hbias=sigmoid_layer.b,
                n_visible = input_len,
                n_hidden = n_hid,
                numpy_rng=rng,
                theano_rng=theano_rng))

            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the 
            # sigmoid_layers are parameters of the StackedDAA
            # the hidden-layer biases in the daa_layers are parameters of those
            # daa_layers, but not the StackedDAA
            self.params.extend(self.sigmoid_layers[-1].params)

            # get ready for the next loop iteration
            input_len = n_hid
            input = self.sigmoid_layers[-1].output
        
        # We now need to add a logistic layer on top of the MLP
        self.logistic_regressor = LogisticRegression(input = input,
                n_in = input_len, n_out = n_classes)

        self.params.extend(self.logistic_regressor.params)

    def pretraining_functions(self, train_set_x, batch_size, learning_rate, k=1):
        if k!=1:
            raise NotImplementedError()
        index   = T.lscalar()    # index to a [mini]batch 
        n_train_batches = train_set_x.value.shape[0] / batch_size
        batch_begin = (index % n_train_batches) * batch_size
        batch_end = batch_begin+batch_size

        print 'TRAIN_SET X', train_set_x.value.shape
        rval = []
        for rbm in self.rbm_layers:
            # N.B. these cd() samples are independent from the
            # samples used for learning
            outputs = list(rbm.cd())[0:2]
            rval.append(function([index], outputs, 
                    updates = rbm.cd_updates(lr=learning_rate),
                    givens = {self.x: train_set_x[batch_begin:batch_end]}))
            if rbm is self.rbm_layers[0]:
                f = rval[-1]
                AA=len(outputs)
                for i, implicit_out in enumerate(f.maker.env.outputs): #[len(outputs):]:
                    print 'OUTPUT ', i
                    theano.printing.debugprint(implicit_out, file=sys.stdout)
                
        return rval

    def finetune(self, datasets, lr, batch_size):

        # unpack the various datasets
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        assert train_set_x.value.shape[0] % batch_size == 0
        assert valid_set_x.value.shape[0] % batch_size == 0
        assert test_set_x.value.shape[0] % batch_size == 0
        n_train_batches = train_set_x.value.shape[0] / batch_size
        n_valid_batches = valid_set_x.value.shape[0] / batch_size
        n_test_batches  = test_set_x.value.shape[0]  / batch_size

        index   = T.lscalar()    # index to a [mini]batch 
        target = self.y

        train_index = index % n_train_batches

        classifier = self.logistic_regressor
        cost = classifier.negative_log_likelihood(target)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, self.params)

        # compute list of fine-tuning updates
        updates = [(param, param - gparam*finetune_lr)
                for param,gparam in zip(self.params, gparams)]

        train_fn = theano.function([index], cost, 
                updates = updates,
                givens = {
                  self.x : train_set_x[train_index*batch_size:(train_index+1)*batch_size],
                  target : train_set_y[train_index*batch_size:(train_index+1)*batch_size]})

        test_score_i = theano.function([index], classifier.errors(target),
                 givens = {
                   self.x: test_set_x[index*batch_size:(index+1)*batch_size],
                   target: test_set_y[index*batch_size:(index+1)*batch_size]})

        valid_score_i = theano.function([index], classifier.errors(target),
                givens = {
                   self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
                   target: valid_set_y[index*batch_size:(index+1)*batch_size]})

        def test_scores():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        def valid_scores():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        return train_fn, valid_scores, test_scores

def load_mnist(filename):
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')

    n_train_examples = train_set[0].shape[0]
    datasets = shared_dataset(train_set), shared_dataset(valid_set), shared_dataset(test_set)

    return n_train_examples, datasets

def dbn_main(finetune_lr = 0.01,
        pretraining_epochs = 10,
        pretrain_lr = 0.1,
        training_epochs = 1000,
        batch_size = 20,
        mnist_file='mnist.pkl.gz'):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used in the finetune stage 
    (factor for the stochastic gradient)

    :param pretraining_epochs: number of epoch to do pretraining

    :param pretrain_lr: learning rate to be used during pre-training

    :param n_iter: maximal number of iterations ot run the optimizer 

    :param mnist_file: path the the pickled mnist_file

    """

    n_train_examples, train_valid_test = load_mnist(mnist_file)

    print "Creating a Deep Belief Network"
    deep_model = DBN(
            input_len=28*28,
            hidden_layers_sizes = [500, 150, 100],
            n_classes=10,
            rng = numpy.random.RandomState())

    ####
    #### Phase 1: Pre-training
    ####
    print "Pretraining (unsupervised learning) ..."

    pretrain_functions = deep_model.pretraining_functions(
            batch_size=batch_size,
            train_set_x=train_valid_test[0][0],
            learning_rate=pretrain_lr,
            )

    start_time = time.clock()  
    for layer_idx, pretrain_fn in enumerate(pretrain_functions):
        # go through pretraining epochs 
        print 'Pre-training layer %i'% layer_idx
        for i in xrange(pretraining_epochs * n_train_examples / batch_size):
            outstuff = pretrain_fn(i)
            xe, negsample = outstuff[:2]
            print (layer_idx, i,
                    n_train_examples / batch_size,
                    float(xe),
                    'Wmin', deep_model.rbm_layers[0].W.value.min(),
                    'Wmax', deep_model.rbm_layers[0].W.value.max(),
                    'vmin', deep_model.rbm_layers[0].vbias.value.min(),
                    'vmax', deep_model.rbm_layers[0].vbias.value.max(),
                    #'x>0.3', (input_i>0.3).sum(),
                    )
            sys.stdout.flush()
            if i % 1000 == 0:
                PIL.Image.fromarray(
                    pylearn.io.image_tiling.tile_raster_images(negsample, (28,28), (10,10),
                            tile_spacing=(1,1))).save('samples_%i_%i.png'%(layer_idx,i))

                PIL.Image.fromarray(
                    pylearn.io.image_tiling.tile_raster_images(
                        deep_model.rbm_layers[0].W.value.T,
                        (28,28), (10,10),
                        tile_spacing=(1,1))).save('filters_%i_%i.png'%(layer_idx,i))
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm expected Xm our buildbot' % (((end_time - start_time))/60.))

    return

    print "Fine tuning (supervised learning) ..."
    train_fn, valid_scores, test_scores =\
        deep_model.finetune_functions(train_valid_test[0][0],
            learning_rate=finetune_lr,      # the learning rate
            batch_size = batch_size)        # number of examples to use at once

    ####
    #### Phase 2: Fine Tuning
    ####

    patience              = 10000 # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_examples, patience/2)
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 

    patience_max = n_train_examples * training_epochs

    best_epoch               = None 
    best_epoch_test_score    = None
    best_epoch_valid_score   = float('inf')
    start_time               = time.clock()

    for i in xrange(patience_max):
        if i >= patience:
            break

        cost_i = train_fn(i)

        if i % validation_frequency == 0:
            validation_i = numpy.mean([score for score in valid_scores()])

            # if we got the best validation score until now
            if validation_i < best_epoch_valid_score:

                # improve patience if loss improvement is good enough
                threshold_i = best_epoch_valid_score * improvement_threshold
                if validation_i < threshold_i:
                    patience = max(patience, i * patience_increase)

                # save best validation score and iteration number
                best_epoch_valid_score = validation_i
                best_epoch = i/validation_i
                best_epoch_test_score = numpy.mean(
                        [score for score in test_scores()])

                print('epoch %i, validation error %f %%, test error %f %%'%(
                    i/validation_frequency, validation_i*100.,
                    best_epoch_test_score*100.))
            else:
                print('epoch %i, validation error %f %%' % (
                    i/validation_frequency, validation_i*100.))
    end_time = time.clock()

    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (finetune_status['best_validation_loss']*100.,
                     finetune_status['test_score']*100.))
    print >> sys.stderr, ('The fine tuning code for file '+os.path.split(__file__)[1]+' ran for %.2fm expected Xm our buildbot' % ((end_time-start_time)/60.))


def rbm_main():
    rbm = RBM(n_visible=20, n_hidden=30,
            numpy_rng = numpy.random.RandomState(34))

    cd_updates = rbm.cd_updates(lr=0.25)

    print cd_updates

    f = function([rbm.input], [],
            updates={rbm.W:cd_updates[rbm.W]})

    theano.printing.debugprint(f.maker.env.outputs[0],
            file=sys.stdout)


if __name__ == '__main__':
    dbn_main()
    #rbm_main()


if 0:
    class DAA(object):
      def __init__(self, n_visible= 784, n_hidden= 500, corruption_level = 0.1,\
                   input = None, shared_W = None, shared_b = None):
        """
        Initialize the dA class by specifying the number of visible units (the 
        dimension d of the input ), the number of hidden units ( the dimension 
        d' of the latent or hidden space ) and the corruption level. The 
        constructor also receives symbolic variables for the input, weights and 
        bias. Such a symbolic variables are useful when, for example the input is 
        the result of some computations, or when weights are shared between the 
        dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1, 
        and the weights of the dA are used in the second stage of training 
        to construct an MLP.
        
        :param n_visible: number of visible units

        :param n_hidden:  number of hidden units

        :param input:     a symbolic description of the input or None 

        :param corruption_level: the corruption mechanism picks up randomly this 
        fraction of entries of the input and turns them to 0
        
        
        """
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        
        # create a Theano random generator that gives symbolic random values
        theano_rng = RandomStreams()
        
        if shared_W != None and shared_b != None : 
            self.W = shared_W
            self.b = shared_b
        else:
            # initial values for weights and biases
            # note : W' was written as `W_prime` and b' as `b_prime`

            # W is initialized with `initial_W` which is uniformely sampled
            # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
            # the output of uniform if converted using asarray to dtype 
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray( numpy.random.uniform( \
                  low = -numpy.sqrt(6./(n_hidden+n_visible)), \
                  high = numpy.sqrt(6./(n_hidden+n_visible)), \
                  size = (n_visible, n_hidden)), dtype = theano.config.floatX)
            initial_b       = numpy.zeros(n_hidden, dtype = theano.config.floatX)
        
        
            # theano shared variables for weights and biases
            self.W       = theano.shared(value = initial_W,       name = "W")
            self.b       = theano.shared(value = initial_b,       name = "b")
        
     
        initial_b_prime= numpy.zeros(n_visible)
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T 
        self.b_prime = theano.shared(value = initial_b_prime, name = "b'")

        # if no input is given, generate a variable representing the input
        if input == None : 
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.matrix(name = 'input') 
        else:
            self.x = input
        # Equation (1)
        # keep 90% of the inputs the same and zero-out randomly selected subset of 10% of the inputs
        # note : first argument of theano.rng.binomial is the shape(size) of 
        #        random numbers that it should produce
        #        second argument is the number of trials 
        #        third argument is the probability of success of any trial
        #
        #        this will produce an array of 0s and 1s where 1 has a 
        #        probability of 1 - ``corruption_level`` and 0 with
        #        ``corruption_level``
        self.tilde_x  = theano_rng.binomial( self.x.shape,  1,  1 - corruption_level) * self.x
        # Equation (2)
        # note  : y is stored as an attribute of the class so that it can be 
        #         used later when stacking dAs. 
        self.y   = T.nnet.sigmoid(T.dot(self.tilde_x, self.W      ) + self.b)
        # Equation (3)
        self.z   = T.nnet.sigmoid(T.dot(self.y, self.W_prime) + self.b_prime)
        # Equation (4)
        # note : we sum over the size of a datapoint; if we are using minibatches,
        #        L will  be a vector, with one entry per example in minibatch
        self.L = - T.sum( self.x*T.log(self.z) + (1-self.x)*T.log(1-self.z), axis=1 ) 
        # note : L is now a vector, where each element is the cross-entropy cost 
        #        of the reconstruction of the corresponding example of the 
        #        minibatch. We need to compute the average of all these to get 
        #        the cost of the minibatch
        self.cost = T.mean(self.L)

        self.params = [ self.W, self.b, self.b_prime ]

    class StackedDAA(DeepLayerwiseModel):
        """Stacked denoising auto-encoder class (SdA)

        A stacked denoising autoencoder model is obtained by stacking several
        dAs. The hidden layer of the dA at layer `i` becomes the input of 
        the dA at layer `i+1`. The first layer dA gets as input the input of 
        the SdA, and the hidden layer of the last dA represents the output. 
        Note that after pretraining, the SdA is dealt with as a normal MLP, 
        the dAs are only used to initialize the weights.
        """

        def __init__(self, n_ins, hidden_layers_sizes, n_outs, 
                     corruption_levels, rng, ):
            """ This class is made to support a variable number of layers. 

            :param train_set_x: symbolic variable pointing to the training dataset 

            :param train_set_y: symbolic variable pointing to the labels of the
            training dataset

            :param n_ins: dimension of the input to the sdA

            :param n_layers_sizes: intermidiate layers size, must contain 
            at least one value

            :param n_outs: dimension of the output of the network

            :param corruption_levels: amount of corruption to use for each 
            layer

            :param rng: numpy random number generator used to draw initial weights

            :param pretrain_lr: learning rate used during pre-trainnig stage

            :param finetune_lr: learning rate used during finetune stage
            """
            
            self.sigmoid_layers     = []
            self.daa_layers         = []
            self.pretrain_functions = []
            self.params             = []
            self.n_layers           = len(hidden_layers_sizes)

            if len(hidden_layers_sizes) < 1 :
                raiseException (' You must have at least one hidden layer ')

            theano_rng = RandomStreams(rng.randint(2**30))

            # allocate symbolic variables for the data
            index   = T.lscalar()    # index to a [mini]batch 
            self.x  = T.matrix('x')  # the data is presented as rasterized images
            self.y  = T.ivector('y') # the labels are presented as 1D vector of 
                                     # [int] labels

            # The SdA is an MLP, for which all weights of intermidiate layers
            # are shared with a different denoising autoencoders 
            # We will first construct the SdA as a deep multilayer perceptron,
            # and when constructing each sigmoidal layer we also construct a 
            # denoising autoencoder that shares weights with that layer, and 
            # compile a training function for that denoising autoencoder

            for i in xrange( self.n_layers ):
                # construct the sigmoidal layer

                sigmoid_layer = SigmoidalLayer(rng,
                        self.layers[-1].output if i else self.x,
                        hidden_layers_sizes[i-1] if i else n_ins, 
                        hidden_layers_sizes[i])

                daa_layer = DAA(corruption_level = corruption_levels[i],
                              input = sigmoid_layer.input,
                              W = sigmoid_layer.W, 
                              b = sigmoid_layer.b)

                # add the layer to the 
                self.sigmoid_layers.append(sigmoid_layer)
                self.daa_layers.append(daa_layer)

                # its arguably a philosophical question...
                # but we are going to only declare that the parameters of the 
                # sigmoid_layers are parameters of the StackedDAA
                # the hidden-layer biases in the daa_layers are parameters of those
                # daa_layers, but not the StackedDAA
                self.params.extend(sigmoid_layer.params)
            
            # We now need to add a logistic layer on top of the MLP
            self.logistic_regressor = LogisticRegression(
                             input = self.sigmoid_layers[-1].output,
                             n_in = hidden_layers_sizes[-1],
                             n_out = n_outs)

            self.params.extend(self.logLayer.params)

        def pretraining_functions(self, train_set_x, batch_size):

            # compiles update functions for each layer, and
            # returns them as a list
            # 
            # Construct a function that trains this dA
            # compute gradients of layer parameters
            gparams = T.grad(dA_layer.cost, dA_layer.params)
            # compute the list of updates
            updates = {}
            for param, gparam in zip(dA_layer.params, gparams):
                updates[param] = param - gparam * pretrain_lr
            
            # create a function that trains the dA
            update_fn = theano.function([index], dA_layer.cost, \
                  updates = updates,
                  givens = { 
                     self.x : train_set_x[index*batch_size:(index+1)*batch_size]})
            # collect this function into a list
            self.pretrain_functions += [update_fn]


