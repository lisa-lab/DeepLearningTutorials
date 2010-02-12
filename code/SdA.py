"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA. 
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting 
 latent representation y is then mapped back to a "reconstructed" vector 
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight 
 matrix W' can optionally be constrained such that W' = W^T, in which case 
 the autoencoder is said to have tied weights. The network is trained such 
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into 
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means 
 of a stochastic mapping. Afterwards y is computed as before (using 
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction 
 error is now measured between z and the uncorrupted input x, which is 
 computed as the cross-entropy : 
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

 For X iteration of the main program loop it takes *** minutes on an 
 Intel Core i7 and *** minutes on GPU (NVIDIA GTX 285 graphics processor).


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and 
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing 
   Systems 19, 2007

"""

import numpy 
import theano
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import gzip
import cPickle


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
        :type input: theano.tensor.dmatrix
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



class dA(object):
  """Denoising Auto-Encoder class (dA) 

  A denoising autoencoders tries to reconstruct the input from a corrupted 
  version of it by projecting it first in a latent space and reprojecting 
  it afterwards back in the input space. Please refer to Vincent et al.,2008
  for more details. If x is the input then equation (1) computes a partially
  destroyed version of x by means of a stochastic mapping q_D. Equation (2) 
  computes the projection of the input into the latent space. Equation (3) 
  computes the reconstruction of the input, while equation (4) computes the 
  reconstruction error.
  
  .. math::

    \tilde{x} ~ q_D(\tilde{x}|x)                                         (1)

    y = s(W \tilde{x} + b)                                               (2)

    x = s(W' y  + b')                                                    (3)

    L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]          (4)

  """

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
        self.x = T.dmatrix(name = 'input') 
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


class DeepNetwork()
   def pretrain( dataset )
   def finetune()


class SdA():
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of 
    the dA at layer `i+1`. The first layer dA gets as input the input of 
    the SdA, and the hidden layer of the last dA represents the output. 
    Note that after pretraining, the SdA is dealt with as a normal MLP, 
    the dAs are only used to initialize the weights.
    """

    def __init__(self, train_set_x, train_set_y, batch_size, n_ins, 
                 hidden_layers_sizes, n_outs, 
                 corruption_levels, rng, pretrain_lr, finetune_lr):
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
        
        self.layers             = []
        self.pretrain_functions = []
        self.params             = []
        self.n_layers           = len(hidden_layers_sizes)

        if len(hidden_layers_sizes) < 1 :
            raiseException (' You must have at least one hidden layer ')


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

            # the size of the input is either the number of hidden units of 
            # the layer below or the input size if we are on the first layer
            if i == 0 :
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0 : 
                layer_input = self.x
            else:
                layer_input = self.layers[-1].output

            layer = SigmoidalLayer(rng, layer_input, input_size, 
                                   hidden_layers_sizes[i] )
            # add the layer to the 
            self.layers += [layer]
            self.params += layer.params
        
            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(input_size, hidden_layers_sizes[i], \
                          corruption_level = corruption_levels[0],\
                          input = layer_input, \
                          shared_W = layer.W, shared_b = layer.b)
        
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

        
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(\
                         input = self.layers[-1].output,\
                         n_in = hidden_layers_sizes[-1], n_out = n_outs)

        self.params += self.logLayer.params
        # construct a function that implements one step of finetunining

        # compute the cost, defined as the negative log likelihood 
        cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, self.params)
        # compute list of updates
        updates = {}
        for param,gparam in zip(self.params, gparams):
            updates[param] = param - gparam*finetune_lr
            
        self.finetune = theano.function([index], cost, 
                updates = updates,
                givens = {
                  self.x : train_set_x[index*batch_size:(index+1)*batch_size],
                  self.y : train_set_y[index*batch_size:(index+1)*batch_size]} )

        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y

        self.errors = self.logLayer.errors(self.y)



def sgd_optimization_mnist( learning_rate=0.1, pretraining_epochs = 20, \
                            pretrain_lr = 0.1, training_epochs = 1000, \
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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    batch_size = 20    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size
    n_valid_batches = valid_set_x.value.shape[0] / batch_size
    n_test_batches  = test_set_x.value.shape[0]  / batch_size

    # allocate symbolic variables for the data
    index   = T.lscalar()    # index to a [mini]batch 
 


    # construct the stacked denoising autoencoder class
    classifier = SdA( train_set_x=train_set_x, train_set_y = train_set_y,\
                      batch_size = batch_size, n_ins=28*28, \
                      hidden_layers_sizes = [1000, 1000, 1000], n_outs=10, \
                      corruption_levels = [ 0.2, 0.2, 0.2],\
                      rng = numpy.random.RandomState(1234),\
                      pretrain_lr = pretrain_lr, finetune_lr = learning_rate )


    start_time = time.clock()  
    ## Pre-train layer-wise 
    for i in xrange(classifier.n_layers):
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            for batch_index in xrange(n_train_batches):
                c = classifier.pretrain_functions[i](batch_index)
            print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),c
 
    end_time = time.clock()

    print ('Pretraining took %f minutes' %((end_time-start_time)/60.))
    # Fine-tune the entire model


    # create a function to compute the mistakes that are made by the model
    # on the validation set, or testing set
    test_model = theano.function([index], classifier.errors,
             givens = {
               classifier.x: test_set_x[index*batch_size:(index+1)*batch_size],
               classifier.y: test_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function([index], classifier.errors,
            givens = {
               classifier.x: valid_set_x[index*batch_size:(index+1)*batch_size],
               classifier.y: valid_set_y[index*batch_size:(index+1)*batch_size]})


    # early-stopping parameters
    patience              = 10000 # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(n_train_batches):

        cost_ij = classifier.finetune(minibatch_index)
        iter    = epoch * n_train_batches + minibatch_index

        if (iter+1) % validation_frequency == 0: 
            
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_train_batches, \
                    this_validation_loss*100.))


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') % 
                             (epoch, minibatch_index+1, n_train_batches,
                              test_score*100.))


        if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))






if __name__ == '__main__':
    sgd_optimization_mnist()


