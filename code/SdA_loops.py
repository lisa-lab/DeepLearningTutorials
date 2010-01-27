"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SDAE. 
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

class dA():
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

  def __init__(self, n_visible= 784, n_hidden= 500, input= None):
    """
    Initialize the DAE class by specifying the number of visible units (the 
    dimension d of the input ), the number of hidden units ( the dimension 
    d' of the latent or hidden space ) and by giving a symbolic variable 
    for the input. Such a symbolic variable is useful when the input is 
    the result of some computations. For example when dealing with SDAEs,
    the dA on layer 2 gets as input the output of the DAE on layer 1. 
    This output can be written as a function of the input to the entire 
    model, and as such can be computed by theano whenever needed. 
    
    :param n_visible: number of visible units

    :param n_hidden:  number of hidden units

    :param input:     a symbolic description of the input or None 

    """
    self.n_visible = n_visible
    self.n_hidden  = n_hidden
    
    # create a Theano random generator that gives symbolic random values
    theano_rng = RandomStreams()
    # create a numpy random generator
    numpy_rng = numpy.random.RandomState()
    
     
    # initial values for weights and biases
    # note : W' was written as `W_prime` and b' as `b_prime`

    # W is initialized with `initial_W` which is uniformely sampled
    # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
    # the output of uniform if converted using asarray to dtype 
    # theano.config.floatX so that the code is runable on GPU
    initial_W = numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(n_visible+n_hidden)), \
              high = numpy.sqrt(6./(n_visible+n_hidden)), \
              size = (n_visible, n_hidden)), dtype = theano.config.floatX)
    initial_b       = numpy.zeros(n_hidden)
    initial_b_prime= numpy.zeros(n_visible)
     
    
    # theano shared variables for weights and biases
    self.W       = theano.shared(value = initial_W,       name = "W")
    self.b       = theano.shared(value = initial_b,       name = "b")
    # tied weights, therefore W_prime is W transpose
    self.W_prime = self.W.T 
    self.b_prime = theano.shared(value = initial_b_prime, name = "b'")

    # if no input is given, generate a variable representing the input
    if input == None : 
        # we use a matrix because we expect a minibatch of several examples,
        # each example being a row
        x = T.dmatrix(name = 'input') 
    else:
        x = input
    # Equation (1)
    # note : first argument of theano.rng.binomial is the shape(size) of 
    #        random numbers that it should produce
    #        second argument is the number of trials 
    #        third argument is the probability of success of any trial
    #
    #        this will produce an array of 0s and 1s where 1 has a 
    #        probability of 0.9 and 0 if 0.1
    tilde_x  = theano_rng.binomial( x.shape,  1,  0.9) * x
    # Equation (2)
    # note  : y is stored as an attribute of the class so that it can be 
    #         used later when stacking dAs. 
    self.y   = T.nnet.sigmoid(T.dot(tilde_x, self.W      ) + self.b)
    # Equation (3)
    z        = T.nnet.sigmoid(T.dot(self.y, self.W_prime) + self.b_prime)
    # Equation (4)
    self.L = - T.sum( x*T.log(z) + (1-x)*T.log(1-z), axis=1 ) 
    # note : L is now a vector, where each element is the cross-entropy cost 
    #        of the reconstruction of the corresponding example of the 
    #        minibatch. We need to compute the average of all these to get 
    #        the cost of the minibatch
    self.cost = T.mean(self.L)
    # note : y is computed from the corrupted `tilde_x`. Later on, 
    #        we will need the hidden layer obtained from the uncorrupted 
    #        input when for example we will pass this as input to the layer 
    #        above
    self.hidden_values = T.nnet.sigmoid( T.dot(x, self.W) + self.b)





class SdA():
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of 
    the dA at layer `i+1`. The first layer dA gets as input the input of 
    the SdA, and the hidden layer of the last dA represents the output. 
    Note that after pretraining, the SdA is dealt with as a normal MLP, 
    the dAs are only used to initialize the weights.
    """

    def __init__(self, input, n_ins, hidden_layers_sizes, n_outs):
        """ This class is costum made for a three layer SdA, and therefore
        is created by specifying the sizes of the hidden layers of the 
        3 dAs used to generate the network. 

        :param input: symbolic variable describing the input of the SdA

        :param n_ins: dimension of the input to the sdA

        :param n_layers_sizes: intermidiate layers size, must contain 
        at least one value

        :param n_outs: dimension of the output of the network
        """
        
        self.layers =[]

        if len(hidden_layers_sizes) < 1 :
            raiseException (' You must have at least one hidden layer ')

        # add first layer:
        layer = dA(n_ins, hidden_layers_sizes[0], input = input)
        self.layers += [layer]
        # add all intermidiate layers
        for i in xrange( 1, len(hidden_layers_sizes) ):
            # input size is that of the previous layer
            # input is the output of the last layer inserted in our list 
            # of layers `self.layers`
            layer = dA( hidden_layers_sizes[i-1],             \
                        hidden_layers_sizes[i],               \
                        input = self.layers[-1].hidden_values )
            self.layers += [layer]
        

        self.n_layers = len(self.layers)
        # now we need to use same weights and biases to define an MLP
        # We can simply use the `hidden_values` of the top layer, which 
        # computes the input that we would normally feed to the logistic
        # layer on top of the MLP and just add a logistic regression on 
        # this values
        
        # W is initialized with `initial_W` which is uniformely sampled
        # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        initial_W = numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(hidden_layers_sizes[-1]+n_outs)), \
              high = numpy.sqrt(6./(hidden_layers_sizes[-1]+n_outs)), \
              size = (hidden_layers_sizes[-1], n_outs)), \
                      dtype = theano.config.floatX)
    
        # theano shared variables for logistic layer weights and biases
        self.log_W  = theano.shared(value = initial_W,           name = "W")
        self.log_b  = theano.shared(value = numpy.zeros(n_outs), name = 'b')
        self.p_y_given_x = T.nnet.softmax( \
            T.dot(self.layers[-1].hidden_values, self.log_W) + self.log_b)
        
        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred = T.argmax( self.p_y_given_x, axis = 1)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|}\mathcal{L} (\theta=\{W,b\}, \mathcal{D}) = 
            \frac{1}{|\mathcal{D}|}\sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D}) 


        :param y: corresponds to a vector that gives for each example the
        :correct label
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch 
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

  

def sgd_optimization_mnist( learning_rate=0.01, pretraining_epochs = 2, \
                            pretraining_lr = 0.1, n_iter = 3):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer 
    perceptron

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used (factor for the stochastic 
    gradient

    :param pretraining_epochs: number of epoch to do pretraining

    :param pretrain_lr: learning rate to be used during pre-training

    :param n_iter: maximal number of iterations ot run the optimizer 

    """

    # Load the dataset 
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # make minibatches of size 20 
    batch_size = 20    # sized of the minibatch

    # Dealing with the training set
    # get the list of training images (x) and their labels (y)
    (train_set_x, train_set_y) = train_set
    # initialize the list of training minibatches with empty list
    train_batches = []
    for i in xrange(0, len(train_set_x), batch_size):
        # add to the list of minibatches the minibatch starting at 
        # position i, ending at position i+batch_size
        # a minibatch is a pair ; the first element of the pair is a list 
        # of datapoints, the second element is the list of corresponding 
        # labels
        train_batches = train_batches + \
               [(train_set_x[i:i+batch_size], train_set_y[i:i+batch_size])]

    # Dealing with the validation set
    (valid_set_x, valid_set_y) = valid_set
    # initialize the list of validation minibatches 
    valid_batches = []
    for i in xrange(0, len(valid_set_x), batch_size):
        valid_batches = valid_batches + \
               [(valid_set_x[i:i+batch_size], valid_set_y[i:i+batch_size])]

    # Dealing with the testing set
    (test_set_x, test_set_y) = test_set
    # initialize the list of testing minibatches 
    test_batches = []
    for i in xrange(0, len(test_set_x), batch_size):
        test_batches = test_batches + \
              [(test_set_x[i:i+batch_size], test_set_y[i:i+batch_size])]


    ishape     = (28,28) # this is the size of MNIST images

    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of 
                          # [long int] labels

    # construct the logistic regression class
    classifier = SdA( input=x.reshape((batch_size,28*28)),\
                      n_ins=28*28, hidden_layers_sizes = [500, 500],\
                      n_outs=10)
    
    ## Pre-train layer-wise 
    for i in xrange(classifier.n_layers):
        # compute gradients of layer parameters
        gW       = T.grad(classifier.layers[i].cost, classifier.layers[i].W)
        gb       = T.grad(classifier.layers[i].cost, classifier.layers[i].b)
        gb_prime = T.grad(classifier.layers[i].cost, \
                                               classifier.layers[i].b_prime)
        # updated value of parameters after each step
        new_W       = classifier.layers[i].W      - gW      * pretraining_lr
        new_b       = classifier.layers[i].b      - gb      * pretraining_lr
        new_b_prime = classifier.layers[i].b_prime- gb_prime* pretraining_lr
        layer_update = theano.function([x],classifier.layers[i].cost, \
              updates = { classifier.layers[i].W       : new_W \
                        , classifier.layers[i].b       : new_b \
                        , classifier.layers[i].b_prime : new_b_prime } )
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            for x_value,y_value in train_batches:
                layer_update(x_value)
            print 'Pre-training layer %i, epoch %d'%(i,epoch)



    # Fine-tune the entire model
    # the cost we minimize during training is the negative log likelihood of 
    # the model
    cost = classifier.negative_log_likelihood(y) 

    # compiling a theano function that computes the mistakes that are made  
    # by the model on a minibatch
    test_model = theano.function([x,y], classifier.errors(y))

    # compute the gradient of cost with respect to theta and add them to the 
    # updates list
    updates = []
    for i in xrange(classifier.n_layers):        
        g_W   = T.grad(cost, classifier.layers[i].W)
        g_b   = T.grad(cost, classifier.layers[i].b)
        new_W = classifier.layers[i].W - learning_rate * g_W
        new_b = classifier.layers[i].b - learning_rate * g_b
        updates += [ (classifier.layers[i].W, new_W) \
                   , (classifier.layers[i].b, new_b) ]
    # add the gradients of the logistic layer
    g_log_W   = T.grad(cost, classifier.log_W)
    g_log_b   = T.grad(cost, classifier.log_b)
    new_log_W = classifier.log_W - learning_rate * g_log_W
    new_log_b = classifier.log_b - learning_rate * g_log_b
    updates += [ (classifier.log_W, new_log_W) \
               , (classifier.log_b, new_log_b) ]

    # compiling a theano function `train_model` that returns the cost, but  
    # in the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model = theano.function([x, y], cost, updates = updates )
    n_minibatches        = len(train_batches) 
 
    # early-stopping parameters
    patience              = 10000 # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = n_minibatches  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time = time.clock()
    # have a maximum of `n_iter` iterations through the entire dataset
    for iter in xrange(n_iter* n_minibatches):

        # get epoch and minibatch index
        epoch           = iter / n_minibatches
        minibatch_index =  iter % n_minibatches

        # get the minibatches corresponding to `iter` modulo
        # `len(train_batches)`
        x,y = train_batches[ minibatch_index ]
        cost_ij = train_model(x,y)

        if (iter+1) % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            this_validation_loss = 0.
            for x,y in valid_batches:
                # sum up the errors for each minibatch
                this_validation_loss += test_model(x,y)
            # get the average by dividing with the number of minibatches
            this_validation_loss /= len(valid_batches)

            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_minibatches, \
                    this_validation_loss*100.))


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set
            
                test_score = 0.
                for x,y in test_batches:
                    test_score += test_model(x,y)
                test_score /= len(test_batches)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') % 
                             (epoch, minibatch_index+1, n_minibatches,
                              test_score*100.))

        if patience <= iter :
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))






if __name__ == '__main__':
    sgd_optimization_mnist()


