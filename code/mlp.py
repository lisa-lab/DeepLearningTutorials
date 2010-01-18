"""
This tutorial introduces the multilayer perceptron using Theano.  

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermidiate layer, called the hidden layer, that has a nonlinear 
activation function (usually tanh or sigmoid) . One can use many such 
hidden layers making the architecture deep. The tutorial will also tackle 
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 5

TODO: recommended preprocessing, lr ranges, regularization ranges (explain 
      to do lr first, then add regularization)

"""
__docformat__ = 'restructedtext en'


import numpy, cPickle, gzip


import theano
import theano.tensor as T

import time 

import theano.tensor.nnet

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermidiate layers usually have as activation function thanh or the 
    sigmoid function  while the top layer is a softamx layer. 
    """



    def __init__(self, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :param input: symbolic variable that describes the input of the 
        architecture (one minibatch)

        :param n_in: number of input units, the dimension of the space in 
        which the datapoints lie

        :param n_hidden: number of hidden units 

        :param n_out: number of output units, the dimension of the space in 
        which the labels lie

        """

        # initialize the parameters theta = (W1,b1,W2,b2) ; note that this 
        # example contains only one hidden layer, but one can have as many 
        # layers as he/she wishes, making the network deeper. The only 
        # problem making the network deep this way is during learning, 
        # backpropagation being unable to move the network from the starting
        # point towards; this is where pre-training helps, giving a good 
        # starting point for backpropagation, but more about this in the 
        # other tutorials
        
        # `W1` is initialized with `W1_values` which is uniformely sampled
        # from -1/sqrt(n_in) and 1/sqrt(n_in)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        W1_values = numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(n_in+n_hidden)), high = numpy.sqrt(6./(n_in+n_hidden)), \
              size = (n_in, n_hidden)), dtype = theano.config.floatX)
        # `W2` is initialized with `W2_values` which is uniformely sampled 
        # from -1/sqrt(n_hidden) and 1/sqrt(n_hidden)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        W2_values = numpy.asarray( numpy.random.uniform( 
              low = numpy.sqrt(6./(n_hidden+n_out)), high= numpy.sqrt(6./(n_hidden+n_out)),\
              size= (n_hidden, n_out)), dtype = theano.config.floatX)

        self.W1 = theano.shared( value = W1_values )
        self.b1 = theano.shared( value = numpy.zeros((n_hidden,), 
                                                dtype= theano.config.floatX))
        self.W2 = theano.shared( value = W2_values )
        self.b2 = theano.shared( value = numpy.zeros((n_out,), 
                                                dtype= theano.config.floatX))

        # symbolic expression computing the values of the hidden layer
        self.hidden = T.tanh(T.dot(input, self.W1)+ self.b1)

        # symbolic expression computing the values of the top layer 
        self.p_y_given_x= T.nnet.softmax(T.dot(self.hidden, self.W2)+self.b2)

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred = T.argmax( self.p_y_given_x, axis =1)
        
        # L1 norm ; one regularization option is to enforce L1 norm to 
        # be small 
        self.L1     = abs(self.W1).sum() + abs(self.W2).sum()

        # square of L2 norm ; one regularization option is to enforce 
        # square of L2 norm to be small
        self.L2_sqr = (self.W1**2).sum() + (self.W2**2).sum()



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



def sgd_optimization_mnist( learning_rate=0.01, L1_reg = 0.0, \
                            L2_reg = 0.0, n_iter=100):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer 
    perceptron

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used (factor for the stochastic 
    gradient

    :param n_iter: number of iterations ot run the optimizer 

    :param L1_reg: L1-norm's weight when added to the cost (see 
    regularization)

    :param L2_reg: L2-norm's weight when added to the cost (see 
    regularization)
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
    classifier = MLP( input=x.reshape((batch_size,28*28)),\
                      n_in=28*28, n_hidden = 100, n_out=10)

    # the cost we minimize during training is the negative log likelihood of 
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr 

    # compiling a theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function([x,y], classifier.errors(y))

    # compute the gradient of cost with respect to theta = (W1, b1, W2, b2) 
    g_W1 = T.grad(cost, classifier.W1)
    g_b1 = T.grad(cost, classifier.b1)
    g_W2 = T.grad(cost, classifier.W2)
    g_b2 = T.grad(cost, classifier.b2)

    # specify how to update the parameters of the model as a dictionary
    updates = \
        { classifier.W1: classifier.W1 - learning_rate*g_W1 \
        , classifier.b1: classifier.b1 - learning_rate*g_b1 \
        , classifier.W2: classifier.W2 - learning_rate*g_W2 \
        , classifier.b2: classifier.b2 - learning_rate*g_b2 }

    # compiling a theano function `train_model` that returns the cost, but in 
    # the same time updates the parameter of the model based on the rules 
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

