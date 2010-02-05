"""
This tutorial introduces the LeNet5 neural network architecture using Theano.  LeNet5 is a
convolutional neural network, good for classifying images. This tutorial shows how to build the
architecture, and comes with all the hyper-parameters you need to reproduce the paper's MNIST
results.

The best results are obtained after X iterations of the main program loop, which takes ***
minutes on my workstation (an Intel Core i7, circa July 2009), and *** minutes on my GPU (an
NVIDIA GTX 285 graphics processor).

This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling by max.
 - Digit classification is implemented with a logistic regression rather than an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""

import numpy, theano, cPickle, gzip, time
import theano.tensor as T
import theano.sandbox.softsign
import pylearn.datasets.MNIST
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LeNetConvPoolLayer(object):
    """WRITEME"""

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1]==filter_shape[1]
        self.input = input
  
        # initialize weights to temporary values until we know the shape of the output feature
        # maps
        W_values = numpy.zeros(filter_shape, dtype=theano.config.floatX)
        self.W = theano.shared(value = W_values)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype= theano.config.floatX)
        self.b = theano.shared(value= b_values)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input, self.W, 
                filter_shape=filter_shape, image_shape=image_shape)

        # there are "num input feature maps * filter height * filter width" inputs
        # to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize)
        # replace weight values with random weights
        W_bound = numpy.sqrt(6./(fan_in + fan_out))
        self.W.value = numpy.asarray( 
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype = theano.config.floatX)
  
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool2D(conv_out, poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will thus
        # be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


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

        self.output = T.tanh(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]


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


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20,50]):
    rng = numpy.random.RandomState(23455)

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

    batch_size = 500    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size
    n_valid_batches = valid_set_x.value.shape[0] / batch_size
    n_test_batches  = test_set_x.value.shape[0]  / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y') # the labels are presented as 1D vector of 
                           # [int] labels


    ishape = (28,28)     # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size,1,28,28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size,1,28,28), 
            filter_shape=(nkerns[0],1,5,5), poolsize=(2,2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size,nkerns[0],12,12), 
            filter_shape=(nkerns[1],nkerns[0],5,5), poolsize=(2,2))

    # the SigmoidalLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = SigmoidalLayer(rng, input=layer2_input, 
                            n_in=nkerns[1]*4*4, n_out=500)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens = {
                x: test_set_x[index*batch_size:(index+1)*batch_size],
                y: test_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens = {
                x: valid_set_x[index*batch_size:(index+1)*batch_size],
                y: valid_set_y[index*batch_size:(index+1)*batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params+ layer2.params+ layer1.params + layer0.params
    
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD
    # Since this model has many parameters, it would be tedious to manually
    # create an update rule for each model parameter. We thus create the updates
    # dictionary by automatically looping over all (params[i],grads[i])  pairs.
    updates = {}
    for param_i, grad_i in zip(params, grads):
        updates[param_i] = param_i - learning_rate * grad_i
    
    train_model = theano.function([index], cost, updates=updates,
          givens = {
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]})


    ###############
    # TRAIN MODEL #
    ###############

    # early-stopping parameters
    patience              = 10000 # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
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
    best_iter            = 0
    test_score           = 0.
    start_time = time.clock()

    for epoch in xrange(n_epochs):
      for minibatch_index in xrange(n_train_batches):
        
        iter = epoch * n_train_batches + minibatch_index

        if iter %100 == 0:
            print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

        if (iter+1) % validation_frequency == 0: 

            # compute zero-one loss on validation set
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
            break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %  
          (best_validation_loss * 100., best_iter, test_score*100.))
    print('The code ran for %f minutes' % ((end_time-start_time)/60.))

if __name__ == '__main__':
    evaluate_lenet5()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
