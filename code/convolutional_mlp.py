
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
from theano.sandbox import conv, downsample

class LeNetConvPoolLayer(object):
    """WRITEME 

    Math of what the layer does, and what symbolic variables are created by the class (w, b,
    output).

    """

    #TODO: implement biases & scales properly. There are supposed to be more parameters.
    #    - one bias & scale per filter
    #    - one bias & scale per downsample feature location (a 2d bias)
    #    - more?

    def __init__(self, rng, input, n_imgs, n_filters, filter_shape=(5,5),
            poolsize=(2,2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :param rng: a random number generator used to initialize weights
        
        :param input: symbolic images.  Shape: (<mini-batch size>, n_imgs, <img height>, <img width>)

        :param n_imgs: input's shape[1] at runtime

        :param n_filters: the number of filters to apply to the image.

        :param filter_shape: the size of the filters to apply
        :type filter_shape: pair (rows, cols)

        :param poolsize: the downsampling (pooling) factor
        :type poolsize: pair (rows, cols)
        """

        # the filter tensor that we will apply is a 4D tensor
        w_shp = (n_filters, n_imgs) + filter_shape
        w_bound =  numpy.sqrt(filter_shape[0] * filter_shape[1] * n_imgs)
        self.w = theano.shared( numpy.asarray(
                    rng.uniform(
                        low=-1.0 / w_bound, 
                        high=1.0 / w_bound,
                        size=w_shp), 
                    dtype=input.dtype))

        # the bias we add is a 1D tensor
        b_shp = (n_filters,)
        self.b = theano.shared( numpy.asarray(
                    rng.uniform(low=-.0, high=0., size=(n_filters,)),
                    dtype=input.dtype))

        self.input = input
        conv_out = conv.conv2d(input, self.w)

        # - why is poolsize an op parameter here?
        # - can we just have a maxpool function that creates this Op internally?
        ds_op = downsample.DownsampleFactorMax(poolsize, ignore_border=True)
        self.output = T.tanh(ds_op(conv_out) + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.w, self.b]


class SigmoidalLayer(object):
    def __init__(self, rng, input, n_in, n_out):
        """
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param w: a symbolic weight matrix of shape (n_in, n_out)
        :param b: symbolic bias terms of shape (n_out,)
        :param squash: an squashing function
        """
        self.input = input
        self.w = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-2/numpy.sqrt(n_in), high=2/numpy.sqrt(n_in),
                    size=(n_in, n_out)), dtype=input.dtype))
        self.b = theano.shared(numpy.asarray(numpy.zeros(n_out), dtype=input.dtype))
        self.output = T.tanh(T.dot(input, self.w) + self.b)
        self.params = [self.w, self.b]

class LogisticRegression(object):
    """WRITEME"""

    def __init__(self, input, n_in, n_out):
        self.w = theano.shared(numpy.zeros((n_in, n_out), dtype=input.dtype))
        self.b = theano.shared(numpy.zeros((n_out,), dtype=input.dtype))
        self.l1 = abs(self.w).sum()
        self.l2_sqr = (self.w**2).sum()
        self.output = T.nnet.softmax(theano.dot(input, self.w)+self.b)
        self.argmax = T.argmax(self.output, axis=1)
        self.params = [self.w, self.b]

    def nll(self, target):
        """Return the negative log-likelihood of the prediction of this model under a given
        target distribution.  Passing symbolic integers here means 1-hot.
        WRITEME
        """
        return T.nnet.categorical_crossentropy(self.output, target)

    def errors(self, target):
        """Return a vector of 0s and 1s, with 1s on every line that was mis-classified.
        """
        if target.ndim != self.argmax.ndim:
            raise TypeError('target should have the same shape as self.argmax', ('target', target.type,
                'argmax', self.argmax.type))
        if target.dtype.startswith('int'):
            return T.neq(self.argmax, target)
        else:
            raise NotImplementedError()

def load_dataset():

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

    return train_batches, valid_batches, test_batches


def evaluate_lenet5(learning_rate=0.01, n_iter=1000):

    rng = numpy.random.RandomState(23455)

    train_batches, valid_batches, test_batches = load_dataset()

    ishape = (28,28)     # this is the size of MNIST images
    batch_size = 20    # sized of the minibatch

    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of [long int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################

    # construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(rng, input=x.reshape((batch_size,1,28,28)),
            n_imgs=1, n_filters=6, filter_shape=(5,5), poolsize=(2,2))

    # construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            n_imgs=6, n_filters=16, filter_shape=(5,5), poolsize=(2,2))

    # construct a fully-connected sigmoidal layer
    layer2 = SigmoidalLayer(rng, input=layer1.output.flatten(2), n_in=16*4*4, n_out=128) # 128 ?

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=128, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.nll(y).mean()

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([x,y], layer3.errors(y))

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params+ layer2.params+ layer1.params + layer0.params
    learning_rate = numpy.asarray(learning_rate, dtype='float32')

    # train_model is a function that updates the model parameters by SGD
    train_model = theano.function([x, y], cost, 
            updates=[(p, p - learning_rate*gp) for p,gp in zip(params, T.grad(cost, params))])


    ###############
    # TRAIN MODEL #
    ###############

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
    for iter in xrange(n_iter * n_minibatches):

        # get epoch and minibatch index
        epoch           = iter / n_minibatches
        minibatch_index =  iter % n_minibatches

        # get the minibatches corresponding to `iter` modulo
        # `len(train_batches)`
        x,y = train_batches[ minibatch_index ]

        print 'training @ iter = ', iter
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
    evaluate_lenet5()

