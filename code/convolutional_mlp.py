
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
import numpy
from theano.compile.sandbox import shared, pfunc
from theano import tensor
from pylearn.shared.layers import LogisticRegression, SigmoidalLayer
import theano.sandbox.softsign
import pylearn.datasets.MNIST


try:
    # this tells theano to use the GPU if possible
    from theano.sandbox.cuda import use
    use()
except Exception, e:
    print('Warning: Attempt to use GPU resulted in error "%s"' % str(e))

class LeNetConvPool(object):
    """WRITEME 

    Math of what the layer does, and what symbolic variables are created by the class (w, b,
    output).

    """

    #TODO: implement biases & scales properly. There are supposed to be more parameters.
    #    - one bias & scale per filter
    #    - one bias & scale per downsample feature location (a 2d bias)
    #    - more?

    def __init__(self, rng, input, n_examples, n_imgs, img_shape, n_filters, filter_shape=(5,5),
            poolsize=(2,2)):
        """
        Allocate a LeNetConvPool layer with shared variable internal parameters.

        :param rng: a random number generator used to initialize weights
        
        :param input: symbolic images.  Shape: (n_examples, n_imgs, img_shape[0], img_shape[1])

        :param n_examples: input's shape[0] at runtime

        :param n_imgs: input's shape[1] at runtime

        :param img_shape: input's shape[2:4] at runtime

        :param n_filters: the number of filters to apply to the image.

        :param filter_shape: the size of the filters to apply
        :type filter_shape: pair (rows, cols)

        :param poolsize: the downsampling (pooling) factor
        :type poolsize: pair (rows, cols)
        """

        #TODO: make a simpler convolution constructor!!
        #    - make dx and dy optional
        #    - why do we have to pass shapes? (Can we make them optional at least?)
        conv_op = ConvOp((n_imgs,)+img_shape, filter_shape, n_filters, n_examples,
                dx=1, dy=1, output_mode='valid')

        # - why is poolsize an op parameter here?
        # - can we just have a maxpool function that creates this Op internally?
        ds_op = DownsampleFactorMax(poolsize, ignore_border=True)

        # the filter tensor that we will apply is a 4D tensor
        w_shp = (n_filters, n_imgs) + filter_shape

        # the bias we add is a 1D tensor
        b_shp = (n_filters,)

        self.w = shared(
                numpy.asarray(
                    rng.uniform(
                        low=-1.0 / numpy.sqrt(filter_shape[0] * filter_shape[1] * n_imgs), 
                        high=1.0 / numpy.sqrt(filter_shape[0] * filter_shape[1] * n_imgs),
                        size=w_shp), 
                    dtype=input.dtype))
        self.b = shared(
                numpy.asarray(
                    rng.uniform(low=-.0, high=0., size=(n_filters,)),
                    dtype=input.dtype))

        self.input = input
        conv_out = conv_op(input, self.w)
        self.output = tensor.tanh(ds_op(conv_out) + b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.w, self.b]

class SigmoidalLayer(object):
    def __init__(self, input, n_in, n_out):
        """
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param w: a symbolic weight matrix of shape (n_in, n_out)
        :param b: symbolic bias terms of shape (n_out,)
        :param squash: an squashing function
        """
        self.input = input
        self.w = shared(
                numpy.asarray(
                    rng.uniform(low=-2/numpy.sqrt(n_in), high=2/numpy.sqrt(n_in),
                    size=(n_in, n_out)), dtype=input.dtype))
        self.b = shared(numpy.asarray(numpy.zeros(n_out), dtype=input.dtype))
        self.output = tensor.tanh(tensor.dot(input, self.w) + self.b)
        self.params = [self.w, self.b]

class LogisticRegression(object):
    """WRITEME"""

    def __init__(self, input, n_in, n_out):
        self.w = shared(numpy.zeros((n_in, n_out), dtype=input.dtype))
        self.b = shared(numpy.zeros((n_out,), dtype=input.dtype))
        self.l1=abs(self.w).sum()
        self.l2_sqr = (self.w**2).sum()
        self.output=nnet.softmax(theano.dot(input, self.w)+self.b)
        self.argmax=theano.tensor.argmax(self.output, axis=1)
        self.params = [self.w, self.b]

    def nll(self, target):
        """Return the negative log-likelihood of the prediction of this model under a given
        target distribution.  Passing symbolic integers here means 1-hot.
        WRITEME
        """
        return nnet.categorical_crossentropy(self.output, target)

    def errors(self, target):
        """Return a vector of 0s and 1s, with 1s on every line that was mis-classified.
        """
        if target.ndim != self.argmax.ndim:
            raise TypeError('target should have the same shape as self.argmax', ('target', target.type,
                'argmax', self.argmax.type))
        if target.dtype.startswith('int'):
            return theano.tensor.neq(self.argmax, target)
        else:
            raise NotImplementedError()

def evaluate_lenet5(batch_size=30, n_iter=1000):
    rng = numpy.random.RandomState(23455)

    mnist = pylearn.datasets.MNIST.train_valid_test()

    ishape=(28,28) #this is the size of MNIST images

    # allocate symbolic variables for the data
    x = tensor.fmatrix()  # the data is presented as rasterized images
    y = tensor.lvector()  # the labels are presented as 1D vector of [long int] labels

    # construct the first convolutional pooling layer
    layer0 = LeNetConvPool.new(rng, input=x.reshape((batch_size,1,28,28)), n_examples=batch_size, 
            n_imgs=1, img_shape=ishape, 
            n_filters=6, filter_shape=(5,5), 
            poolsize=(2,2))

    # construct the second convolutional pooling layer
    layer1 = LeNetConvPool.new(rng, input=layer0.output, n_examples=batch_size, 
            n_imgs=6, img_shape=(12,12),
            n_filters=16, filter_shape=(5,5),
            poolsize=(2,2))

    # construct a fully-connected sigmoidal layer
    layer2 = SigmoidalLayer.new(rng, input=layer1.output.flatten(2), n_in=16*16, n_out=128) # 128 ?

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression.new(input=layer2.output, n_in=128, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.nll(y).mean()

    # create a function to compute the mistakes that are made by the model
    test_model = pfunc([x,y], layer3.errors(y))

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params+ layer2.params+ layer1.params + layer0.params
    learning_rate = numpy.asarray(0.01, dtype='float32')

    # train_model is a function that updates the model parameters by SGD
    train_model = pfunc([x, y], cost, 
            updates=[(p, p - learning_rate*gp) for p,gp in zip(params, tensor.grad(cost, params))])

    # IS IT MORE SIMPLE TO USE A MINIMIZER OR THE DIRECT CODE?

    best_valid_score = float('inf')
    for i in xrange(n_iter):
        for j in xrange(len(mnist.train.x)/batch_size):
            cost_ij = train_model(
                    mnist.train.x[j*batch_size:(j+1)*batch_size],
                    mnist.train.y[j*batch_size:(j+1)*batch_size])
            #if 0 == j % 100:
                #print('epoch %i:%i, training error %f' % (i, j*batch_size, cost_ij))
        valid_score = numpy.mean([test_model(
                    mnist.valid.x[j*batch_size:(j+1)*batch_size],
                    mnist.valid.y[j*batch_size:(j+1)*batch_size])
                for j in xrange(len(mnist.valid.x)/batch_size)])
        print('epoch %i, validation error %f' % (i, valid_score))
        if valid_score < best_valid_score:
            best_valid_score = valid_score
            test_score = numpy.mean([test_model(
                        mnist.test.x[j*batch_size:(j+1)*batch_size],
                        mnist.test.y[j*batch_size:(j+1)*batch_size])
                    for j in xrange(len(mnist.test.x)/batch_size)])
            print('epoch %i, test error of best model %f' % (i, test_score))

if __name__ == '__main__':
    evaluate_lenet5()

