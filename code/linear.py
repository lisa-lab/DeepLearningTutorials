
"""
This tutorial introduces logistic regression using Theano.  

This tutorial presents a stochastic gradient descent optimization method suitable for large
datasets, and a conjugate gradient optimization method that is suitable for smaller datasets.


References:

    - textbooks: Bishop, other...

TODO: apply to each dataset

TODO: recommended preprocessing, lr ranges, regularization ranges (explain to do lr first, then
add regularization)


"""

import numpy
import pylearn.datasets.MNIST

from theano.compile.sandbox import shared, pfunc
from theano import tensor
import theano.tensor.nnet

def load_mnist_batches(batch_size):
    """
    WRITEME
    """
    mnist = pylearn.datasets.MNIST.train_valid_test()
    train_batches = [(mnist.train.x[i:i+batch_size], mnist.train.y[i:i+batch_size])
            for i in xrange(0, len(mnist.train.x), batch_size)]
    valid_batches = [(mnist.valid.x[i:i+batch_size], mnist.valid.y[i:i+batch_size])
            for i in xrange(0, len(mnist.valid.x), batch_size)]
    test_batches = [(mnist.test.x[i:i+batch_size], mnist.test.y[i:i+batch_size])
            for i in xrange(0, len(mnist.test.x), batch_size)]
    return train_batches, valid_batches, test_batches



class LogisticRegression(object):
    """
    
    w: the linear part of the affine transform
    b: the constant part of the affine transform
    TODO: add latex math formulas, reference to publication for complex models
    """

    def __init__(self, input, n_in, n_out):
        self.w = shared(numpy.zeros((n_in, n_out), dtype=input.dtype))
        self.b = shared(numpy.zeros((n_out,), dtype=input.dtype))
        self.l1=abs(self.w).sum()
        self.l2_sqr = (self.w**2).sum()
        self.output=tensor.nnet.softmax(theano.dot(input, self.w)+self.b)
        self.argmax=theano.tensor.argmax(self.output, axis=1)
        self.params = [self.w, self.b]

    def nll(self, target):
        """Return the negative log-likelihood of the prediction of this model under a given
        target distribution.  Passing symbolic integers here means 1-hot.

        WRITEME
        """
        # TODO: inline NLL formula, refer to theano function
        return tensor.nnet.categorical_crossentropy(self.output, target)

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

def sgd_optimization_mnist(batch_size=10, learning_rate=0.01, l1_reg=0.00, l2_reg=0.0,
        n_iter=100):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear model using model
    definition and symbolic gradient in theano.

    This is demonstrated on MNIST.

    """
    train_batches, valid_batches, test_batches = load_mnist_batches(batch_size)

    ishape=(28,28) #this is the size of MNIST images

    # allocate symbolic variables for the data
    x = tensor.fmatrix()  # the data is presented as rasterized images
    y = tensor.lvector()  # the labels are presented as 1D vector of [long int] labels

    # construct the first convolutional pooling layer
    classifier = LogisticRegression(input=x.reshape((batch_size,28*28)), n_in=28*28, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = classifier.nll(y).mean() + l1_reg * classifier.l1 + l2_reg * classifier.l2_sqr

    # create a function to compute the mistakes that are made by the model
    test_model = pfunc([x,y], classifier.errors(y))

    # train_model is a function that updates the model parameters by SGD
    train_model = pfunc([x, y], cost, 
            updates=[(p, p - numpy.asarray(learning_rate,dtype=x.dtype)*gp) 
                for (p, gp) in zip(classifier.params, tensor.grad(cost, classifier.params))])

    best_valid_score = float('inf')

    for i in xrange(n_iter):
        for x,y in train_batches:
            cost_ij = train_model(x, y)
        valid_score = numpy.mean([test_model(x, y) for (x,y) in valid_batches])

        print('epoch %i, validation error %f' % (i, valid_score))
        if valid_score < best_valid_score:
            best_valid_score = valid_score
            test_score = numpy.mean([test_model(x, y) for (x,y) in test_batches])
            print('epoch %i, test error of best model %f' % (i, test_score))

    print('Optimization complete with best validation score of %f, with test performance %f' %
            (best_valid_score, test_score))

def cg_optimization_mnist(batch_size=16, learning_rate=0.01, l1_reg=0.0001, l2_reg=0.001, n_iter=50):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear model using model
    definition and symbolic gradient in theano.

    This is demonstrated on MNIST.

    """
    #TODO: Tzanetakis

    ishape=(28,28) #this is the size of MNIST images

    n_in = 28*28
    n_out = 10

    # allocate symbolic variables for the data
    x = tensor.fmatrix()  # the data is presented as rasterized images
    y = tensor.lvector()  # the labels are presented as 1D vector of [long int] labels
    w_b = tensor.fvector() # rasterized storage for all parameters of affine model

    # unpack rasterized storage to the affine transform parameters (w, b)
    w = w_b[0:n_in*n_out].reshape((n_in, n_out))
    b = w_b[n_in*n_out:]
    posterior = tensor.nnet.softmax(theano.dot(x, w) + b)
    decision = tensor.argmax(posterior, axis=1) #an integer decision for each batch element

    nll = tensor.nnet.categorical_crossentropy(posterior, y).mean()

    cost = nll #+ l1_reg  *l1 + l2_reg * classifier.l2_sqr

    # create a function to compute the mistakes that are made by the model
    print ("Compiling theano functions...")
    test_model = pfunc([x,y, w_b], tensor.neq(y, decision))
    batch_grad = pfunc([x, y, w_b], tensor.grad(cost, w_b))
    batch_cost = pfunc([x, y, w_b], cost)

    train_batches, valid_batches, test_batches = load_mnist_batches(batch_size)

    def train_fn(w_b_value):
        return numpy.mean([batch_cost(x, y, w_b_value) for (x,y) in train_batches])

    def train_fn_grad(w_b_value):
        return numpy.mean([batch_grad(x, y, w_b_value) for (x,y) in train_batches], axis=0)

    validation_scores = [float('inf'), 0]
    def callback(w_b_value):
        valid_score = numpy.mean([test_model(x, y, w_b_value) for (x,y) in valid_batches])
        print('validation error %f' % (valid_score,))
        if valid_score < validation_scores[0]:
            validation_scores[0] = valid_score
            validation_scores[1] = numpy.mean([test_model(x, y, w_b_value) for (x,y) in test_batches])

    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    best_w_b = scipy.optimize.fmin_cg(
            f=train_fn, 
            x0=numpy.zeros((n_in+1)*n_out, dtype=x.dtype),
            fprime=train_fn_grad,
            callback=callback,
            disp=0,
            maxiter=n_iter)

    print('Optimization complete with best validation score of %f, with test performance %f' %
        tuple(validation_scores))

if __name__ == '__main__':
    sgd_optimization_mnist()
    #cg_optimization_mnist()

