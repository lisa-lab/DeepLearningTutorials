"""
This tutorial introduces logistic regression using Theano and conjugate
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

   - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2


"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import load_data


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture ( one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoint lies

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the target lies

        """

        # initialize theta = (W,b) with 0s; W gets the shape (n_in, n_out),
        # while b is a vector of n_out elements, making theta a vector of
        # n_in*n_out + n_out elements
        self.theta = theano.shared(value=numpy.zeros(n_in * n_out + n_out,
                                                   dtype=theano.config.floatX),
                                   name='theta',
                                   borrow=True)
        # W is represented by the fisr n_in*n_out elements of theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        # b is the rest (last n_out elements)
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """Return the negative log-likelihood of the prediction of this model
        under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|}\mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|}\sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
                  the correct label
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


def cg_optimization_mnist(n_epochs=50, mnist_pkl_gz='mnist.pkl.gz'):
    """Demonstrate conjugate gradient optimization of a log-linear model

    This is demonstrated on MNIST.

    :type n_epochs: int
    :param n_epochs: number of epochs to run the optimizer

    :type mnist_pkl_gz: string
    :param mnist_pkl_gz: the path of the mnist training file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    #############
    # LOAD DATA #
    #############
    datasets = load_data(mnist_pkl_gz)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    batch_size = 600    # size of the minibatch

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ishape = (28, 28)  # this is the size of MNIST images
    n_in = 28 * 28  # number of input units
    n_out = 10  # number of output units

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    minibatch_offset = T.lscalar()  # offset to the start of a [mini]batch
    x = T.matrix()   # the data is presented as rasterized images
    y = T.ivector()  # the labels are presented as 1D vector of
                     # [int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y).mean()

    # compile a theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function([minibatch_offset], classifier.errors(y),
            givens={
                x: test_set_x[minibatch_offset:minibatch_offset + batch_size],
                y: test_set_y[minibatch_offset:minibatch_offset + batch_size]},
            name="test")

    validate_model = theano.function([minibatch_offset], classifier.errors(y),
            givens={
                x: valid_set_x[minibatch_offset:
                               minibatch_offset + batch_size],
                y: valid_set_y[minibatch_offset:
                               minibatch_offset + batch_size]},
            name="validate")

    #  compile a thenao function that returns the cost of a minibatch
    batch_cost = theano.function([minibatch_offset], cost,
            givens={
                x: train_set_x[minibatch_offset:
                               minibatch_offset + batch_size],
                y: train_set_y[minibatch_offset:
                               minibatch_offset + batch_size]},
            name="batch_cost")

    # compile a theano function that returns the gradient of the minibatch
    # with respect to theta
    batch_grad = theano.function([minibatch_offset],
                                 T.grad(cost, classifier.theta),
                                 givens={
                                     x: train_set_x[minibatch_offset:
                                            minibatch_offset + batch_size],
                                     y: train_set_y[minibatch_offset:
                                            minibatch_offset + batch_size]},
            name="batch_grad")

    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        train_losses = [batch_cost(i * batch_size)
                        for i in xrange(n_train_batches)]
        return numpy.mean(train_losses)

    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        grad = batch_grad(0)
        for i in xrange(1, n_train_batches):
            grad += batch_grad(i * batch_size)
        return grad / n_train_batches

    validation_scores = [numpy.inf, 0]

    # creates the validation function
    def callback(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        #compute the validation loss
        validation_losses = [validate_model(i * batch_size)
                             for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        print('validation error %f %%' % (this_validation_loss * 100.,))

        # check if it is better then best validation score got until now
        if this_validation_loss < validation_scores[0]:
            # if so, replace the old one, and compute the score on the
            # testing dataset
            validation_scores[0] = this_validation_loss
            test_losses = [test_model(i * batch_size)
                           for i in xrange(n_test_batches)]
            validation_scores[1] = numpy.mean(test_losses)

    ###############
    # TRAIN MODEL #
    ###############

    # using scipy conjugate gradient optimizer
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = time.clock()
    best_w_b = scipy.optimize.fmin_cg(
               f=train_fn,
               x0=numpy.zeros((n_in + 1) * n_out, dtype=x.dtype),
               fprime=train_fn_grad,
               callback=callback,
               disp=0,
               maxiter=n_epochs)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%, with '
          'test performance %f %%') %
               (validation_scores[0] * 100., validation_scores[1] * 100.))

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    cg_optimization_mnist()
