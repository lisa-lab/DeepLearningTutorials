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

TODO: recommended preprocessing, lr ranges, regularization ranges (explain 
      to do lr first, then add regularization)

"""
__docformat__ = 'restructedtext en'


import numpy, cPickle, gzip

import time

import theano
import theano.tensor as T
import theano.tensor.nnet


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
        architecture ( one minibatch)

        :param n_in: number of input units, the dimension of the space in 
        which the datapoint lies

        :param n_out: number of output units, the dimension of the space in 
        which the target lies

        """ 

        # initialize theta = (W,b) with 0s; W gets the shape (n_in, n_out), 
        # while b is a vector of n_out elements, making theta a vector of
        # n_in*n_out + n_out elements
        self.theta = theano.shared( value = numpy.zeros(n_in*n_out+n_out) )
        # W is represented by the fisr n_in*n_out elements of theta
        self.W = self.theta[0:n_in*n_out].reshape((n_in,n_out))
        # b is the rest (last n_out elements)
        self.b = self.theta[n_in*n_out:n_in*n_out+n_out]


        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)





    def negative_log_likelihood(self, y):
        """Return the negative log-likelihood of the prediction of this model
        under a given target distribution.  

        TODO : add description of the categorical_crossentropy

        :param y: corresponds to a vector that gives for each example the
        :correct label
        """
        # TODO: inline NLL formula, refer to theano function
        return T.nnet.categorical_crossentropy(self.p_y_given_x, y)





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







def cg_optimization_mnist( n_iter=50 ):
    """Demonstrate conjugate gradient optimization of a log-linear model 

    This is demonstrated on MNIST.
    
    :param n_iter: number of iterations ot run the optimizer 

    """
    #TODO: Tzanetakis

    # Load the dataset ; note that the dataset is already divided in
    # minibatches of size 10; 
    f = gzip.open('mnist.pkl.gz','rb')
    train_batches, valid_batches, test_batches = cPickle.load(f)
    f.close()

    ishape     = (28,28) # this is the size of MNIST images
    batch_size = 5       # size of the minibatch 
    n_in       = 28*28   # number of input units
    n_out      = 10      # number of output units
    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of 
                          # [long int] labels

 
    # construct the logistic regression class
    classifier = LogisticRegression( \
                   input=x.reshape((batch_size,28*28)), n_in=28*28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of 
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y).mean() 

    # compile a theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function([x,y], classifier.errors(y))
    # compile a theano function that returns the gradient of the minibatch 
    # with respect to theta
    batch_grad = theano.function([x, y], T.grad(cost, classifier.theta))
    #  compile a thenao function that returns the cost of a minibatch
    batch_cost = theano.function([x, y], cost)

    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        classifier.theta.value = theta_value
        cost = 0.
        for x,y in train_batches :
            cost += batch_cost(x,y)
        return cost / len(train_batches)

    # creates a function that computes the average gradient of cost with 
    # respect to theta
    def train_fn_grad(theta_value):
        classifier.theta.value = theta_value
        grad = numpy.zeros(n_in * n_out + n_out)
        for x,y in train_batches:
            grad += batch_grad(x,y)
        return grad/ len(train_batches)



    validation_scores = [float('inf'), 0]
 
    # creates the validation function
    def callback(theta_value):
        classifier.theta.value = theta_value
        #compute the validation loss
        this_validation_loss = 0.
        for x,y in valid_batches:
            this_validation_loss += test_model(x,y)

        this_validation_loss /= len(valid_batches)

        print('validation error %f %%' % (this_validation_loss*100.,))
        
        # check if it is better then best validation score got until now
        if this_validation_loss < validation_scores[0]:
            # if so, replace the old one, and compute the score on the 
            # testing dataset
            validation_scores[0] = this_validation_loss
            test_score = 0.
            for x,y in test_batches:
                test_score += test_model(x,y)
            validation_scores[1] = test_score / len(test_batches)

    # using scipy conjugate gradient optimizer 
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = time.clock()
    best_w_b = scipy.optimize.fmin_cg(
            f=train_fn, 
            x0=numpy.zeros((n_in+1)*n_out, dtype=x.dtype),
            fprime=train_fn_grad,
            callback=callback,
            disp=0,
            maxiter=n_iter)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%, with'
          'test performance %f %%') % 
               (best_validation_loss*100., test_score*100.))

    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))







if __name__ == '__main__':
    cg_optimization_mnist()

