"""
 This tutorial introduces denoising auto-encoders using Theano. 

 Denoising autoencoders can be used as building blocks for deep networks. 
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
from theano import tensor
from theano.compile.sandbox import shared, pfunc
from theano.compile.sandbox.shared_randomstreams import RandomStreams
from theano.tensor import nnet
import pylearn.datasets.MNIST


try:
    #this tells theano to use the GPU if possible
    from theano.sandbox.cuda import use
    use()
except Exception,e:
    print ('Warning: Attempt to use GPU resulted in error "%s"'%str(e))


def load_mnist_batches(batch_size):
    """
    We should remove the dependency on pylearn.datasets.MNIST .. and maybe
    provide a pickled version of the dataset.. 
    """
    mnist = pylearn.datasets.MNIST.train_valid_test()
    train_batches = [(mnist.train.x[i:i+batch_size],mnist.train.y[i:i+batch_size])
            for i in xrange(0, len(mnist.train.x), batch_size)]
    valid_batches = [(mnist.valid.x[i:i+batch_size], mnist.valid.y[i:i+batch_size])
            for i in xrange(0, len(mnist.valid.x), batch_size)]
    test_batches = [(mnist.test.x[i:i+batch_size], mnist.test.y[i:i+batch_size])
            for i in xrange(0, len(mnist.test.x), batch_size)]
    return train_batches, valid_batches, test_batches




class DAE():
  """Denoising Auto-Encoder class 

  A denoising autoencoders tried to reconstruct the input from a corrupted 
  version of it by projecting it first in a latent space and reprojecting 
  it in the input space. Please refer to Vincent et al.,2008 for more 
  details. If x is the input then equation (1) computes a partially destroyed
  version of x by means of a stochastic mapping q_D. Equation (2) computes 
  the projection of the input into the latent space. Equation (3) computes 
  the reconstruction of the input, while equation (4) computes the 
  reconstruction error.
  
  .. latex-eqn:
    \tilde{x} ~ q_D(\tilde{x}|x)                                         (1)
    y = s(W \tilde{x} + b)                                               (2)
    x = s(W' y  + b')                                                    (3)
    L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]          (4)

  Tricks and thumbrules for DAE 
     - learning rate should be used in a logarithmic scale ...
  """

  def __init__(self, n_visible= 784, n_hidden= 500, lr= 1e-1, input= None):
    """
    Initialize the DAE class by specifying the number of visible units (the 
    dimension d of the input ), the number of hidden units ( the dimension 
    d' of the latent or hidden space ), a initial value for the learning rate
    and by giving a symbolic description of the input. Such a symbolic 
    description is of no importance for the simple DAE and therefore can be 
    ignored. This feature is useful when stacking DAEs, since the input of 
    intermediate layers can be symbolically described in terms of the hidden
    units of the previous layer. See the tutorial on SDAE for more details.
    
    :param n_visible: number of visible units
    :param n_hidden:  number of hidden units
    :param lr:        a initial value for the learning rate
    :param input:     a symbolic description of the input or None 
    """
    self.n_visible = n_visible
    self.n_hidden  = n_hidden
    
    # create a Theano random generator that gives symbolic random values
    theano_rng = RandomStreams( seed = 1234 )
    # create a numpy random generator
    numpy_rng = numpy.random.RandomState( seed = 52432 )
    
     
    # initial values for weights and biases
    # note : W' was written as W_prime and b' as b_prime
    initial_W       = numpy_rng.uniform(size = (n_visible, n_hidden))
    # transform W such that all values are between -.01 and .01
    initial_W       = (initial_W*2.0       - 1.0)*.01 
    initial_b       = numpy.zeros(n_hidden)
    initial_W_prime = numpy_rng.uniform(size = (n_hidden, n_visible))
    # transform W_prime such that all values are between -.01 and .01
    initial_W_prime = (initial_W_prime*2.0 - 1.0)*.01 
    initial_b_prime= numpy.zeros(n_visible)
     
    
    # theano shared variables for weights and biases
    self.W       = shared(value = initial_W      , name = "W")
    self.b       = shared(value = initial_b      , name = "b")
    self.W_prime = shared(value = initial_W_prime, name = "W'") 
    self.b_prime = shared(value = initial_b_prime, name = "b'")

    # theano shared variable for the learning rate 
    self.lr      = shared(value = lr             , name = "learning_rate")
      
    # if no input is given generate a variable representing the input
    if input == None : 
        # we use a matrix because we expect a minibatch of several examples,
        # each example being a row
        x = tensor.dmatrix(name = 'input') 
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
    #         used later when stacking DAEs. 
    self.y   = nnet.sigmoid(tensor.dot(tilde_x, self.W      ) + self.b)
    # Equation (3)
    z        = nnet.sigmoid(tensor.dot(self.y,  self.W_prime) + self.b_prime)
    # Equation (4)
    L = - tensor.sum( x*tensor.log(z) + (1-x)*tensor.log(1-z), axis=1 ) 
    # note : L is now a vector, where each element is the cross-entropy cost 
    #        of the reconstruction of the corresponding example of the 
    #        minibatch. We need to sum all these to get the cost of the
    #        minibatch
    cost = tensor.sum(L)
    # parameters with respect to whom we need to compute the gradient
    self.params = [ self.W, self.b, self.W_prime, self.b_prime]
    # use theano automatic differentiation to get the gradients
    gW, gb, gW_prime, gb_prime = tensor.grad(cost, self.params)
    # update the parameters in the direction of the gradient using the 
    # learning rate
    updated_W       = self.W       - gW       * self.lr
    updated_b       = self.b       - gb       * self.lr
    updated_W_prime = self.W_prime - gW_prime * self.lr
    updated_b_prime = self.b_prime - gb_prime * self.lr

    # defining the function that evaluate the symbolic description of 
    # one update step 
    self.update  = pfunc(params = [x], outputs = cost, updates = 
                                { self.W       : updated_W, 
                                  self.b       : updated_b,
                                  self.W_prime : updated_W_prime,
                                  self.b_prime : updated_b_prime } )
    self.get_cost = pfunc(params = [x], outputs = cost)









   

def train_DAE_mnist():
  """
  Trains a DAE on the MNIST dataset (http://yann.lecun.com/exdb/mnist)
  """

  # load dataset as batches  
  train_batches,valid_batches,test_batches=load_mnist_batches(batch_size=16)

  # Create a denoising auto-encoders with 28*28 = 784 input units, and 500
  # units in the hidden layer (latent layer); Learning rate is set to 1e-1
  dae = DAE( n_visible = 784,  n_hidden = 500, lr = 1e-2)

  # Number of iterations (epochs) to run
  n_iter = 30
  best_valid_score = float('inf')
  test_score       = float('inf')
  for i in xrange(n_iter):
    # train once over the dataset
    for x,y in train_batches:
        cost = dae.update(x)
     
    # compute validation error
    valid_cost = 0.
    for x,y in valid_batches:
        valid_cost = valid_cost + dae.get_cost(x)
    valid_cost = valid_cost / len(valid_batches)
    print('epoch %i, validation reconstruction error %f '%(i,valid_cost))

    if valid_cost < best_valid_score :
        best_valid_score = valid_cost
        # compute test error !?
        test_score = 0.
        for x,y in test_batches:
            test_score = test_score + dae.get_cost(x)
        test_score = test_score / len(test_batches)
        print('epoch %i, test error of best model %f' % (i, test_score))
    
  print('Optimization done. Best validation score %f, test performance %f' %
            (best_valid_score, test_score))



if __name__ == "__main__":
    train_DAE_mnist()

