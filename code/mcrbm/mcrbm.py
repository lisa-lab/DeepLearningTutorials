"""
This file implements the Mean & Covariance RBM discussed in

    Ranzato, M. and Hinton, G. E. (2010)
    Modeling pixel means and covariances using factored third-order Boltzmann machines.
    IEEE Conference on Computer Vision and Pattern Recognition.

and performs one of the experiments on CIFAR-10 discussed in that paper.  There are some minor
discrepancies between the paper and the accompanying code (train_mcRBM.py), and the
accompanying code has been taken to be correct in those cases because I couldn't get things to
work otherwise.


Math
====

Energy of "covariance RBM"

    E = -0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i C_{if} v_i )^2
      = -0.5 \sum_f (\sum_k P_{fk} h_k) ( \sum_i C_{if} v_i )^2
                    "vector element f"           "vector element f"

In some parts of the paper, the P matrix is chosen to be a diagonal matrix with non-positive
diagonal entries, so it is helpful to see this as a simpler equation:

    E =  \sum_f h_f ( \sum_i C_{if} v_i )^2



Version in paper
----------------

Full Energy of the Mean and Covariance RBM, with
:math:`h_k = h_k^{(c)}`,
:math:`g_j = h_j^{(m)}`,
:math:`b_k = b_k^{(c)}`,
:math:`c_j = b_j^{(m)}`,
:math:`U_{if} = C_{if}`,

    E (v, h, g) =
        - 0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i (U_{if} v_i) / |U_{.f}|*|v| )^2
        - \sum_k b_k h_k
        + 0.5 \sum_i v_i^2
        - \sum_j \sum_i W_{ij} g_j v_i
        - \sum_j c_j g_j

For the energy function to correspond to a probability distribution, P must be non-positive.  P
is initialized to be a diagonal or a topological pooling matrix, and in our experience it can
be left as such because even in the paper it has a very low learning rate, and is only allowed
to be updated after the filters in U are learned (in effect).

Version in published train_mcRBM code
-------------------------------------

The train_mcRBM file implements learning in a similar but technically different Energy function:

    E (v, h, g) =
         0.5 \sum_f \sum_k P_{fk} h_k (\sum_i U_{if} v_i / sqrt(\sum_i v_i^2/I + 0.5))^2
        - \sum_k b_k h_k
        + 0.5 \sum_i v_i^2
        - \sum_j \sum_i W_{ij} g_j v_i
        - \sum_j c_j g_j

There are two differences with respect to the paper:

    - 'v' is not normalized by its length, but rather it is normalized to have length close to
      the square root of the number of its components.  The variable called 'small' that
      "avoids division by zero" is orders larger than machine precision, and is on the order of
      the normalized sum-of-squares, so I've included it in the Energy function.

    - 'U' is also not normalized by its length.  U is initialized to have columns that are
      shorter than unit-length (approximately 0.2 with the 105 principle components in the
      train_mcRBM data).  During training, the columns of U are constrained manually to have
      equal lengths (see the use of normVF), but Euclidean norm is allowed to change.  During
      learning it quickly converges towards 1 and then exceeds 1.  It does not seem like this
      column-wise normalization of U is justified by maximum-likelihood, I have no intuition
      for why it is used.


Version in this code
--------------------

This file implements the same algorithm as the train_mcRBM code, except that the P matrix is
omitted for clarity, and replaced analytically with a negative identity matrix.

    E (v, h, g) =
        + 0.5 \sum_k h_k (\sum_i U_{ik} v_i / sqrt(\sum_i v_i^2/I + 0.5))^2
        - \sum_k b_k h_k
        + 0.5 \sum_i v_i^2
        - \sum_j \sum_i W_{ij} g_j v_i
        - \sum_j c_j g_j

    E (v, h, g) =
        - 0.5 \sum_f \sum_k P_{fk} h_k (\sum_i U_{if} v_i / sqrt(\sum_i v_i^2/I + 0.5))^2
        - \sum_k b_k h_k
        + 0.5 \sum_i v_i^2
        - \sum_j \sum_i W_{ij} g_j v_i
        - \sum_j c_j g_j



Conventions in this file
========================

This file contains some global functions, as well as a class (MeanCovRBM) that makes using them a little
more convenient.


Global functions like `free_energy` work on an mcRBM as parametrized in a particular way.
Suppose we have
 - I input dimensions,
 - F squared filters,
 - J mean variables, and
 - K covariance variables.

The mcRBM is parametrized by 6 variables:

 - `P`, a matrix whose rows indicate covariance filter groups (F x K)
 - `U`, a matrix whose rows are visible covariance directions (I x F)
 - `W`, a matrix whose rows are visible mean directions (I x J)
 - `b`, a vector of hidden covariance biases (K)
 - `c`, a vector of hidden mean biases  (J)

Matrices are generally layed out and accessed according to a C-order convention.

"""

#
# WORKING NOTES
# THIS DERIVATION IS BASED ON THE ** PAPER ** ENERGY FUNCTION
# NOT THE ENERGY FUNCTION IN THE CODE!!!
#
# Free energy is the marginal energy of visible units
# Recall:
#   Q(x) = exp(-E(x))/Z ==> -log(Q(x)) - log(Z) = E(x)
#
#
#   E (v, h, g) =
#       - 0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i U_{if} v_i )^2 / |U_{*f}|^2 |v|^2
#       - \sum_k b_k h_k
#       + 0.5 \sum_i v_i^2
#       - \sum_j \sum_i W_{ij} g_j v_i
#       - \sum_j c_j g_j
#       - \sum_i a_i v_i
#
#
# Derivation, in which partition functions are ignored.
#
# E(v) = -\log(Q(v))
#  = -\log( \sum_{h,g} Q(v,h,g))
#  = -\log( \sum_{h,g} exp(-E(v,h,g)))
#  = -\log( \sum_{h,g} exp(-
#       - 0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i U_{if} v_i )^2 / (|U_{*f}| * |v|)
#       - \sum_k b_k h_k
#       + 0.5 \sum_i v_i^2
#       - \sum_j \sum_i W_{ij} g_j v_i
#       - \sum_j c_j g_j
#       - \sum_i a_i v_i ))
#
# Get rid of double negs  in exp
#  = -\log(  \sum_{h} exp(
#       + 0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i U_{if} v_i )^2 / (|U_{*f}| * |v|)
#       + \sum_k b_k h_k
#       - 0.5 \sum_i v_i^2
#       ) * \sum_{g} exp(
#       + \sum_j \sum_i W_{ij} g_j v_i
#       + \sum_j c_j g_j))
#    - \sum_i a_i v_i
#
# Break up log
#  = -\log(  \sum_{h} exp(
#       + 0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i U_{if} v_i )^2 / (|U_{*f}|*|v|)
#       + \sum_k b_k h_k
#       ))
#    -\log( \sum_{g} exp(
#       + \sum_j \sum_i W_{ij} g_j v_i
#       + \sum_j c_j g_j )))
#    + 0.5 \sum_i v_i^2
#    - \sum_i a_i v_i
#
# Use domain h is binary to turn log(sum(exp(sum...))) into sum(log(..
#  = -\log(\sum_{h} exp(
#       + 0.5 \sum_f \sum_k P_{fk} h_k ( \sum_i U_{if} v_i )^2 / (|U_{*f}|* |v|)
#       + \sum_k b_k h_k
#       ))
#    - \sum_{j} \log(1 + exp(\sum_i W_{ij} v_i + c_j ))
#    + 0.5 \sum_i v_i^2
#    - \sum_i a_i v_i
#
#  = - \sum_{k} \log(1 + exp(b_k + 0.5 \sum_f P_{fk}( \sum_i U_{if} v_i )^2 / (|U_{*f}|*|v|)))
#    - \sum_{j} \log(1 + exp(\sum_i W_{ij} v_i + c_j ))
#    + 0.5 \sum_i v_i^2
#    - \sum_i a_i v_i
#
# For negative-one-diagonal P this gives:
#
#  = - \sum_{k} \log(1 + exp(b_k - 0.5 \sum_i (U_{ik} v_i )^2 / (|U_{*k}|*|v|)))
#    - \sum_{j} \log(1 + exp(\sum_i W_{ij} v_i + c_j ))
#    + 0.5 \sum_i v_i^2
#    - \sum_i a_i v_i

import sys, os, logging
import numpy as np
import numpy

import theano
from theano import function, shared, dot
from theano import tensor as TT
floatX = theano.config.floatX

sharedX = lambda X, name : shared(numpy.asarray(X, dtype=floatX), name=name)

import pylearn
from pylearn.sampling.hmc import HMC_sampler
from pylearn.io import image_tiling
from pylearn.gd.sgd import sgd_updates
import pylearn.dataset_ops.image_patches

###########################################
#
# Candidates for factoring
#
###########################################

def l1(X):
    """
    :param X: TensorType variable

    :rtype: TensorType scalar

    :returns: the sum of absolute values of the terms in X

    :math: \sum_i |X_i|

    Where i is an appropriately dimensioned index.

    """
    return abs(X).sum()

def l2(X):
    """
    :param X: TensorType variable

    :rtype: TensorType scalar

    :returns: the sum of absolute values of the terms in X

    :math: \sqrt{ \sum_i X_i^2 }

    Where i is an appropriately dimensioned index.

    """
    return TT.sqrt((X**2).sum())

def contrastive_cost(free_energy_fn, pos_v, neg_v):
    """
    :param free_energy_fn: lambda (TensorType matrix MxN) ->  TensorType vector of M free energies
    :param pos_v: TensorType matrix MxN of M "positive phase" particles
    :param neg_v: TensorType matrix MxN of M "negative phase" particles

    :returns: TensorType scalar that's the sum of the difference of free energies

    :math: \sum_i free_energy(pos_v[i]) - free_energy(neg_v[i])

    """
    return (free_energy_fn(pos_v) - free_energy_fn(neg_v)).sum()

def contrastive_grad(free_energy_fn, pos_v, neg_v, wrt, other_cost=0):
    """
    :param free_energy_fn: lambda (TensorType matrix MxN) ->  TensorType vector of M free energies
    :param pos_v: positive-phase sample of visible units
    :param neg_v: negative-phase sample of visible units
    :param wrt: TensorType variables with respect to which we want gradients (similar to the
        'wrt' argument to tensor.grad)
    :param other_cost: TensorType scalar

    :returns: TensorType variables for the gradient on each of the 'wrt' arguments


    :math: Cost = other_cost + \sum_i free_energy(pos_v[i]) - free_energy(neg_v[i])
    :math: d Cost / dW for W in `wrt`


    This function is similar to tensor.grad - it returns the gradient[s] on a cost with respect
    to one or more parameters.  The difference between tensor.grad and this function is that
    the negative phase term (`neg_v`) is considered constant, i.e. d `Cost` / d `neg_v` = 0.
    This is desirable because `neg_v` might be the result of a sampling expression involving
    some of the parameters, but the contrastive divergence algorithm does not call for
    backpropagating through the sampling procedure.

    Warning - if other_cost depends on pos_v or neg_v and you *do* want to backpropagate from
    the `other_cost` through those terms, then this function is inappropriate.  In that case,
    you should call tensor.grad separately for the other_cost and add the gradient expressions
    you get from ``contrastive_grad(..., other_cost=0)``

    """
    cost=contrastive_cost(free_energy_fn, pos_v, neg_v)
    if other_cost:
        cost = cost + other_cost
    return theano.tensor.grad(cost,
            wrt=wrt,
            consider_constant=[neg_v])

###########################################
#
# Expressions that are mcRBM-specific
#
###########################################

class mcRBM(object):
    """Light-weight class that provides the math related to inference

    Attributes:

      - U - the covariance filters (theano shared variable)
      - W - the mean filters (theano shared variable)
      - a - the visible bias (theano shared variable)
      - b - the covariance bias (theano shared variable)
      - c - the mean bias (theano shared variable)

    """
    def __init__(self, U, W, a, b, c):
        self.U = U
        self.W = W
        self.a = a
        self.b = b
        self.c = c

    def hidden_cov_units_preactivation_given_v(self, v, small=0.5):
        """Return argument to the sigmoid that would give mean of covariance hid units
        return b - 0.5 * dot(v/||v||, U)**2
        """
        unit_v = v / (TT.sqrt(TT.mean(v**2, axis=1)+small)).dimshuffle(0,'x') # adjust row norm
        return self.b - 0.5 * dot(unit_v, self.U)**2

    def free_energy_terms_given_v(self, v):
        """Returns theano expression for the terms that are added to form the free energy of
        visible vector `v` in an mcRBM.

         1.  Free energy related to covariance hiddens
         2.  Free energy related to mean hiddens
         3.  Free energy related to L2-Norm of `v`
         4.  Free energy related to projection of `v` onto biases `a`
        """
        t0 = -TT.sum(TT.nnet.softplus(self.hidden_cov_units_preactivation_given_v(v)),axis=1)
        t1 = -TT.sum(TT.nnet.softplus(self.c + dot(v,self.W)), axis=1)
        t2 =  0.5 * TT.sum(v**2, axis=1)
        t3 = -TT.dot(v, self.a)
        return [t0, t1, t2, t3]

    def free_energy_given_v(self, v):
        """Returns theano expression for free energy of visible vector `v` in an mcRBM
        """
        return TT.add(*self.free_energy_terms_given_v(v))

    def expected_h_g_given_v(self, v):
        """Returns tuple (`h`, `g`) of theano expression conditional expectations in an mcRBM.

        `h` is the conditional on the covariance units.
        `g` is the conditional on the mean units.

        """
        h = TT.nnet.sigmoid(self.hidden_cov_units_preactivation_given_v(v))
        g = TT.nnet.sigmoid(self.c + dot(v,self.W))
        return (h, g)

    def n_visible_units(self):
        """Return the number of visible units of this RBM

        For an RBM made from shared variables, this will return an integer,
        for a purely symbolic RBM this will return a theano expression.

        """
        try:
            return self.W.get_value(borrow=True).shape[0]
        except AttributeError:
            return self.W.shape[0]

    def n_hidden_cov_units(self):
        """Return the number of hidden units for the covariance in this RBM

        For an RBM made from shared variables, this will return an integer,
        for a purely symbolic RBM this will return a theano expression.

        """
        try:
            return self.U.get_value(borrow=True).shape[1]
        except AttributeError:
            return self.U.shape[1]

    def n_hidden_mean_units(self):
        """Return the number of hidden units for the mean in this RBM

        For an RBM made from shared variables, this will return an integer,
        for a purely symbolic RBM this will return a theano expression.

        """
        try:
            return self.W.get_value(borrow=True).shape[1]
        except AttributeError:
            return self.W.shape[1]

    def CD1_sampler(self, v, n_particles, n_visible=None, rng=8923984):
        """Return a symbolic negative-phase particle obtained by simulating the Hamiltonian
        associated with the energy function.
        """
        #TODO: why not expose all HMC arguments somehow?
        if not hasattr(rng, 'randn'):
            rng = np.random.RandomState(rng)
        if n_visible is None:
            n_visible = self.n_visible_units()

        # create a dummy hmc object because we want to use *some* of it
        hmc = HMC_sampler.new_from_shared_positions(
                shared_positions=v, # v is not shared, so some functionality will not work
                energy_fn=self.free_energy_given_v,
                seed=int(rng.randint(2**30)),
                shared_positions_shape=(n_particles,n_visible),
                compile_simulate=False)
        updates = dict(hmc.updates())
        final_p = updates.pop(v)
        return hmc, final_p, updates

    def sampler(self, n_particles, n_visible=None, rng=7823748):
        """Return an `HMC_sampler` that will draw samples from the distribution over visible
        units specified by this RBM.

        :param n_particles: this many parallel chains will be simulated.
        :param rng: seed or numpy RandomState object to initialize particles, and to drive the simulation.
        """
        #TODO: why not expose all HMC arguments somehow?
        #TODO: Consider returning a sample kwargs for passing to HMC_sampler?
        if not hasattr(rng, 'randn'):
            rng = np.random.RandomState(rng)
        if n_visible is None:
            n_visible = self.n_visible_units()
        rval = HMC_sampler.new_from_shared_positions(
            shared_positions = sharedX(
                rng.randn(
                    n_particles,
                    n_visible),
                name='particles'),
            energy_fn=self.free_energy_given_v,
            seed=int(rng.randint(2**30)))
        return rval

    def params(self):
        """Return the elements of [U,W,a,b,c] that are shared variables

        WRITEME : a *prescriptive* definition of this method suitable for mention in the API
        doc.

        """
        return list(self._params)

    @classmethod
    def alloc(cls, n_I, n_K, n_J, rng = 8923402190,
            U_range=0.02,
            W_range=0.05,
            a_ival=0,
            b_ival=2,
            c_ival=-2):
        """
        Return a MeanCovRBM instance with randomly-initialized shared variable parameters.

        :param n_I: input dimensionality
        :param n_K: number of covariance hidden units
        :param n_J: number of mean filters (linear)
        :param rng: seed or numpy RandomState object to initialize parameters

        :note:
        Constants for initial ranges and values taken from train_mcRBM.py.
        """
        if not hasattr(rng, 'randn'):
            rng = np.random.RandomState(rng)

        rval =  cls(
                U = sharedX(U_range * rng.randn(n_I, n_K),'U'),
                W = sharedX(W_range * rng.randn(n_I, n_J),'W'),
                a = sharedX(np.ones(n_I)*a_ival,'a'),
                b = sharedX(np.ones(n_K)*b_ival,'b'),
                c = sharedX(np.ones(n_J)*c_ival,'c'),)
        rval._params = [rval.U, rval.W, rval.a, rval.b, rval.c]
        return rval

def topological_connectivity(out_shape=(12,12), window_shape=(3,3), window_stride=(2,2),
        **kwargs):

    in_shape = (window_stride[0] * out_shape[0],
            window_stride[1] * out_shape[1])

    rval = numpy.zeros(in_shape + out_shape, dtype=theano.config.floatX)
    A,B,C,D = rval.shape

    # for each output position (out_r, out_c)
    for out_r in range(out_shape[0]):
        for out_c in range(out_shape[1]):
            # for each window position (win_r, win_c)
            for win_r in range(window_shape[0]):
                for win_c in range(window_shape[1]):
                    # add 1 to the corresponding input location
                    in_r = out_r * window_stride[0] + win_r
                    in_c = out_c * window_stride[1] + win_c
                    rval[in_r%A, in_c%B, out_r%C, out_c%D] += 1

    # This normalization algorithm is a guess, based on inspection of the matrix loaded from
    # see CVPR2010paper_material/topo2D_3x3_stride2_576filt.mat
    rval = rval.reshape((A*B, C*D))
    rval = (rval.T / rval.sum(axis=1)).T

    rval /= rval.sum(axis=0)
    return rval

class mcRBM_withP(mcRBM):
    """Light-weight class that provides the math related to inference

    Attributes:

      - U - the covariance filters (theano shared variable)
      - W - the mean filters (theano shared variable)
      - a - the visible bias (theano shared variable)
      - b - the covariance bias (theano shared variable)
      - c - the mean bias (theano shared variable)

    """
    def __init__(self, U, W, a, b, c, P):
        self.P = P
        super(mcRBM_withP, self).__init__(U,W,a,b,c)

    def hidden_cov_units_preactivation_given_v(self, v, small=0.5):
        """Return argument to the sigmoid that would give mean of covariance hid units

        See the math at the top of this file for what 'adjusted' means.

        return b - 0.5 * dot(adjusted(v), U)**2
        """
        unit_v = v / (TT.sqrt(TT.mean(v**2, axis=1)+small)).dimshuffle(0,'x') # adjust row norm
        return self.b + 0.5 * dot(dot(unit_v, self.U)**2, self.P)

    def n_hidden_cov_units(self):
        """Return the number of hidden units for the covariance in this RBM

        For an RBM made from shared variables, this will return an integer,
        for a purely symbolic RBM this will return a theano expression.

        """
        try:
            return self.P.get_value(borrow=True).shape[1]
        except AttributeError:
            return self.P.shape[1]

    @classmethod
    def alloc(cls, n_I, n_K, n_J, *args, **kwargs):
        """
        Return a MeanCovRBM instance with randomly-initialized shared variable parameters.

        :param n_I: input dimensionality
        :param n_K: number of covariance hidden units
        :param n_J: number of mean filters (linear)
        :param rng: seed or numpy RandomState object to initialize parameters

        :note:
        Constants for initial ranges and values taken from train_mcRBM.py.
        """
        return cls.alloc_with_P(
            -numpy.eye((n_K, n_K)).astype(theano.config.floatX),
            n_I,
            n_J,
            *args, **kwargs)

    @classmethod
    def alloc_topo_P(cls, n_I, n_J, p_out_shape=(12,12), p_win_shape=(3,3), p_win_stride=(2,2),
            **kwargs):
        return cls.alloc_with_P(
                -topological_connectivity(p_out_shape, p_win_shape, p_win_stride),
                n_I=n_I, n_J=n_J, **kwargs)

    @classmethod
    def alloc_with_P(cls, Pval, n_I, n_J, rng = 8923402190,
            U_range=0.02,
            W_range=0.05,
            a_ival=0,
            b_ival=2,
            c_ival=-2):
        n_F, n_K = Pval.shape
        if not hasattr(rng, 'randn'):
            rng = np.random.RandomState(rng)
        rval =  cls(
                U = sharedX(U_range * rng.randn(n_I, n_F),'U'),
                W = sharedX(W_range * rng.randn(n_I, n_J),'W'),
                a = sharedX(np.ones(n_I)*a_ival,'a'),
                b = sharedX(np.ones(n_K)*b_ival,'b'),
                c = sharedX(np.ones(n_J)*c_ival,'c'),
                P = sharedX(Pval, 'P'),)
        rval._params = [rval.U, rval.W, rval.a, rval.b, rval.c, rval.P]
        return rval

class mcRBMTrainer(object):
    """Light-weight class encapsulating math for mcRBM training

    Attributes:
      - rbm  - an mcRBM instance
      - sampler - an HMC_sampler instance
      - normVF - geometrically updated norm of U matrix columns (shared var)
      - learn_rate - SGD learning rate [un-annealed]
      - learn_rate_multipliers - the learning rates for each of the parameters of the rbm (in
        order corresponding to what's returned by ``rbm.params()``)
      - l1_penalty - float or TensorType scalar to modulate l1 penalty of rbm.U and rbm.W
      - iter - number of cd_updates (shared var) - used to anneal the effective learn_rate
      - lr_anneal_start - scalar or TensorType scalar - iter at which time to start decreasing
            the learning rate proportional to 1/iter

    """
    # TODO: accept a GD algo as an argument?
    @classmethod
    def alloc_for_P(cls, rbm, visible_batch, batchsize, initial_lr_per_example=0.075, rng=234,
            l1_penalty=0,
            l1_penalty_start=0,
            learn_rate_multipliers=None,
            lr_anneal_start=2000,
            p_training_start=4000,
            p_training_lr=0.02,
            persistent_chains=True
            ):
        if learn_rate_multipliers is None:
            p_lr = sharedX(0.0, 'P_lr_multiplier')
            learn_rate_multipliers = [2, .2, .02, .1, .02, p_lr]
        else:
            p_lr = None
        rval = cls.alloc(rbm, visible_batch, batchsize, initial_lr_per_example, rng, l1_penalty,
                l1_penalty_start, learn_rate_multipliers, lr_anneal_start, persistent_chains)

        rval.p_mask = sharedX((rbm.P.get_value(borrow=True) != 0).astype('float32'), 'p_mask')

        rval.p_lr = p_lr
        rval.p_training_start=p_training_start
        rval.p_training_lr=p_training_lr
        return rval


    @classmethod
    def alloc(cls, rbm, visible_batch, batchsize, initial_lr_per_example=0.075, rng=234,
            l1_penalty=0,
            l1_penalty_start=0,
            learn_rate_multipliers=[2, .2, .02, .1, .02],
            lr_anneal_start=2000,
            persistent_chains=True
            ):

        """
        :param rbm: mcRBM instance to train
        :param visible_batch: TensorType variable for training data
        :param batchsize: the number of rows in visible_batch
        :param initial_lr_per_example: the learning rate (may be annealed)
        :param rng: seed or RandomState to initialze PCD sampler
        :param l1_penalty: see class doc
        :param learn_rate_multipliers: see class doc
        :param lr_anneal_start: see class doc
        """
        #TODO: :param lr_anneal_iter: the iteration at which 1/t annealing will begin

        #TODO: get batchsize from visible_batch??
        # allocates shared var for negative phase particles


        # TODO: should normVF be initialized to match the size of rbm.U ?

        if (l1_penalty_start > 0) and (l1_penalty != 0.0):
            effective_l1_penalty = sharedX(0.0, 'effective_l1_penalty')
        else:
            effective_l1_penalty = l1_penalty

        if persistent_chains:
            sampler = rbm.sampler(batchsize, rng=rng)
        else:
            sampler = None

        return cls(
                rbm=rbm,
                batchsize=batchsize,
                visible_batch=visible_batch,
                sampler=sampler,
                normVF=sharedX(1.0, 'normVF'),
                learn_rate=sharedX(initial_lr_per_example/batchsize, 'learn_rate'),
                iter=sharedX(0, 'iter'),
                effective_l1_penalty=effective_l1_penalty,
                l1_penalty=l1_penalty,
                l1_penalty_start=l1_penalty_start,
                learn_rate_multipliers=learn_rate_multipliers,
                lr_anneal_start=lr_anneal_start,
                persistent_chains=persistent_chains,)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def normalize_U(self, new_U):
        """
        :param new_U: a proposed new value for rbm.U

        :returns: a pair of TensorType variables:
            a corrected new value for U, and a new value for self.normVF

        This is a weird normalization procedure, but the sample code for the paper has it, and
        it seems to be important.
        """
        U_norms = TT.sqrt((new_U**2).sum(axis=0))
        new_normVF = .95 * self.normVF + .05 * TT.mean(U_norms)
        return (new_U * new_normVF / U_norms), new_normVF

    def contrastive_grads(self, neg_v = None):
        """Return the contrastive divergence gradients on the parameters of self.rbm """
        if neg_v is None:
            neg_v = self.sampler.positions
        return contrastive_grad(
                free_energy_fn=self.rbm.free_energy_given_v,
                pos_v=self.visible_batch,
                neg_v=neg_v,
                wrt = self.rbm.params(),
                other_cost=(l1(self.rbm.U)+l1(self.rbm.W)) * self.effective_l1_penalty)

    def cd_updates(self):
        """
        Return a dictionary of shared variable updates that implements contrastive divergence
        learning by stochastic gradient descent with an annealed learning rate.
        """

        ups = {}

        if self.persistent_chains:
            grads = self.contrastive_grads()
            ups.update(dict(self.sampler.updates()))
        else:
            cd1_sampler, final_p, cd1_updates = self.rbm.CD1_sampler(self.visible_batch,
                    self.batchsize)
            self._last_cd1_sampler = cd1_sampler # hacked in here for the unit test
            #ignore the cd1_sampler
            grads = self.contrastive_grads(neg_v = final_p)
            ups.update(dict(cd1_updates))


        # contrastive divergence updates
        # TODO: sgd_updates is a particular optization algo (others are possible)
        #       parametrize so that algo is plugin
        #       the normalization normVF might be sgd-specific though...

        # TODO: when sgd has an annealing schedule, this should
        #       go through that mechanism.

        lr = TT.clip(
                self.learn_rate * TT.cast(self.lr_anneal_start / (self.iter+1), floatX),
                0.0, #min
                self.learn_rate) #max

        ups.update(dict(sgd_updates(
                    self.rbm.params(),
                    grads,
                    stepsizes=[a*lr for a in self.learn_rate_multipliers])))

        ups[self.iter] = self.iter + 1


        # add trainer updates (replace CD update of U)
        ups[self.rbm.U], ups[self.normVF] = self.normalize_U(ups[self.rbm.U])

        #l1_updates:
        if (self.l1_penalty_start > 0) and (self.l1_penalty != 0.0):
            ups[self.effective_l1_penalty] = TT.switch(
                    self.iter >= self.l1_penalty_start,
                    self.l1_penalty,
                    0.0)

        if getattr(self,'p_lr', None):
            ups[self.p_lr] = TT.switch(self.iter > self.p_training_start,
                    self.p_training_lr,
                    0)
            new_P = ups[self.rbm.P] * self.p_mask
            no_pos_P = TT.switch(new_P<0, new_P, 0)
            ups[self.rbm.P] = - no_pos_P / no_pos_P.sum(axis=0) #normalize to that columns sum 1

        return ups

