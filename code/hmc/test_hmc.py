
from __future__ import print_function

import numpy
import theano

try:
    from hmc import HMC_sampler
except ImportError as e:
    # python 3 compatibility
    # http://stackoverflow.com/questions/3073259/python-nose-import-error
    from hmc.hmc import HMC_sampler


def sampler_on_nd_gaussian(sampler_cls, burnin, n_samples, dim=10):
    batchsize = 3

    rng = numpy.random.RandomState(123)

    # Define a covariance and mu for a gaussian
    mu = numpy.array(rng.rand(dim) * 10, dtype=theano.config.floatX)
    cov = numpy.array(rng.rand(dim, dim), dtype=theano.config.floatX)
    cov = (cov + cov.T) / 2.
    cov[numpy.arange(dim), numpy.arange(dim)] = 1.0
    cov_inv = numpy.linalg.inv(cov)

    # Define energy function for a multi-variate Gaussian
    def gaussian_energy(x):
        return 0.5 * (theano.tensor.dot((x - mu), cov_inv) *
                      (x - mu)).sum(axis=1)

    # Declared shared random variable for positions
    position = rng.randn(batchsize, dim).astype(theano.config.floatX)
    position = theano.shared(position)

    # Create HMC sampler
    sampler = sampler_cls(position, gaussian_energy,
                          initial_stepsize=1e-3, stepsize_max=0.5)

    # Start with a burn-in process
    garbage = [sampler.draw() for r in range(burnin)]  # burn-in Draw
    # `n_samples`: result is a 3D tensor of dim [n_samples, batchsize,
    # dim]
    _samples = numpy.asarray([sampler.draw() for r in range(n_samples)])
    # Flatten to [n_samples * batchsize, dim]
    samples = _samples.T.reshape(dim, -1).T

    print('****** TARGET VALUES ******')
    print('target mean:', mu)
    print('target cov:\n', cov)

    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean: ', samples.mean(axis=0))
    print('empirical_cov:\n', numpy.cov(samples.T))

    print('****** HMC INTERNALS ******')
    print('final stepsize', sampler.stepsize.get_value())
    print('final acceptance_rate', sampler.avg_acceptance_rate.get_value())

    return sampler


def test_hmc():
    sampler = sampler_on_nd_gaussian(HMC_sampler.new_from_shared_positions,
                                     burnin=1000, n_samples=1000, dim=5)
    assert abs(sampler.avg_acceptance_rate.get_value() -
               sampler.target_acceptance_rate) < .1
    assert sampler.stepsize.get_value() >= sampler.stepsize_min
    assert sampler.stepsize.get_value() <= sampler.stepsize_max
