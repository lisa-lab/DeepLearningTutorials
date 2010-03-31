import convolutional_mlp, logistic_cg, logistic_sgd, mlp, SdA, dA, rbm , DBN
from nose.plugins.skip import SkipTest
import theano
import time, sys


def test_logistic_sgd():
    t0=time.time()
    logistic_sgd.sgd_optimization_mnist(n_epochs=10)
    print >> sys.stderr, "test_logistic_sgd took %.3fs expected 15.2s in our buildbot"%(time.time()-t0)


def test_logistic_cg():
    t0=time.time()
    logistic_cg.cg_optimization_mnist(n_epochs=10)
    print >> sys.stderr, "test_logistic_cg took %.3fs expected 14s in our buildbot"%(time.time()-t0)


def test_mlp():
    t0=time.time()
    mlp.test_mlp(n_epochs=5)
    print >> sys.stderr, "test_mlp took %.3fs expected 118s in our buildbot"%(time.time()-t0)


def test_convolutional_mlp():
    t0=time.time()
    convolutional_mlp.evaluate_lenet5(n_epochs=5,nkerns=[5,5])
    print >> sys.stderr, "test_convolutional_mlp took %.3fs expected 168s in our buildbot"%(time.time()-t0)



def test_dA():
    t0=time.time()
    dA.test_dA(training_epochs = 3, output_folder = 'tmp_dA_plots')
    print >> sys.stderr, "test_dA took %.3fs expected Xs in our buildbot"%(time.time()-t0)


def test_SdA():
    t0=time.time()
    SdA.test_SdA(pretraining_epochs = 2, training_epochs = 3, batch_size = 300)
    print >> sys.stderr, "test_SdA took %.3fs expected 971s in our buildbot"%(time.time()-t0)


def test_dbn():
    t0=time.time()
    DBN.test_DBN(pretraining_epochs = 1, training_epochs = 2, batch_size =300)
    print >> sys.stderr, "test_mlp took %.3fs expected ??s in our buildbot"%(time.time()-t0)



def test_rbm():
    t0=time.time()
    rbm.test_rbm(training_epochs = 1, batch_size = 300, n_chains = 1, n_samples = 1, 
            output_folder =  'tmp_rbm_plots')
    print >> sys.stderr, "test_rbm took %.3fs expected ??s in our buildbot"%(time.time()-t0)


