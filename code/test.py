#import convolutional_mlp, dbn, logistic_cg, logistic_sgd, mlp, rbm, SdA_loops, SdA
import convolutional_mlp, logistic_cg, logistic_sgd, mlp, SdA, dA
from nose.plugins.skip import SkipTest
import time,sys
#TODO: dbn, rbm, SdA, SdA_loops, convolutional_mlp
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
def test_dbn():
    raise SkipTest('Implementation not finished')
def test_rbm():
    raise SkipTest('Implementation not finished')
def test_dA():
    t0=time.time()
    dA.test_dA(training_epochs = 3)
    print >> sys.stderr, "test_dA took %.3fs expected Xs in our buildbot"%(time.time()-t0)
def test_SdA():
    t0=time.time()
    SdA.test_SdA(pretraining_epochs = 2, training_epochs = 3)
    print >> sys.stderr, "test_SdA took %.3fs expected 971s in our buildbot"%(time.time()-t0)
