import convolutional_mlp, logistic_cg, logistic_sgd, mlp, SdA, dA
from nose.plugins.skip import SkipTest
import theano
#TODO: rbm, DBN, deep

def test_logistic_sgd():
    logistic_sgd.sgd_optimization_mnist(n_epochs=10)
def test_logistic_cg():
    logistic_cg.cg_optimization_mnist(n_epochs=10)
def test_mlp():
    mlp.test_mlp(n_epochs=5)
def test_convolutional_mlp():
    convolutional_mlp.evaluate_lenet5(n_epochs=5,nkerns=[5,5])
def test_dbn():
    raise SkipTest('Implementation not finished')
def test_rbm():
    raise SkipTest('Implementation not finished')
def test_dA():
    dA.test_dA(training_epochs = 2)
def test_SdA():
    SdA.test_SdA(pretraining_epochs = 1, training_epochs = 2)
