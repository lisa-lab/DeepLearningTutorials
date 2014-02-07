import sys

import numpy
import theano

import convolutional_mlp
import dA
import DBN
import logistic_cg
import logistic_sgd
import mlp
import rbm
import rnnrbm
import SdA


def test_logistic_sgd():
    logistic_sgd.sgd_optimization_mnist(n_epochs=10)


def test_logistic_cg():
    try:
        import scipy
        logistic_cg.cg_optimization_mnist(n_epochs=10)
    except ImportError:
        from nose.plugins.skip import SkipTest
        raise SkipTest(
            'SciPy not available. Needed for the logistic_cg example.')


def test_mlp():
    mlp.test_mlp(n_epochs=1)


def test_convolutional_mlp():
    convolutional_mlp.evaluate_lenet5(n_epochs=1, nkerns=[5, 5])


def test_dA():
    dA.test_dA(training_epochs=1, output_folder='tmp_dA_plots')


def test_SdA():
    SdA.test_SdA(pretraining_epochs=1, training_epochs=1, batch_size=300)


def test_dbn():
    DBN.test_DBN(pretraining_epochs=1, training_epochs=1, batch_size=300)


def test_rbm():
    rbm.test_rbm(training_epochs=1, batch_size=300, n_chains=1, n_samples=1,
                 n_hidden=20, output_folder='tmp_rbm_plots')


def test_rnnrbm():
    rnnrbm.test_rnnrbm(num_epochs=1)


def speed():
    """
    This fonction modify the configuration theano and don't restore it!
    """

    algo = ['logistic_sgd', 'logistic_cg', 'mlp', 'convolutional_mlp',
            'dA', 'SdA', 'DBN', 'rbm', 'rnnrbm']
    to_exec = [True] * len(algo)
#    to_exec = [False] * len(algo)
#    to_exec[-1] = True
    do_float64 = True
    do_float32 = True
    do_gpu = True

    algo_executed = [s for idx, s in enumerate(algo) if to_exec[idx]]
    #Timming expected are from the buildbot that have an i7-920 @
    # 2.67GHz with hyperthread enabled for the cpu, 12G of ram. An GeForce GTX
    # 285 for the GPU. OS=Fedora 14, gcc=4.5.1, python/BLAS from EPD
    # 7.1-2 (python 2.7.2, mkl unknow). BLAS with only 1 thread.

    expected_times_64 = numpy.asarray([10.0, 22.5, 76.1, 73.7, 116.4,
                                       346.9, 381.9, 558.1, 186.3])
    expected_times_32 = numpy.asarray([11.6, 29.6, 42.5, 66.5, 71,
                                       191.2, 226.8, 432.8, 176.2])

    # Number with just 1 decimal are new value that are faster with
    # the Theano version 0.5rc2 Other number are older. They are not
    # updated, as we where faster in the past!
    # TODO: find why and fix this!

# Here is the value for the buildbot on February 3th 2012.
#              sgd,         cg           mlp          conv        da
#              sda          dbn          rbm
#    gpu times[3.72957802,  9.94316864,  29.1772666,  9.13857198, 25.91144657,
#              18.30802011, 53.38651466, 285.41386175]
#    expected [3.076634879, 7.555234910, 18.99226785, 9.58915591, 24.130070450,
#              24.77524018, 92.66246653, 322.340329170]
#              sgd,         cg           mlp          conv        da
#              sda          dbn          rbm
#expected/get [0.82492841,  0.75984178,  0.65092691,  1.04930573, 0.93125138
#              1.35324519 1.7356905   1.12937868]
    expected_times_gpu = numpy.asarray([3.07663488, 7.55523491, 18.99226785,
                                        9.6, 24.13007045,
                                        20.4,  56, 302.6, 315.4])
    expected_times_64 = [s for idx, s in enumerate(expected_times_64)
                         if to_exec[idx]]
    expected_times_32 = [s for idx, s in enumerate(expected_times_32)
                         if to_exec[idx]]
    expected_times_gpu = [s for idx, s in enumerate(expected_times_gpu)
                          if to_exec[idx]]

    def time_test(m, l, idx, f, **kwargs):
        if not to_exec[idx]:
            return
        print algo[idx]
        ts = m.call_time
        try:
            f(**kwargs)
        except Exception, e:
            print >> sys.stderr, 'test', algo[idx], 'FAILED', e
            l.append(numpy.nan)
            return
        te = m.call_time
        l.append(te - ts)

    def do_tests():
        m = theano.compile.mode.get_default_mode()
        l = []
        time_test(m, l, 0, logistic_sgd.sgd_optimization_mnist, n_epochs=30)
        time_test(m, l, 1, logistic_cg.cg_optimization_mnist, n_epochs=30)
        time_test(m, l, 2, mlp.test_mlp, n_epochs=5)
        time_test(m, l, 3, convolutional_mlp.evaluate_lenet5, n_epochs=5,
                  nkerns=[5, 5])
        time_test(m, l, 4, dA.test_dA, training_epochs=2,
                  output_folder='tmp_dA_plots')
        time_test(m, l, 5, SdA.test_SdA, pretraining_epochs=1,
                  training_epochs=2, batch_size=300)
        time_test(m, l, 6, DBN.test_DBN, pretraining_epochs=1,
                  training_epochs=2, batch_size=300)
        time_test(m, l, 7, rbm.test_rbm, training_epochs=1, batch_size=300,
                  n_chains=1, n_samples=1, output_folder='tmp_rbm_plots')
        time_test(m, l, 8, rnnrbm.test_rnnrbm, num_epochs=1)
        return numpy.asarray(l)

    #test in float64 in FAST_RUN mode on the cpu
    import theano
    if do_float64:
        theano.config.floatX = 'float64'
        theano.config.mode = 'FAST_RUN'
        float64_times = do_tests()
        print >> sys.stderr, algo_executed
        print >> sys.stderr, 'float64 times', float64_times
        print >> sys.stderr, 'float64 expected', expected_times_64
        print >> sys.stderr, 'float64 % expected/get', (
            expected_times_64 / float64_times)

    #test in float32 in FAST_RUN mode on the cpu
    theano.config.floatX = 'float32'
    if do_float32:
        float32_times = do_tests()
        print >> sys.stderr, algo_executed
        print >> sys.stderr, 'float32 times', float32_times
        print >> sys.stderr, 'float32 expected', expected_times_32
        print >> sys.stderr, 'float32 % expected/get', (
            expected_times_32 / float32_times)

        if do_float64:
            print >> sys.stderr, 'float64/float32', (
                float64_times / float32_times)
            print >> sys.stderr
            print >> sys.stderr, 'Duplicate the timing to have everything in one place'
            print >> sys.stderr, algo_executed
            print >> sys.stderr, 'float64 times', float64_times
            print >> sys.stderr, 'float64 expected', expected_times_64
            print >> sys.stderr, 'float64 % expected/get', (
                expected_times_64 / float64_times)
            print >> sys.stderr, 'float32 times', float32_times
            print >> sys.stderr, 'float32 expected', expected_times_32
            print >> sys.stderr, 'float32 % expected/get', (
                expected_times_32 / float32_times)

            print >> sys.stderr, 'float64/float32', (
                float64_times / float32_times)
            print >> sys.stderr, 'expected float64/float32', (
                expected_times_64 / float32_times)

    #test in float32 in FAST_RUN mode on the gpu
    import theano.sandbox.cuda
    if do_gpu:
        theano.sandbox.cuda.use('gpu')
        gpu_times = do_tests()
        print >> sys.stderr, algo_executed
        print >> sys.stderr, 'gpu times', gpu_times
        print >> sys.stderr, 'gpu expected', expected_times_gpu
        print >> sys.stderr, 'gpu % expected/get', (
            expected_times_gpu / gpu_times)

        if do_float64:
            print >> sys.stderr, 'float64/gpu', float64_times / gpu_times

        if (do_float64 + do_float32 + do_gpu) > 1:
            print >> sys.stderr
            print >> sys.stderr, 'Duplicate the timing to have everything in one place'
            print >> sys.stderr, algo_executed
            if do_float64:
                print >> sys.stderr, 'float64 times', float64_times
                print >> sys.stderr, 'float64 expected', expected_times_64
                print >> sys.stderr, 'float64 % expected/get', (
                    expected_times_64 / float64_times)
            if do_float32:
                print >> sys.stderr, 'float32 times', float32_times
                print >> sys.stderr, 'float32 expected', expected_times_32
                print >> sys.stderr, 'float32 % expected/get', (
                    expected_times_32 / float32_times)
            if do_gpu:
                print >> sys.stderr, 'gpu times', gpu_times
                print >> sys.stderr, 'gpu expected', expected_times_gpu
                print >> sys.stderr, 'gpu % expected/get', (
                    expected_times_gpu / gpu_times)

            if do_float64 and do_float32:
                print >> sys.stderr, 'float64/float32', (
                    float64_times / float32_times)
                print >> sys.stderr, 'expected float64/float32', (
                    expected_times_64 / float32_times)
            if do_float64 and do_gpu:
                print >> sys.stderr, 'float64/gpu', float64_times / gpu_times
                print >> sys.stderr, 'expected float64/gpu', (
                    expected_times_64 / gpu_times)
            if do_float32 and do_gpu:
                print >> sys.stderr, 'float32/gpu', float32_times / gpu_times
                print >> sys.stderr, 'expected float32/gpu', (
                    expected_times_32 / gpu_times)

    def compare(x, y):
        ratio = x / y
        # If there is more then 5% difference between the expected
        # time and the real time, we consider this an error.
        return sum((ratio < 0.95) + (ratio > 1.05))

    if do_float64:
        err = compare(expected_times_64, float64_times)
        print >> sys.stderr, 'speed_failure_float64=' + str(err)
    if do_float32:
        err = compare(expected_times_32, float32_times)
        print >> sys.stderr, 'speed_failure_float32=' + str(err)
    if do_gpu:
        err = compare(expected_times_gpu, gpu_times)
        print >> sys.stderr, 'speed_failure_gpu=' + str(err)

        assert not numpy.isnan(gpu_times).any()
