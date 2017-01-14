from __future__ import absolute_import, print_function, division
import sys

import numpy

import convolutional_mlp
import dA
import DBN
import logistic_cg
import logistic_sgd
import mlp
import rbm
import rnnrbm
import SdA
import rnnslu
import lstm


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


def test_rnnslu():
    s = {'fold': 3,
         # 5 folds 0,1,2,3,4
         'data': 'atis',
         'lr': 0.0970806646812754,
         'verbose': 1,
         'decay': True,
         # decay on the learning rate if improvement stops
         'win': 7,
         # number of words in the context window
         'nhidden': 200,
         # number of hidden units
         'seed': 345,
         'emb_dimension': 50,
         # dimension of word embedding
         'nepochs': 1, # CHANGED
         'savemodel': False}
    rnnslu.main(s)


def test_lstm():
    lstm.train_lstm(max_epochs=1, test_size=1000, saveto='')


def speed():
    """
    This fonction modify the configuration theano and don't restore it!
    """

    algo = ['logistic_sgd', 'logistic_cg', 'mlp', 'convolutional_mlp',
            'dA', 'SdA', 'DBN', 'rbm', 'rnnrbm', 'rnnslu', 'lstm']
    to_exec = [True] * len(algo)
#    to_exec = [False] * len(algo)
#    to_exec[-1] = True
    do_float64 = True
    do_float32 = True
    do_gpu = True

    algo_executed = [s for idx, s in enumerate(algo) if to_exec[idx]]
 
    def time_test(m, l, idx, f, **kwargs):
        if not to_exec[idx]:
            return
        print(algo[idx])
        ts = m.call_time
        try:
            f(**kwargs)
        except Exception as e:
            print('test', algo[idx], 'FAILED', e, file=sys.stderr)
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
        s = {'fold': 3,
             # 5 folds 0,1,2,3,4
             'data': 'atis',
             'lr': 0.0970806646812754,
             'verbose': 1,
             'decay': True,
             # decay on the learning rate if improvement stops
             'win': 7,
             # number of words in the context window
             'nhidden': 200,
             # number of hidden units
             'seed': 345,
             'emb_dimension': 50,
             # dimension of word embedding
             'nepochs': 1,
             # 60 is recommended
             'savemodel': False}
        time_test(m, l, 9, rnnslu.main, param=s)
        time_test(m, l, 10, lstm.train_lstm, max_epochs=1, test_size=1000,
                  saveto='')
        return numpy.asarray(l)

    # Initialize test count and results dictionnary
    test_total = 0
    times_dic = {}

    #test in float64 in FAST_RUN mode on the cpu
    import theano
    if do_float64:
        theano.config.floatX = 'float64'
        theano.config.mode = 'FAST_RUN'
        float64_times = do_tests()
        times_dic['float64'] = float64_times
        test_total += numpy.size(float64_times)
        print(algo_executed, file=sys.stderr)
        print('float64 times', float64_times, file=sys.stderr)

    #test in float32 in FAST_RUN mode on the cpu
    theano.config.floatX = 'float32'
    if do_float32:
        float32_times = do_tests()
        times_dic['float32'] = float32_times
        test_total += numpy.size(float32_times)
        print(algo_executed, file=sys.stderr)
        print('float32 times', float32_times, file=sys.stderr)

        if do_float64:
            print('float64/float32', (
                float64_times / float32_times), file=sys.stderr)
            print(file=sys.stderr)
            print(('Duplicate the timing to have everything '
                                  'in one place'), file=sys.stderr)
            print(algo_executed, file=sys.stderr)
            print('float64 times', float64_times, file=sys.stderr)
            print('float32 times', float32_times, file=sys.stderr)

            print('float64/float32', (
                float64_times / float32_times), file=sys.stderr)

    #test in float32 in FAST_RUN mode on the gpu
    import theano.gpuarray
    if do_gpu:
        theano.gpuarray.use('cuda')
        gpu_times = do_tests()
        times_dic['gpu'] = gpu_times
        test_total += numpy.size(gpu_times)
        print(algo_executed, file=sys.stderr)
        print('gpu times', gpu_times, file=sys.stderr)

        if do_float64:
            print('float64/gpu', float64_times / gpu_times, file=sys.stderr)

        if (do_float64 + do_float32 + do_gpu) > 1:
            print(file=sys.stderr)
            print(('Duplicate the timing to have everything '
                                  'in one place'), file=sys.stderr)
            print(algo_executed, file=sys.stderr)
            if do_float64:
                print('float64 times', float64_times, file=sys.stderr)
            if do_float32:
                print('float32 times', float32_times, file=sys.stderr)
            if do_gpu:
                print('gpu times', gpu_times, file=sys.stderr)

            print()
            if do_float64 and do_float32:
                print('float64/float32', (
                    float64_times / float32_times), file=sys.stderr)
            if do_float64 and do_gpu:
                print('float64/gpu', float64_times / gpu_times, file=sys.stderr)
            if do_float32 and do_gpu:
                print('float32/gpu', float32_times / gpu_times, file=sys.stderr)
        
    # Generate JUnit performance report
    for label, times in times_dic.items():
        with open('speedtests_{label}.xml'.format(label=label), 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<testsuite name="dlt_speedtests_{label}" tests="{ntests}">\n'
                    .format(label=label, ntests=test_total/len(times_dic)))
            for algo, time in zip(algo_executed, times):
                f.write('   <testcase classname="speed" name="{algo}" time="{time}">'
                        .format(label=label, algo=algo, time=time))
                f.write('   </testcase>\n')
            f.write('</testsuite>\n')

    if do_gpu:
        assert not numpy.isnan(gpu_times).any()
