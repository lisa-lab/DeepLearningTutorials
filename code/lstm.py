'''
Build a tweet sentiment analyzer
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import random

from collections import OrderedDict

import imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}


def get_minibatches_idx(n, nb_batches, shuffle=False):

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(nb_batches):
        if i < n % nb_batches:
            minibatch_size = n // nb_batches + 1
        else:
            minibatch_size = n // nb_batches

        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return zip(range(nb_batches), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype('float32')
    # rconv
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype('float32')
    params['b'] = numpy.zeros((options['ydim'],)).astype('float32')

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'rconv': ('param_init_rconv', 'rconv_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer')}


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def param_init_fflayer(options, params, prefix='ff'):
    weights = numpy.random.randn(options['dim_proj'], options['dim_proj'])
    biases = numpy.zeros((options['dim_proj'], ))
    params[_p(prefix, 'W')] = 0.01 * weights.astype('float32')
    params[_p(prefix, 'b')] = biases.astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv', **kwargs):
    pre_act = (tensor.dot(state_below,
                          tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])
    return eval(options['activ'])(pre_act)


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def param_init_lstm(options, params, prefix='lstm'):
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype('float32')

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(0., n_samples,
                                                           dim_proj),
                                              tensor.alloc(0., n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


def param_init_rconv(options, params, prefix='rconv'):
    params[_p(prefix, 'W')] = ortho_weight(options['dim_proj'])
    params[_p(prefix, 'U')] = ortho_weight(options['dim_proj'])
    b = numpy.zeros((options['dim_proj'],)).astype('float32')
    params[_p(prefix, 'b')] = b
    gw = 0.01 * numpy.random.randn(options['dim_proj'], 3).astype('float32')
    params[_p(prefix, 'GW')] = gw
    gu = 0.01 * numpy.random.randn(options['dim_proj'], 3).astype('float32')
    params[_p(prefix, 'GU')] = gu
    params[_p(prefix, 'Gb')] = numpy.zeros((3,)).astype('float32')

    return params


def rconv_layer(tparams, state_below, options, prefix='rconv', mask=None):
    nsteps = state_below.shape[0]

    assert mask is not None

    def _step(m_, p_):
        l_ = p_
        # new activation
        ps_ = tensor.zeros_like(p_)
        ps_ = tensor.set_subtensor(ps_[1:], p_[:-1])
        ls_ = ps_
        ps_ = tensor.dot(ps_, tparams[_p(prefix, 'U')])
        pl_ = tensor.dot(p_, tparams[_p(prefix, 'W')])
        newact = eval(options['activ'])(ps_+pl_+tparams[_p(prefix, 'b')])

        # gater
        gt_ = (tensor.dot(ls_, tparams[_p(prefix, 'GU')]) +
               tensor.dot(l_, tparams[_p(prefix, 'GW')]) +
               tparams[_p(prefix, 'Gb')])
        if l_.ndim == 3:
            gt_shp = gt_.shape
            gt_ = gt_.reshape((gt_shp[0] * gt_shp[1], gt_shp[2]))
        gt_ = tensor.nnet.softmax(gt_)
        if l_.ndim == 3:
            gt_ = gt_.reshape((gt_shp[0], gt_shp[1], gt_shp[2]))

        if p_.ndim == 3:
            gn = gt_[:, :, 0].dimshuffle(0, 1, 'x')
            gl = gt_[:, :, 1].dimshuffle(0, 1, 'x')
            gr = gt_[:, :, 2].dimshuffle(0, 1, 'x')
        else:
            gn = gt_[:, 0].dimshuffle(0, 'x')
            gl = gt_[:, 1].dimshuffle(0, 'x')
            gr = gt_[:, 2].dimshuffle(0, 'x')

        act = newact * gn + ls_ * gl + l_ * gr

        if p_.ndim == 3:
            m_ = m_.dimshuffle('x', 0, 'x')
        else:
            m_ = m_.dimshuffle('x', 0)
        return tensor.switch(m_, act, l_)

    rval, updates = theano.scan(_step,
                                sequences=[mask[1:]],
                                outputs_info=[state_below],
                                name='layer_%s' % prefix,
                                n_steps=nsteps-1)

    seqlens = tensor.cast(mask.sum(axis=0), 'int64')-1
    roots = rval[-1]

    if state_below.ndim == 3:
        def _grab_root(seqlen, one_sample, prev_sample):
            return one_sample[seqlen]

        dim_proj = options['dim_proj']
        roots, updates = theano.scan(_grab_root,
                                     sequences=[seqlens,
                                                roots.dimshuffle(1, 0, 2)],
                                     outputs_info=[tensor.alloc(0., dim_proj)],
                                     name='grab_root_%s' % prefix)
    else:
        roots = roots[seqlens]  # there should be only one, so it's fine.

    return roots


def adadelta(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup+rg2up)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup)

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U'])+tparams['b'])

    f_pred_prob = theano.function([x, mask], pred)
    f_pred = theano.function([x, mask], pred.argmax(axis=1))

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + 1e-8).mean()

    return trng, use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype('float32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy.float32(valid_err) / len(data[0])

    return valid_err


def train(dim_proj=100,
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          activ='lambda x: tensor.tanh(x)',
          decay_c=0.,
          lrate=0.01,
          n_words=100000,
          data_sym=False,
          optimizer='rmsprop',
          encoder='rconv',
          saveto='model.npz',
          noise_std=0.,
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          maxlen=50,
          batch_size=16,
          valid_batch_size=16,
          dataset='sentiment140',
          use_dropout=False):

    # Model options
    model_options = locals().copy()

    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    train, valid, test = load_data(n_words=n_words, valid_portion=0.01)

    ydim = numpy.max(train[1])+1

    model_options['ydim'] = ydim

    print 'Building model'
    params = init_params(model_options)
    tparams = init_tparams(params)

    (trng, use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U']**2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost)

    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad = theano.function([x, mask, y], grads)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads,
                                              x, mask, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]),
                                   len(valid[0]) / valid_batch_size,
                                   shuffle=True)
    kf_test = get_minibatches_idx(len(test[0]),
                                  len(test[0]) / valid_batch_size,
                                  shuffle=True)

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        kf = get_minibatches_idx(len(train[0]), len(train[0])/batch_size,
                                 shuffle=True)

        for _, train_index in kf:
            n_samples += train_index.shape[0]
            uidx += 1
            use_noise.set_value(1.)

            y = [train[1][t] for t in train_index]
            x, mask, y = prepare_data([train[0][t]for t in train_index],
                                      y, maxlen=maxlen)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            cost = f_grad_shared(x, mask, y)
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = pred_error(f_pred, prepare_data, train, kf)
                valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                test_err = pred_error(f_pred, prepare_data, test, kf_test)

                history_errs.append([valid_err, test_err])

                if (uidx == 0 or
                    valid_err <= numpy.array(history_errs)[:,
                                                           0].min()):

                    best_p = unzip(tparams)
                    bad_counter = 0
                if (len(history_errs) > patience and
                    valid_err >= numpy.array(history_errs)[:-patience,
                                                           0].min()):
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                print ('Train ', train_err, 'Valid ', valid_err,
                       'Test ', test_err)

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = pred_error(f_pred, prepare_data, train, kf)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err,
                valid_err=valid_err, test_err=test_err,
                history_errs=history_errs, **params)

    return train_err, valid_err, test_err


def main(job_id, params):
    print ('Anything printed here will end up in the output directory'
           'for job #%d' % job_id)
    print params
    use_dropout = True if params['use-dropout'][0] else False
    trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        dim_proj=params['dim-proj'][0],
                                        n_words=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        activ=params['activ'][0],
                                        encoder=params['encoder'][0],
                                        maxlen=600,
                                        batch_size=16,
                                        valid_batch_size=16,
                                        validFreq=10000,
                                        dispFreq=10,
                                        saveFreq=100000,
                                        dataset='imdb',
                                        use_dropout=use_dropout)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['model_lstm.npz'],
        'encoder': ['lstm'],
        'dim-proj': [128],
        'n-words': [10000],
        'optimizer': ['adadelta'],
        'activ': ['lambda x: tensor.tanh(x)'],
        'decay-c': [0.],
        'use-dropout': [1],
        'learning-rate': [0.0001]})
