"""
 This tutorial introduces deep belief networks (DBN) using Theano.
"""

import numpy, time, cPickle, gzip

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM


class DBN(object):
    """ DBN """

    def __init__(self, numpy_rng, theano_rng = None, n_ins = 784, 
                 hidden_layers_sizes = [500,500], n_outs = 10):
    
        self.sigmoid_layers = []
        self.rbms           = []
        self.params         = []
        self.n_layers       = len(hidden_layers_sizes)

        assert self.n_layers > 0 

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            if i == 0 : 
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng = numpy_rng, input = layer_input, 
                                         n_in = input_size, 
                                         n_out = hidden_layers_sizes[i],
                                         activation = T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            rbm = RBM(numpy_rng = numpy_rng, theano_rng = theano_rng, input = layer_input,  
                      n_visible = input_size, 
                      n_hidden  = hidden_layers_sizes[i],
                      W = sigmoid_layer.W, hbias = sigmoid_layer.b)
            self.rbms.append(rbm)

        self.logLayer = LogisticRegression( 
                           input = self.sigmoid_layers[-1].output, 
                           n_in = hidden_layers_sizes[-1], n_out = n_outs)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

        self.params.extend(self.logLayer.params)
        self.PCD_chains = {}


            
    def build_pretraining_functions(self, train_set_x, batch_size,type = 'CD' ):

        index = T.lscalar()
        lr    = T.scalar()
            
        n_batches   = train_set_x.value.shape[0] / batch_size
        batch_begin = (index % n_batches) * batch_size
        batch_end   = batch_begin + batch_size
        data_size   = train_set_x.value.shape[1]

        pretrain_fns = []
        for rbm in self.rbms :
            if type == "CD":
                 updates = rbm.cd(lr = lr) 
            elif type == 'PCD':
                 persistent_chain = theano.shared( numpy.zeros((batch_size,data_size)))
                 self.PCD_chain[rbm] = persistent_chain
                 updates = rbm.cd(lr = lr, presistent =  persistent_chain)
            else:
                raise NotImplementedError()

            fn = theano.function([index, theano.Param(lr, default = 0.1)], [],
                           updates = updates, 
                           givens = {self.x: train_set_x[batch_begin:batch_end]})

            pretrain_fns.append(fn)

        return pretrain_fns


    def finetune(self, datasets, batch_size):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x , test_set_y ) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.value.shape[0] / batch_size
        n_test_batches  = test_set_x.value.shape[0]  / batch_size

        index   = T.lscalar()    # index to a [mini]batch 
        lr      = T.scalar()

        
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
              updates[param] = param - gparam*lr

        train_fn = theano.function(inputs = [index, theano.Param(lr,default=0.1)], 
               outputs =   self.finetune_cost, 
               updates = updates,
               givens  = {
                    self.x : train_set_x[index*batch_size:(index+1)*batch_size],
                    self.y : train_set_y[index*batch_size:(index+1)*batch_size]})

        test_score_i = theano.function([index], self.errors,
               givens = {
                   self.x: test_set_x[index*batch_size:(index+1)*batch_size],
                   self.y: test_set_y[index*batch_size:(index+1)*batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens = {
                 self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
                 self.y: valid_set_y[index*batch_size:(index+1)*batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
           return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_DBN( finetune_lr = 0.1, pretraining_epochs = 2, \
              pretrain_lr = 0.1, training_epochs = 1000, \
              dataset='mnist.pkl.gz'):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]



    batch_size = 20    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    dbn = DBN( numpy_rng = numpy_rng, n_ins = 28*28, 
               hidden_layers_sizes = [100,100,100],
               n_outs = 10)
    

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.build_pretraining_functions( 
                                        train_set_x   = train_set_x, 
                                        batch_size    = batch_size, 
                                        type = 'CD' ) 

    print '... pre-training the model'
    start_time = time.clock()  
    ## Pre-train layer-wise 
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            for batch_index in xrange(n_train_batches):
                 pretraining_fns[i](batch_index,pretrain_lr)
            print 'Pre-training layer %i, epoch %d '%(i,epoch)
 
    end_time = time.clock()

    print ('Pretraining took %f minutes' %((end_time-start_time)/60.))
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.finetune ( 
                datasets = datasets, batch_size = batch_size) 

    print '... finetunning the model'
    # early-stopping parameters
    patience              = 10000 # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(n_train_batches):

        minibatch_avg_cost = train_fn(minibatch_index, finetune_lr)
        iter    = epoch * n_train_batches + minibatch_index

        if (iter+1) % validation_frequency == 0: 
            
            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_train_batches, \
                    this_validation_loss*100.))


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = test_model()
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') % 
                             (epoch, minibatch_index+1, n_train_batches,
                              test_score*100.))


        if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))






if __name__ == '__main__':
    test_DBN()


