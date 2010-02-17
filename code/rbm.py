"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs 
to those without visible-visible and hidden-hidden connections. 
"""


import numpy
import theano
import theano.tensor as T
import time
import gzip
import cPickle

from theano.tensor.shared_randomstreams import RandomStreams

# python library dealing with images
import PIL.Image

##### FUNCTION FOR PLOTTING SAMPLES / FILTERS for RBM 


def scale_to_unit_interval(ndar,eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape,tile_spacing = (0,0), 
              scale_rows_to_unit_interval = True, output_pixel_vals = True):
    """
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can 
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.  
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
 
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image 
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        )+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array



class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=784, n_hidden=500, \
        W = None, hbias = None, vbias = None, numpy_rng = None, 
        theano_rng = None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing 
        to a shared hidden units bias vector in case RBM is part of a 
        different network

        :param vbias: None for standalone RBMs or a symbolic variable 
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden  = n_hidden


        if W is None : 
           # W is initialized with `initial_W` which is uniformely sampled
           # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
           # the output of uniform if converted using asarray to dtype 
           # theano.config.floatX so that the code is runable on GPU
           initial_W = numpy.asarray( numpy.random.uniform( 
                     low = -numpy.sqrt(6./(n_hidden+n_visible)), 
                     high = numpy.sqrt(6./(n_hidden+n_visible)), 
                     size = (n_visible, n_hidden)), 
                     dtype = theano.config.floatX)
           # theano shared variables for weights and biases
           W = theano.shared(value = initial_W, name = 'W')

        if hbias is None :
           # create shared variable for hidden units bias
           hbias = theano.shared(value = numpy.zeros(n_hidden, 
                               dtype = theano.config.floatX), name='hbias')

        if vbias is None :
            # create shared variable for visible units bias
            vbias = theano.shared(value =numpy.zeros(n_visible, 
                                dtype = theano.config.floatX),name='vbias')

        if numpy_rng is None:    
            # create a number generator 
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None : 
            theano_rng = RandomStreams(numpy_rng.randint(2**30))


        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input if input else T.dmatrix('input')

        self.W          = W
        self.hbias      = hbias
        self.vbias      = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list 
        # other than shared variables created in this function.
        self.params     = [self.W, self.hbias, self.vbias]
        self.batch_size = self.input.shape[0]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.sum(T.dot(v_sample, self.vbias))
        hidden_term = T.sum(T.log(1+T.exp(wx_b)))
        return -hidden_term - vbias_term

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of the visibles
        h1_mean = T.nnet.sigmoid(T.dot(v0_sample, self.W) + self.hbias)
        # get a sample of the hiddens given their activation
        h1_sample = self.theano_rng.binomial(size = h1_mean.shape, n = 1, prob = h1_mean)
        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = T.nnet.sigmoid(T.dot(h0_sample, self.W.T) + self.vbias)
        # get a sample of the visible given their activation
        v1_sample = self.theano_rng.binomial(size = v1_mean.shape,n = 1,prob = v1_mean)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling, 
            starting from the hidden state'''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]
 
    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling, 
            starting from the visible state'''
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [h1_mean, h1_sample, v1_mean, v1_sample]
 
    def cd(self, lr = 0.1, persistent=None):
        """ 
        This functions implements one step of CD-1 or PCD-1

        :param lr: learning rate used to train the RBM 
        :param persistent: None for CD. For PCD, shared variable containing old state
        of Gibbs chain. This must be a shared variable of size (batch size, number of
        hidden units).

        Returns the updates dictionary. The dictionary contains the update rules for weights
        and biases but also an update of the shared variable used to store the persistent
        chain, if one is used.
        """

        # compute positive phase
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        [nv_mean, nv_sample, nh_mean, nh_sample] = self.gibbs_hvh(chain_start)

        # determine gradients on RBM parameters
        g_vbias = T.sum( self.input - nv_mean, axis = 0)/self.batch_size
        g_hbias = T.sum( ph_mean    - nh_mean, axis = 0)/self.batch_size
        g_W = T.dot(ph_mean.T, self.input   )/ self.batch_size - \
              T.dot(nh_mean.T, nv_mean      )/ self.batch_size

        gparams = [g_W.T, g_hbias, g_vbias]

        # constructs the update dictionary
        updates = {}
        for gparam, param in zip(gparams, self.params):
           updates[param] = param + gparam * lr

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = T.cast(nh_sample, dtype=theano.config.floatX)


        return updates


def test_rbm( learning_rate=0.1, training_epochs = 15, \
                            dataset='mnist.pkl.gz'):
    """
    Demonstrate ***

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM 

    :param training_eqpochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    """

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x,  test_set_y  = shared_dataset(test_set)

    batch_size = 20    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    # initialize storage fot the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, 500)))

    # construct the RBM class
    rbm = RBM( input = x, n_visible=28*28, \
               n_hidden = 500,numpy_rng = rng, theano_rng = theano_rng)

    # get the cost and the gradient corresponding to one step of CD
    updates = rbm.cd(lr=learning_rate, persistent=persistent_chain)


    #################################
    #     Training the RBM          #
    #################################

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index], [],
           updates = updates, 
           givens = { x: train_set_x[index*batch_size:(index+1)*batch_size]})

    plotting_time = 0.
    start_time = time.clock()  

    # go through training epochs 
    for epoch in xrange(training_epochs):

        # go through the training set
        c = []
        for batch_index in xrange(n_train_batches):
           train_rbm(batch_index)

        print 'Training epoch %d '%epoch

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix 
        image = PIL.Image.fromarray(tile_raster_images( X = rbm.W.value.T,
                 img_shape = (28,28),tile_shape = (10,10), 
                 tile_spacing=(1,1)))
        image.save('filters_at_epoch_%i.png'%epoch) 
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' %(pretraining_time/60.))

  
    #################################
    #     Sampling from the RBM     #
    #################################

    # find out the number of test samples  
    number_of_test_samples = test_set_x.value.shape[0]

    # pick two initial starting points randomly 
    sample = rng.randint(number_of_test_samples-20)

    # Initialize the persistent chain with some sample from the test 
    persistent_vis_chain = theano.shared(test_set_x.value[sample:sample+20])

    # define one step of Gibbs sampling (mf = mean-field)
    [hid_mf, hid_sample, vis_mf, vis_sample] =  rbm.gibbs_vhv(persistent_vis_chain)

    # the sample at the end of the channel is returned by ``gibbs_1`` as 
    # its second output; note that this is computed as a binomial draw, 
    # therefore it is formed of ints (0 and 1) and therefore needs to 
    # be converted to the same dtype as ``persistent_vis_chain``
    vis_sample = T.cast(vis_sample, dtype=theano.config.floatX)

    # construct the function that implements our persistent chain 
    # we generate the "mean field" activations for plotting and the actual samples for
    # reinitializing the state of our persistent chain
    sample_fn = theano.function([], [vis_mf, vis_sample],
                      updates = { persistent_vis_chain:vis_sample})

    # sample the RBM, plotting every `plot_every`-th sample; do this 
    # until you plot at least `n_samples`
    n_samples = 10
    plot_every = 1000

    for idx in xrange(n_samples):
        # do `plot_every` intermediate samplings of which we do not care
        for jdx in  xrange(plot_every):
            vis_mf, vis_sample = sample_fn()

        # construct image
        image = PIL.Image.fromarray(tile_raster_images( 
                                         X          = vis_mf,
                                         img_shape  = (28,28),
                                         tile_shape = (10,10),
                                         tile_spacing = (1,1) ) )
        print ' ... plotting sample ', idx
        image.save('sample_%i_step_%i.png'%(idx,idx*jdx))

if __name__ == '__main__':
    test_rbm()
