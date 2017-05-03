import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer, \
        NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import batch_norm, BatchNormLayer
from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer
from lasagne.layers import Upscale1DLayer as UpscaleLayer
from lasagne.layers import PadLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.nonlinearities import softmax, linear, rectify


def conv_bn_relu(net, incoming_layer, depth, num_filters, filter_size, pad = 'same'):
    net['conv'+str(depth)] = ConvLayer(net[incoming_layer],
                num_filters = num_filters, filter_size = filter_size,
                pad = pad, nonlinearity=None)
    net['bn'+str(depth)] = BatchNormLayer(net['conv'+str(depth)])
    net['relu'+str(depth)] = NonlinearityLayer( net['bn'+str(depth)], nonlinearity = rectify)
    incoming_layer = 'relu'+str(depth)

    return incoming_layer

# start-snippet-bn_relu_conv
def bn_relu_conv(net, incoming_layer, depth, num_filters, filter_size, pad = 'same'):

    net['bn'+str(depth)] = BatchNormLayer(net[incoming_layer])
    net['relu'+str(depth)] = NonlinearityLayer( net['bn'+str(depth)], nonlinearity = rectify)
    net['conv'+str(depth)] = ConvLayer(net['relu'+str(depth)],
                num_filters = num_filters, filter_size = filter_size,
                pad = pad, nonlinearity=None)
    incoming_layer = 'conv'+str(depth)

    return incoming_layer
# end-snippet-bn_relu_conv

# start-snippet-convolutions
def build_model(input_var,
	    n_classes = 6,
	    nb_in_channels = 2,
        filter_size=25,
        n_filters = 64,
        depth = 8,
        last_filter_size = 1,
        block = 'bn_relu_conv',
        out_nonlin = softmax):
    '''
    Parameters:
    -----------
    input_var : theano 3Dtensor shape(n_samples, n_in_channels, ray_length)
    filter_size : odd int (to fit with same padding)
    n_filters : int, number of filters for each convLayer
    n_classes : int, number of classes to segment
    depth : int, number of stacked convolution before concatenation
    last_filter_size : int, last convolution filter size to obtain n_classes feature maps
    out_nonlin : default=softmax, non linearity function
    '''


    net = {}

    net['input'] = InputLayer((None, nb_in_channels, 200), input_var)
    incoming_layer = 'input'

    #Convolution layers
    for d in range(depth):
        if block == 'bn_relu_conv':
            incoming_layer = bn_relu_conv(net, incoming_layer, depth = d,
                            num_filters= n_filters, filter_size=filter_size)
            # end-snippet-convolutions
        elif block == 'conv_bn_relu':
            incoming_layer = conv_bn_relu(net, incoming_layer, depth = d,
                            num_filters= n_filters, filter_size=filter_size)
    # start-snippet-output
    #Output layer
    net['final_conv'] = ConvLayer(net[incoming_layer],
                    num_filters = n_classes,
                    filter_size = last_filter_size,
                    pad='same')
    incoming_layer = 'final_conv'

    #DimshuffleLayer and ReshapeLayer to fit the softmax implementation
    #(it needs a 1D or 2D tensor, not a 3D tensor)
    net['final_dimshuffle'] = DimshuffleLayer(net[incoming_layer], (0,2,1))
    incoming_layer = 'final_dimshuffle'

    layerSize = lasagne.layers.get_output(net[incoming_layer]).shape
    net['final_reshape'] = ReshapeLayer(net[incoming_layer],
                                (T.prod(layerSize[0:2]),layerSize[2]))
                                # (200*batch_size,n_classes))
    incoming_layer = 'final_reshape'


    #This is the layer that computes the prediction
    net['last_layer'] = NonlinearityLayer(net[incoming_layer],
                    nonlinearity = out_nonlin)
    incoming_layer = 'last_layer'

    #Layers needed to visualize the prediction of the network
    net['probs_reshape'] = ReshapeLayer(net[incoming_layer],
                    (layerSize[0], layerSize[1], n_classes))
    incoming_layer = 'probs_reshape'

    net['probs_dimshuffle'] = DimshuffleLayer(net[incoming_layer], (0,2,1))


    return [net[l] for l in ['last_layer']], net

	# end-snippet-output

