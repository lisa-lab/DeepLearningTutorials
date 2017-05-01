import numpy as np

import theano.tensor as T

import lasagne
# start-snippet-1
from lasagne.layers import InputLayer, ConcatLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseMergeLayer
from lasagne.layers import Deconv2DLayer as DeconvLayer
# end-snippet-1
from lasagne.nonlinearities import softmax


# start-snippet-2
def buildUnet(nb_in_channels, dropout, input_var=None,
              path_unet="/data/lisatmp4/erraqabi/results/Unet/" +
              "polyp_unet_drop_penal1e-05_dataAugm_nbEpochs100/" +
              "u_net_model.npz",
              nclasses=2, trainable=False, padding=92):
    """
    Build u-net model
    """

    net = {}
    net['input'] = InputLayer((None, nb_in_channels, None, None),input_var)

    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=padding)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3)

    net['pool1'] = PoolLayer(net['conv1_2'], 2, ignore_border=False)

    # end-snippet-2

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3)

    net['pool2'] = PoolLayer(net['conv2_2'], 2, ignore_border=False)

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3)

    net['pool3'] = PoolLayer(net['conv3_2'], 2, ignore_border=False)

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3)

    if dropout:
        net['drop1'] = DropoutLayer(net['conv4_2'])
        prev_layer1 = 'drop1'
    else:
        prev_layer1 = 'conv4_2'

    net['pool4'] = PoolLayer(net[prev_layer1], 2, ignore_border=False)

    net['conv5_1'] = ConvLayer(net['pool4'], 1024, 3)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 1024, 3)

    if dropout:
        net['drop2'] = DropoutLayer(net['conv5_2'])
        prev_layer2 = 'drop2'
    else:
        prev_layer2 = 'conv5_2'

    net['upconv4'] = DeconvLayer(net[prev_layer2], 512, 2, stride=2)
    net['Concat_4'] = ConcatLayer(
        (net['conv4_2'], net['upconv4']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv6_1'] = ConvLayer(net['Concat_4'], 512, 3)
    net['conv6_2'] = ConvLayer(net['conv6_1'], 512, 3)

    net['upconv3'] = DeconvLayer(net['conv6_2'], 256, 2, stride=2)
    net['Concat_3'] = ConcatLayer(
        (net['conv3_2'], net['upconv3']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv7_1'] = ConvLayer(net['Concat_3'], 256, 3)
    net['conv7_2'] = ConvLayer(net['conv7_1'], 256, 3)

    net['upconv2'] = DeconvLayer(net['conv7_2'], 128, 2, stride=2)
    net['Concat_2'] = ConcatLayer(
        (net['conv2_2'], net['upconv2']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv8_1'] = ConvLayer(net['Concat_2'], 128, 3)
    net['conv8_2'] = ConvLayer(net['conv8_1'], 128, 3)

    # start-snippet-3
    net['upconv1'] = DeconvLayer(net['conv8_2'], 64, 2, stride=2)
    net['Concat_1'] = ConcatLayer(
        (net['conv1_2'], net['upconv1']), axis=1,
        cropping=[None, None, 'center', 'center'])
    # end-snippet-3
    net['conv9_1'] = ConvLayer(net['Concat_1'], 64, 3)
    net['conv9_2'] = ConvLayer(net['conv9_1'], 64, 3)

    net['conv10'] = ConvLayer(net['conv9_2'], nclasses, 1,
                              nonlinearity=lasagne.nonlinearities.identity)
    net['input_tmp'] = InputLayer((None, nclasses, None, None),
                                  input_var[:, :-1, :-2*padding, :-2*padding])

    net['final_crop'] = ElemwiseMergeLayer((net['input_tmp'], net['conv10']),
                                           merge_function=lambda input, deconv:
                                           deconv,
                                           cropping=[None, None,
                                                     'center', 'center'])


    net_final = lasagne.layers.DimshuffleLayer(net['final_crop'], (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net_final).shape
    net_final = lasagne.layers.ReshapeLayer(net_final,
                                            (T.prod(laySize[0:3]),
                                             laySize[3]))
    net_final = lasagne.layers.NonlinearityLayer(net_final,
                                                 nonlinearity=softmax)

    return net_final

# start-snippet-4
if __name__ == '__main__':

    unet = buildUnet(input_var=T.tensor4('inputs_var'),
                    nb_in_channels = 3,
                    n_classes = 2,
                    dropout=True,
                    trainable=True)
# end-snippet-4
