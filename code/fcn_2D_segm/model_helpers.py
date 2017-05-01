import theano
import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer


def freezeParameters(net, single=True):
    """
    Freeze parameters of a layer or a network so that they are not trainable
    anymore

    Parameters
    ----------
    net: a network layer
    single: whether to freeze a single layer of all of the layers below as well
    """
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].remove('trainable')
            except KeyError:
                pass


def unfreezeParameters(net, single=True):
    """
    Unfreeze parameters of a layer or a network so that they become trainable
    again

    Parameters
    ----------
    net: a network layer
    single: whether to freeze a single layer of all of the layers below as well
    """
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].add('trainable')
            except KeyError:
                pass


def softmax4D(x):
    """
    Softmax activation function for a 4D tensor of shape (b, c, 0, 1)

    Parameters
    ----------
    net: x - 4d tensor with shape (b, c, 0, 1)
    """
    # Compute softmax activation
    stable_x = x - theano.gradient.zero_grad(x.max(1, keepdims=True))
    exp_x = stable_x.exp()
    softmax_x = exp_x / exp_x.sum(1)[:, None, :, :]

    return softmax_x


def concatenate(net, in_layer, concat_h, concat_vars, pos):
    """
    Auxiliary function that checks whether we should concatenate the output of
    a layer `in_layer` of a network `net` to some a tensor in `concat_vars`

    Parameters
    ----------
    net: dictionary containing layers of a network
    in_layer: name of a layer in net
    concat_h: list of layers to concatenate
    concat_vars: list of variables (tensors) to concatenate
    pos: position in lists `concat_h` and `concat_vars` we want to check
    """
    if pos < len(concat_h) and concat_h[pos] == 'input':
        concat_h[pos] = in_layer

    # if this is the layer we want to concatenate, create an InputLayer with the
    # tensor we want to concatenate and a ConcatLayer that does the job afterwards
    if in_layer in concat_h:
        net[in_layer + '_h'] = InputLayer((None, net[in_layer].input_shape[1] if
                                      (concat_h[pos] != 'noisy_input' and
                                      concat_h[pos] != 'input')
                                      else 3, None, None), concat_vars[pos])
        net[in_layer + '_concat'] = ConcatLayer((net[in_layer + '_h'],
                                            net[in_layer]), axis=1, cropping=None)
        pos += 1
        out = in_layer + '_concat'

        laySize = net[out].output_shape
        n_cl = laySize[1]
        print('Number of feature maps (concat):', n_cl)
    else:
        out = in_layer

    if concat_h and pos <= len(concat_h) and concat_h[pos-1] == 'noisy_input':
        concat_h[pos-1] = 'input'

    return pos, out
