.. _deep:

Deep Learning
=============

The breakthrough to effective training strategies for deep architectures came in
2006 with the algorithms for training deep belief networks
(DBN) [Hinton07]_ and stacked auto-encoders [Ranzato07]_ , [Bengio07]_ .
All these methods are based on a similar approach: **greedy layer-wise unsupervised
pre-training** followed by **supervised fine-tuning**.

The pretraining strategy consists in using unsupervised learning to guide the
training of intermediate levels of representation. Each layer is pre-trained
with an unsupervised learning algorithm, which attempts to learn a nonlinear
transformation of its input, in order to captures its main variations.  Higher
levels of abstractions are created by feeding the output of one layer, to the
input of the subsequent layer.

The resulting an architecture can then be seen in two lights:

* the pre-trained deep network can be used to initialize the weights of all, but
  the last layer of a deep neural network. The weights are then further adapted
  to a supervised task (such as classification) through traditional gradient
  descent (see :ref:`Multilayer perceptron <mlp>`). This is referred to as the
  fine-tuning step.

* the pre-trained deep network can also serve solely as a feature extractor. The
  output of the last layer is fed to a classifier, such as logistic regression,
  which is trained independently. Better results can be obtained by
  concatenating the output of the last layer, with the hidden representations of
  all intermediate layers [Lee09]_.

For the purposes of this tutorial, we will focus on the first interpretation,
as that is what was first proposed in [Hinton06]_. 

Deep Coding
+++++++++++

Since Deep Belief Networks (DBN) and Stacked Denoising-AutoEncoders (SDA) share
much of the same architecture and have very similar training algorithms (in
terms of pretraining and fine-tuning stages), it makes sense to implement them
in a similar fashion, as part of a "Deep Learning" framework.

We thus define a generic interface, which both of these architectures will
share.

.. code-block:: python

    class DeepLayerwiseModel(object):

        def layerwise_pretrain(self, layer_fns, pretrain_amounts):
            """
            """

        def finetune(self, datasets, lr, batch_size):
            """

    class DBN(DeepLayerwiseModel):
        """
        """

    class StackedDAA(DeepLayerwiseModel):
        """
        """

.. code-block:: python

    def deep_main(learning_rate=0.1,
            pretraining_epochs=20,
            pretrain_lr=0.1,
            training_epochs=1000,
            batch_size=20,
            mnist_file='mnist.pkl.gz'):
     
        n_train_examples, train_valid_test = load_mnist(mnist_file)

        # instantiate model
        deep_model = ...

        ####
        #### Phase 1: Pre-training
        ####

        # create an array of functions, which will be used for the greedy
        # layer-wise unsupervised training procedure

        pretrain_functions = deep_model.pretrain_functions(
                batch_size=batch_size,
                train_set_x=train_set_x,
                learning_rate=pretrain_lr,
                ...
                )

        # loop over all the layers in our network
        for layer_idx, pretrain_fn in enumerate(pretrain_functions):

            # iterate over a certain number of epochs) 
            for i in xrange(pretraining_epochs * n_train_examples / batch_size):

                # follow one step in the gradient of the unsupervised cost
                # function, at the given layer
                layer_fn(i)
    

.. code-block:: python

        ####
        #### Phase 2: Fine Tuning
        ####

        # create theano functions for fine-tuning, as well as
        # validation and testing our model.

        train_fn, valid_scores, test_scores =\
            deep_model.finetune_functions(
                train_valid_test[0][0],       # training dataset
                learning_rate=finetune_lr,    # the learning rate
                batch_size=batch_size)        # number of examples to use at once

        
        # use these functions as part of the generic early-stopping procedure
        for i in xrange(patience_max):

            if i >= patience:
                break

            cost_i = train_fn(i)

            ...







