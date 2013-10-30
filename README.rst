Deep Learning Tutorials
=======================

Deep Learning is a new area of Machine Learning research, which has been
introduced with the objective of moving Machine Learning closer to one of its
original goals: Artificial Intelligence.  Deep Learning is about learning
multiple levels of representation and abstraction that help to make sense of
data such as images, sound, and text.  The tutorials presented here will
introduce you to some of the most important deep learning algorithms and will
also show you how to run them using Theano.  Theano is a python library that
makes writing deep learning models easy, and gives the option of training them
on a GPU.

The easiest way to follow the tutorials is to `browse them online
<http://deeplearning.net/tutorial/>`_.

`Main development <http://github.com/lisa-lab/DeepLearningTutorials>`_
of this project.

.. image:: https://secure.travis-ci.org/lisa-lab/DeepLearningTutorials.png
   :target: http://travis-ci.org/lisa-lab/DeepLearningTutorials

Project Layout
--------------

Subdirectories:

- code - Python files corresponding to each tutorial
- data - data and scripts to download data that is used by the tutorials
- doc  - restructured text used by Sphinx to build the tutorial website
- html - built automatically by doc/Makefile, contains tutorial website
- issues_closed - issue tracking
- issues_open - issue tracking
- misc - administrative scripts


Build instructions
------------------

To build the html version of the tutorials, install sphinx and run doc/Makefile
