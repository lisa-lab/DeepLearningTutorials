#!/bin/sh

wget -c http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
wget -c http://www.iro.umontreal.ca/~lisa/deep/data/Nottingham.zip && unzip Nottingham.zip
wget -c http://www.iro.umontreal.ca/~lisa/deep/midi.zip && unzip midi.zip -d ../code && echo "extracted Modified Python MIDI package (GPL)"
