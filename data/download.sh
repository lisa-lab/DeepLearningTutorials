#!/bin/sh

wget -c http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
wget -c http://www.iro.umontreal.ca/~lisa/deep/data/Nottingham.zip && unzip -u Nottingham.zip
wget -c http://www.iro.umontreal.ca/~lisa/deep/midi.zip && unzip -u midi.zip -d ../code && echo "extracted Modified Python MIDI package (GPL)"
