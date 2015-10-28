#!/bin/sh

which wget >/dev/null 2>&1
WGET=$?
which curl >/dev/null 2>&1
CURL=$?
if [ "$WGET" -eq 0 ]; then
    DL_CMD="wget -c"
elif [ "$CURL" -eq 0 ]; then
    DL_CMD="curl -C - -O"
else
    echo "You need wget or curl installed to download"
    exit 1
fi

$DL_CMD http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
$DL_CMD http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz
$DL_CMD http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl.gz && gunzip imdb.pkl.gz
$DL_CMD http://www.iro.umontreal.ca/~lisa/deep/data/imdb.dict.pkl.gz && gunzip imdb.dict.pkl.gz
$DL_CMD http://www.iro.umontreal.ca/~lisa/deep/data/Nottingham.zip && unzip -u Nottingham.zip
$DL_CMD http://www.iro.umontreal.ca/~lisa/deep/midi.zip && unzip -u midi.zip -d ../code && echo "extracted Modified Python MIDI package (GPL)"
$DL_CMD http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold0.pkl.gz
$DL_CMD http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold1.pkl.gz
$DL_CMD http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold2.pkl.gz
$DL_CMD http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold3.pkl.gz
$DL_CMD http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold4.pkl.gz
