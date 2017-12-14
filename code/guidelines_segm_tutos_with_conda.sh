#!/usr/bin/env bash
### Base installation.

# Create and enter main directory.
mkdir main_directory
cd main_directory
# Create and activate conda environment.
conda create --yes -n tuto python=2
source activate tuto
# Install theano.
conda install --yes -c mila-udem theano
# Install Lasagne.
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne/
pip install -e .
cd ..
# Install dataset_loaders.
conda install --yes matplotlib numpy Pillow scipy scikit-image seaborn h5py
git clone https://github.com/fvisin/dataset_loaders.git
cd dataset_loaders/
pip install -e .
cd ..
# Create config.ini.
cd dataset_loaders/dataset_loaders
touch config.ini
cd ../../
# Get tutorials code.
git clone https://github.com/lisa-lab/DeepLearningTutorials.git

# NB: Don't forget to correctly set config.ini with section [general]
# and other relevant sections for segmentation tutorials before
# running following lines.
# Field `datasets_local_path` in [general] section should indicate a working
# directory for dataset_loaders module. You can use a directory within
# the main directory, for example main_directory/datasets_local_dir.
# If specified folder does not exist, it will be created.

# NB: Following lines should be executed in the main directory created above.
# If any problem occures, consider deleting folder save_models (created by tutorial scripts)
# and wordking directory you specified for dataset_loaders:
# rm -rf save_models datasets_local_dir

### Tutorial FCN 2D.
## Get polyps_split7.zip from https://drive.google.com/file/d/0B_60jvsCt1hhZWNfcW4wbHE5N3M/view
## Directory for [polyps912] section in config.ini should be full path to main_directory/polyps_split7
unzip polyps_split7.zip
THEANO_FLAGS=device=cuda,floatX=float32 python DeepLearningTutorials/code/fcn_2D_segm/train_fcn8.py --num_epochs 60

### Tutorial UNET.
## Get test-volume.tif, train-labels.tif, train-volume.tif from ISBI challenge: http://brainiac2.mit.edu/isbi_challenge/home
## Directory for [isbi_em_stacks] section in config.ini should be full path to main_directory/isbi
pip install simpleitk
mkdir isbi
mv test-volume.tif  train-labels.tif  train-volume.tif isbi
THEANO_FLAGS=device=cuda,floatX=float32 python DeepLearningTutorials/code/unet/train_unet.py --num_epochs 60

### Tutorial FCN 1D.
## Get TrainingData190417.tar.gz from https://drive.google.com/file/d/0B3tbeSUS2FsVOVlIamlDdkNBQUE/edit
## Directory for [cortical_layers] section in config.ini should be full path to main_directory/cortical_layers
mkdir cortical_layers
cd cortical_layers/
tar -xvf ../TrainingData190417.tar.gz
mv TrainingData 6layers_segmentation
cd ..
THEANO_FLAGS=device=cuda,floatX=float32 python DeepLearningTutorials/code/cnn_1D_segm/train_fcn1D.py --num_epochs 60
