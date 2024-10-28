#!/bin/bash

set -xe

CC="gcc"
LIBS="-lm"

if [ ! -d "mnist" ]; then
	mkdir mnist
    wget https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/train-images-idx3-ubyte.gz -P mnist
    gunzip mnist/train-images-idx3-ubyte.gz
	wget https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/train-labels-idx1-ubyte.gz -P mnist
    gunzip mnist/train-labels-idx1-ubyte.gz
fi

mkdir -p ./build
$CC mnist.c -o ./build/mnist $LIBS
./build/mnist
