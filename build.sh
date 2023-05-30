#!/bin/bash

set -xe

export PKG_CONFIG_PATH=$HOME/Software/raylib/lib/pkgconfig/

LIBS="-lm"
gcc `pkg-config --cflags raylib` -o test test.c `pkg-config --libs raylib` $LIBS
