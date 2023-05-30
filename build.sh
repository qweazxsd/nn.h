#!/bin/bash

set -xe

export PKG_CONFIG_PATH=$HOME/Software/raylib/lib/pkgconfig/

LIBS="-lm"
CFLAGS="-Wall -Wextra"

clang `pkg-config --cflags raylib` -o xor xor.c `pkg-config --libs raylib` $LIBS
