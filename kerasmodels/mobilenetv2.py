# insighted by https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
import torch
import torch.nn  as nn
from torchvision import models

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, BatchNormalization, Activation, ReLU, Dropout, ZeroPadding2D,
    AveragePooling2D, Input, Flatten, DepthwiseConv2D, GlobalAveragePooling2D, add
)
from tensorflow.keras.models import Model, save_model

import numpy as np

def _inverted_residual(inputs, inp, oup, stride, expand_ratio, prefix='', eps=1e-5):
    assert stride in [1, 2], f'stride: {stride} not in [1, 2]'

    hidden_dim = inp*expand_ratio
    use_res_connect = stride == 1 and inp == oup

    x = inputs

    prefix = prefix + '.conv'

    layer_no = 0
    # pw
    if expand_ratio != 1:
        x = Conv2D(hidden_dim, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.{layer_no}.0')(x)
        x = BatchNormalization(name=f'{prefix}.{layer_no}.1', epsilon=eps)(x)
        x = ReLU(max_value=6, name=f'{prefix}.{layer_no}.2')(x)
        layer_no += 1

    # dw
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='valid', use_bias=False, name=f'{prefix}.{layer_no}.0')(x)
    x = BatchNormalization(name=f'{prefix}.{layer_no}.1', epsilon=eps)(x)
    x = ReLU(max_value=6, name=f'{prefix}.{layer_no}.2')(x)

    layer_no += 1

    # pw
    x = Conv2D(oup, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.{layer_no}')(x)
    layer_no += 1
    x = BatchNormalization(name=f'{prefix}.{layer_no}', epsilon=eps)(x)

    if use_res_connect:
        return add([x, inputs])
    return x

def mobilenetv2(n_class=1000, input_size=224, width_mult=1.0, eps=1e-5):

    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    input_channel = 32
    last_channel = 1280
    
    input_channel = int(input_channel * width_mult)
    last_channel = max(last_channel, int(last_channel*width_mult))

    block = _inverted_residual

    inputs = Input(shape=[input_size, input_size, 3])

    prefix = 'features'
    block_no = 0
    x = inputs
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(input_channel, kernel_size=3, strides=2, padding='valid', use_bias=False, name=f'{prefix}.{block_no}.0')(x)
    x = BatchNormalization(name=f'{prefix}.{block_no}.1', epsilon=eps)(x)
    x = ReLU(max_value=6, name=f'{prefix}.{block_no}.2')(x)

    block_no += 1
    for t, c, n, s in inverted_residual_setting:
        output_channel = int(c*width_mult)
        for i in range(n):
            stride = s if i==0 else 1
            x = block(x, input_channel, output_channel, stride, t, f'{prefix}.{block_no}')
            input_channel = output_channel
            block_no += 1
    x = Conv2D(last_channel, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.{block_no}.0')(x)
    x = BatchNormalization(name=f'{prefix}.{block_no}.1', epsilon=eps)(x)
    x = ReLU(max_value=6, name=f'{prefix}.{block_no}.2')(x)
    x = GlobalAveragePooling2D()(x)

    prefix = 'classifier'
    x = Dropout(0.2, name=f'{prefix}.0')(x)
    x = Dense(n_class, name=f'{prefix}.1')(x)
    model = Model(inputs=inputs, outputs=x)
    return model
