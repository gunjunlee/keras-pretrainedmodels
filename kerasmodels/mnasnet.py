import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Dense, Conv2D, BatchNormalization, 
    Activation, ReLU, Dropout, Concatenate, 
    Permute, AveragePooling2D, Input, Flatten, 
    DepthwiseConv2D, GlobalAveragePooling2D,
    Reshape, GlobalMaxPooling2D, MaxPooling2D,
    Lambda, add, ZeroPadding2D
)
from tensorflow.keras.models import Model, save_model

def _InvertedResidual(inputs, out_ch, kernel_size, strides, expandsion_factor, eps=1e-5, prefix=''):
    assert strides in [1, 2]
    assert kernel_size in [3, 5]
    in_ch = inputs.shape[-1]
    mid_ch = in_ch.value * expandsion_factor
    apply_residual = (in_ch == out_ch and strides == 1)

    prefix = prefix + '.layers'
    x = Conv2D(mid_ch, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.0')(inputs)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.1')(x)
    x = ReLU(name=f'{prefix}.2')(x)

    if strides == 2:
        x = ZeroPadding2D(padding=kernel_size//2)(x)
    padding = 'valid' if strides == 2 else 'same'
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=f'{prefix}.3')(x)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.4')(x)
    x = ReLU(name=f'{prefix}.5')(x)

    x = Conv2D(out_ch, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.6')(x)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.7')(x)

    if apply_residual:
        x = add([x, inputs])

    return x

def _stack(inputs, out_ch, kernel_size, strides, expandsion_factor, repeats, prefix=''):
    assert repeats >= 1

    x = _InvertedResidual(inputs, out_ch, kernel_size, strides, expandsion_factor, prefix=f'{prefix}.0')
    for i in range(1, repeats):
        x = _InvertedResidual(x, out_ch, kernel_size, 1, expandsion_factor, prefix=f'{prefix}.{i}')
    return x

def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val+divisor/2) //divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor

def _scale_depths(depths, alpha):
    return [_round_to_multiple_of(depth*alpha, 8) for depth in depths]

def mnasnet(input_size=224, channels=3, width_multiplier=1.0, num_classes=1000, dropout=0.2, eps=1e-5):
    depths = _scale_depths([24, 40, 80, 96, 192, 320], width_multiplier)
    inputs = Input(shape=[input_size, input_size, channels])

    prefix = 'layers'
    x = inputs
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(32, kernel_size=3, strides=2, padding='valid', use_bias=False, name=f'{prefix}.0')(x)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.1')(x)
    x = ReLU(name=f'{prefix}.2')(x)

    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False, name=f'{prefix}.3')(x)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.4')(x)
    x = ReLU(name=f'{prefix}.5')(x)

    x = Conv2D(16, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.6')(x)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.7')(x)

    x = _stack(x, depths[0], 3, 2, 3, 3, prefix=f'{prefix}.8')
    x = _stack(x, depths[1], 5, 2, 3, 3, prefix=f'{prefix}.9')
    x = _stack(x, depths[2], 5, 2, 6, 3, prefix=f'{prefix}.10')
    x = _stack(x, depths[3], 3, 1, 6, 2, prefix=f'{prefix}.11')
    x = _stack(x, depths[4], 5, 2, 6, 4, prefix=f'{prefix}.12')
    x = _stack(x, depths[5], 3, 1, 6, 1, prefix=f'{prefix}.13')
    
    x = Conv2D(1280, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'{prefix}.14')(x)
    x = BatchNormalization(epsilon=eps, name=f'{prefix}.15')(x)
    x = ReLU(name=f'{prefix}.16')(x)

    prefix = 'classifier'
    x = GlobalAveragePooling2D(name='globalavgpool')(x)
    x = Dropout(dropout, name=f'{prefix}.0')(x)
    x = Dense(num_classes, name=f'{prefix}.1')(x)
    model = Model(inputs=inputs, outputs=x)
    return model