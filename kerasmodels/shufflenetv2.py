# insighted by https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Dense, Conv2D, BatchNormalization, 
    Activation, ReLU, Dropout, Concatenate, 
    Permute, AveragePooling2D, Input, Flatten, 
    DepthwiseConv2D, GlobalAveragePooling2D,
    Reshape, GlobalMaxPooling2D, MaxPooling2D,
    Lambda, ZeroPadding2D
)
from tensorflow.keras.models import Model, save_model

def _ChannelShuffle(inputs, groups):
    _, h, w, c = inputs.shape
    if c % groups != 0:
        raise ValueError('c%groups = {c%groups} != 0,')
    x = Reshape((h*w, groups, c//groups))(inputs)
    x = Permute((1, 3, 2))(x)
    x = Reshape((h, w, c))(x)
    return x

def _InvertedResidual(inputs, inp, oup, strides, prefix='', eps=1e-5):
    if not strides in [1, 2]: 
        raise ValueError('illegal strides value')

    branch_features = oup // 2
    assert (strides != 1) or (inp == branch_features << 1)
    
    if strides == 1:
        branch1 = Lambda(lambda x: x[:,:,:,:inp//2])(inputs)
        branch2 = Lambda(lambda x: x[:,:,:,inp//2:])(inputs)
    else:
        branch1 = inputs
        branch2 = inputs
        branch1_prefix = prefix+'.branch1'
        branch1 = ZeroPadding2D(padding=(1, 1))(branch1)
        branch1 = DepthwiseConv2D(kernel_size=3, strides=strides, padding='valid', use_bias=False, name=branch1_prefix+'.0')(branch1)
        branch1 = BatchNormalization(epsilon=eps, name=branch1_prefix+'.1')(branch1)
        branch1 = Conv2D(branch_features, kernel_size=1, strides=1, padding='same', use_bias=False, name=branch1_prefix+'.2')(branch1)
        branch1 = BatchNormalization(epsilon=eps, name=branch1_prefix+'.3')(branch1)
        branch1 = ReLU(name=branch1_prefix+'.4')(branch1)

    branch2_prefix = prefix+'.branch2'
    branch2 = Conv2D(branch_features, kernel_size=1, strides=1, padding='same', use_bias=False, name=branch2_prefix+'.0')(branch2)
    branch2 = BatchNormalization(epsilon=eps, name=branch2_prefix+'.1')(branch2)
    branch2 = ReLU(name=branch2_prefix+'.2')(branch2)
    branch2 = ZeroPadding2D(padding=(1, 1))(branch2)
    branch2 = DepthwiseConv2D(kernel_size=3, strides=strides, padding='valid', use_bias=False, name=branch2_prefix+'.3')(branch2)
    branch2 = BatchNormalization(epsilon=eps, name=branch2_prefix+'.4')(branch2)
    branch2 = Conv2D(branch_features, kernel_size=1, strides=1, padding='same', use_bias=False, name=branch2_prefix+'.5')(branch2)
    branch2 = BatchNormalization(epsilon=eps, name=branch2_prefix+'.6')(branch2)
    branch2 = ReLU(name=branch2_prefix+'.7')(branch2)
    out = Concatenate()([branch1, branch2])
    out = _ChannelShuffle(out, 2)
    return out

def make_shuffleNetV2(input_size=224, stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024], n_class=1000, eps=1e-5):
    if len(stages_repeats) != 3:
        raise ValueError('expected stages_repeats as list of 3 positive ints')
    if len(stages_out_channels) != 5:
        raise ValueError('expected stages_out_channels as list of 5 positive ints')

    inp_ch = 3
    oup_ch = stages_out_channels[0]

    inputs = Input(shape=[input_size, input_size, 3])
    prefix = 'conv1'

    x = inputs
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(oup_ch, kernel_size=3, strides=2, padding='valid', use_bias=False, name=prefix+'.0')(x)
    x = BatchNormalization(epsilon=eps, name=prefix+'.1')(x)
    x = ReLU(name=prefix+'.2')(x)

    inp_ch = oup_ch

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
    for name, repeats, oup_ch in zip(stage_names, stages_repeats, stages_out_channels[1:]):
        x = _InvertedResidual(x, inp_ch, oup_ch, 2, prefix=f'{name}.{0}')
        for i in range(repeats-1):
            x = _InvertedResidual(x, oup_ch, oup_ch, 1, prefix=f'{name}.{i+1}')
        inp_ch = oup_ch

    oup_ch = stages_out_channels[-1]

    prefix = 'conv5'
    x = Conv2D(oup_ch, kernel_size=1, strides=1, padding='same', use_bias=False, name=prefix+'.0')(x)
    x = BatchNormalization(epsilon=eps, name=prefix+'.1')(x)
    x = ReLU(name=prefix+'.2')(x)
    x = GlobalAveragePooling2D(name='globalavgpool')(x)
    x = Dense(n_class, name='fc')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

def shufflenet_v2(input_size=224, n_class=1000, width_multiplier=1.0):
    assert width_multiplier in [0.5, 1.0, 1.5, 2.0], f'width multiplier {width_multiplier} not in [0.5, 1.0, 1.5, 2.0]'
    stage = {
        0.5: [[4, 8, 4], [24, 48, 96, 192, 1024]],
        1.0: [[4, 8, 4], [24, 116, 232, 464, 1024]],
        1.5: [[4, 8, 4], [24, 176, 352, 704, 1024]],
        2.0: [[4, 8, 4], [24, 244, 488, 976, 2048]],
    }

    return make_shuffleNetV2(input_size=input_size, n_class=n_class, stages_repeats=stage[width_multiplier][0], stages_out_channels=stage[width_multiplier][1])
