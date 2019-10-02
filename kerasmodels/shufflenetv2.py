# codes and weights are converted from https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py copyrighted by soumith

from .engine import backend_wrapper

_MODEL_URLS = {
    'shufflenet_v2_x0_5': {
        'URL': 'https://github.com/kevin970401/keras-pretrainedmodels/releases/download/download/shufflenet_v2_x0_5.h5',
        'HASH': 'c47624f0bd2d4a4473a41edb285566ab',
    },
    'shufflenet_v2_x1_0': {
        'URL': 'https://github.com/kevin970401/keras-pretrainedmodels/releases/download/download/shufflenet_v2_x1_0.h5',
        'HASH': '5ad92b3de8af699539aad6f9f05a8c59',
    },
    'shufflenet_v2_x1_5': None,
    'shufflenet_v2_x2_0': None,
}

def _ChannelShuffle(inputs, groups):
    _, h, w, c = inputs.shape
    h, w, c = int(h), int(w), int(c) # for keras compatibility
    if c % groups != 0:
        raise ValueError('c%groups = {c%groups} != 0,')
    x = L.Reshape((h*w, groups, c//groups))(inputs)
    x = L.Permute((1, 3, 2))(x)
    x = L.Reshape((h, w, c))(x)
    return x

def _InvertedResidual(inputs, inp, oup, strides, prefix='', eps=1e-5):
    if not strides in [1, 2]: 
        raise ValueError('illegal strides value')

    branch_features = oup // 2
    assert (strides != 1) or (inp == branch_features << 1)
    
    if strides == 1:
        branch1 = L.Lambda(lambda x: x[:,:,:,:inp//2])(inputs)
        branch2 = L.Lambda(lambda x: x[:,:,:,inp//2:])(inputs)
    else:
        branch1 = inputs
        branch2 = inputs
        branch1_prefix = prefix+'.branch1'
        branch1 = L.ZeroPadding2D(padding=(1, 1))(branch1)
        branch1 = L.DepthwiseConv2D(kernel_size=3, strides=strides, padding='valid', use_bias=False, name=branch1_prefix+'.0')(branch1)
        branch1 = L.BatchNormalization(epsilon=eps, name=branch1_prefix+'.1')(branch1)
        branch1 = L.Conv2D(branch_features, kernel_size=1, strides=1, padding='same', use_bias=False, name=branch1_prefix+'.2')(branch1)
        branch1 = L.BatchNormalization(epsilon=eps, name=branch1_prefix+'.3')(branch1)
        branch1 = L.ReLU(name=branch1_prefix+'.4')(branch1)

    branch2_prefix = prefix+'.branch2'
    branch2 = L.Conv2D(branch_features, kernel_size=1, strides=1, padding='same', use_bias=False, name=branch2_prefix+'.0')(branch2)
    branch2 = L.BatchNormalization(epsilon=eps, name=branch2_prefix+'.1')(branch2)
    branch2 = L.ReLU(name=branch2_prefix+'.2')(branch2)
    branch2 = L.ZeroPadding2D(padding=(1, 1))(branch2)
    branch2 = L.DepthwiseConv2D(kernel_size=3, strides=strides, padding='valid', use_bias=False, name=branch2_prefix+'.3')(branch2)
    branch2 = L.BatchNormalization(epsilon=eps, name=branch2_prefix+'.4')(branch2)
    branch2 = L.Conv2D(branch_features, kernel_size=1, strides=1, padding='same', use_bias=False, name=branch2_prefix+'.5')(branch2)
    branch2 = L.BatchNormalization(epsilon=eps, name=branch2_prefix+'.6')(branch2)
    branch2 = L.ReLU(name=branch2_prefix+'.7')(branch2)
    out = L.Concatenate()([branch1, branch2])
    out = _ChannelShuffle(out, 2)
    return out

def make_shuffleNetV2(input_shape=[224, 224, 3], stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024], num_classes=1000, eps=1e-5):
    if len(stages_repeats) != 3:
        raise ValueError('expected stages_repeats as list of 3 positive ints')
    if len(stages_out_channels) != 5:
        raise ValueError('expected stages_out_channels as list of 5 positive ints')

    inp_ch = 3
    oup_ch = stages_out_channels[0]

    inputs = L.Input(shape=input_shape)
    prefix = 'conv1'

    x = inputs
    x = L.ZeroPadding2D(padding=(1, 1))(x)
    x = L.Conv2D(oup_ch, kernel_size=3, strides=2, padding='valid', use_bias=False, name=prefix+'.0')(x)
    x = L.BatchNormalization(epsilon=eps, name=prefix+'.1')(x)
    x = L.ReLU(name=prefix+'.2')(x)

    inp_ch = oup_ch

    x = L.ZeroPadding2D(padding=(1, 1))(x)
    x = L.MaxPooling2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
    for name, repeats, oup_ch in zip(stage_names, stages_repeats, stages_out_channels[1:]):
        x = _InvertedResidual(x, inp_ch, oup_ch, 2, prefix=f'{name}.{0}', eps=eps)
        for i in range(repeats-1):
            x = _InvertedResidual(x, oup_ch, oup_ch, 1, prefix=f'{name}.{i+1}', eps=eps)
        inp_ch = oup_ch

    oup_ch = stages_out_channels[-1]

    prefix = 'conv5'
    x = L.Conv2D(oup_ch, kernel_size=1, strides=1, padding='same', use_bias=False, name=prefix+'.0')(x)
    x = L.BatchNormalization(epsilon=eps, name=prefix+'.1')(x)
    x = L.ReLU(name=prefix+'.2')(x)
    x = L.GlobalAveragePooling2D(name='globalavgpool')(x)
    x = L.Dense(num_classes, name='fc')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

@backend_wrapper
def shufflenet_v2(input_shape=(224, 224, 3), num_classes=1000, width_multiplier=1.0, pretrained=False, eps=1e-5):
    if pretrained:
        assert width_multiplier in [0.5, 1.0, 1.5, 2.0], f'width multiplier {width_multiplier} not in [0.5, 1.0, 1.5, 2.0]'
    stage = {
        0.5: [[4, 8, 4], [24, 48, 96, 192, 1024]],
        1.0: [[4, 8, 4], [24, 116, 232, 464, 1024]],
        1.5: [[4, 8, 4], [24, 176, 352, 704, 1024]],
        2.0: [[4, 8, 4], [24, 244, 488, 976, 2048]],
    }

    return make_shuffleNetV2(input_shape=input_shape, num_classes=num_classes, stages_repeats=stage[width_multiplier][0], stages_out_channels=stage[width_multiplier][1], eps=eps)

def _load_pretrained(model_name, model):
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError("No checkpoint is available for model type {}".format(model_name))
    checkpoint_url = _MODEL_URLS[model_name]['URL']
    checkpoint_hash = _MODEL_URLS[model_name]['HASH']
    weights_path = keras.utils.get_file('{}.h5'.format(model_name), checkpoint_url, cache_subdir='models', file_hash=checkpoint_hash)
    model.load_weights(weights_path)
    return model

@backend_wrapper
def shufflenet_v2_x0_5(pretrained=False, **kwargs):
    model = shufflenet_v2(width_multiplier=0.5, **kwargs)
    if pretrained:
        model = _load_pretrained('shufflenet_v2_x0_5', model)
    return model

@backend_wrapper
def shufflenet_v2_x1_0(pretrained=False, **kwargs):
    model = shufflenet_v2(width_multiplier=1.0, **kwargs)
    if pretrained:
        model = _load_pretrained('shufflenet_v2_x1_0', model)
    return model

@backend_wrapper
def shufflenet_v2_x1_5(pretrained=False, **kwargs):
    model = shufflenet_v2(width_multiplier=1.5, **kwargs)
    if pretrained:
        model = _load_pretrained('shufflenet_v2_x1_5', model)
    return model

@backend_wrapper
def shufflenet_v2_x2_0(pretrained=False, **kwargs):
    model = shufflenet_v2(width_multiplier=2.0, **kwargs)
    if pretrained:
        model = _load_pretrained('shufflenet_v2_x2_0', model)
    return model

