# codes and weights are converted from https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py copyrighted by soumith

from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D, ReLU, Dropout, Concatenate, 
    Input, GlobalAveragePooling2D,
    MaxPooling2D, ZeroPadding2D
)
from tensorflow.keras.models import Model

_MODEL_URLS = {
    'squeezenet1_0': {
        'URL': 'https://github.com/kevin970401/keras-pretrainedmodels/releases/download/download/squeezenet1_0.h5',
        'HASH': '02cf552933596c1379207c257dacb8e8'
        },
    'squeezenet1_1': {
        'URL': 'https://github.com/kevin970401/keras-pretrainedmodels/releases/download/download/squeezenet1_1.h5',
        'HASH': '8b67f4e6eb5d77a00b28215455be7104'
    },
}

def fire(inputs, squeeze_planes, expand1x1_planes, expand3x3_planes, prefix=''):
    x = Conv2D(squeeze_planes, kernel_size=1, strides=1, padding='same', name=prefix+'.squeeze')(inputs)
    x = ReLU(name=prefix+'.squeeze_activation')(x)
    branch1 = Conv2D(expand1x1_planes, kernel_size=1, strides=1, padding='same', name=prefix+'.expand1x1')(x)
    branch1 = ReLU(name=prefix+'.expand1x1_activation')(branch1)
    branch2 = Conv2D(expand3x3_planes, kernel_size=3, strides=1, padding='same', name=prefix+'.expand3x3')(x)
    branch2 = ReLU(name=prefix+'.expand3x3_activation')(branch2)
    x = Concatenate()([branch1, branch2])
    return x

def squeezenet(input_shape=[224, 224, 3], version='1_0', num_classes=1000):
    inputs = Input(shape=input_shape)
    x = inputs
    prefix = 'features'
    if version == '1_0':
        x = Conv2D(96, kernel_size=7, strides=2, padding='valid', name=prefix+'.0')(x)
        x = ReLU(name=prefix+'.1')(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name=prefix+'.2')(x)

        x = fire(x, 16, 64, 64, prefix=prefix+'.3')
        x = fire(x, 16, 64, 64, prefix=prefix+'.4')
        x = fire(x, 32, 128, 128, prefix=prefix+'.5')
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x) # ceil_mode = True in torchvision squeezenet1_0
        x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name=prefix+'.6')(x)

        x = fire(x, 32, 128, 128, prefix=prefix+'.7')
        x = fire(x, 48, 192, 192, prefix=prefix+'.8')
        x = fire(x, 48, 192, 192, prefix=prefix+'.9')
        x = fire(x, 64, 256, 256, prefix=prefix+'.10')
        x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name=prefix+'.11')(x)

        x = fire(x, 64, 256, 256, prefix=prefix+'.12')
    if version == '1_1':
        x = Conv2D(64, kernel_size=3, strides=2, padding='valid', name=prefix+'.0')(x)
        x = ReLU(name=prefix+'.1')(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name=prefix+'.2')(x)

        x = fire(x, 16, 64, 64, prefix=prefix+'.3')
        x = fire(x, 16, 64, 64, prefix=prefix+'.4')
        x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name=prefix+'.5')(x)
        

        x = fire(x, 32, 128, 128, prefix=prefix+'.6')
        x = fire(x, 32, 128, 128, prefix=prefix+'.7')
        x = MaxPooling2D(pool_size=3, strides=2, padding='valid', name=prefix+'.8')(x)
        
        x = fire(x, 48, 192, 192, prefix=prefix+'.9')
        x = fire(x, 48, 192, 192, prefix=prefix+'.10')
        x = fire(x, 64, 256, 256, prefix=prefix+'.11')
        x = fire(x, 64, 256, 256, prefix=prefix+'.12')

    prefix = 'classifier'
    x = Dropout(0.5, name=prefix+'.0')(x)
    x = Conv2D(num_classes, kernel_size=1, strides=1, padding='same', name=prefix+'.1')(x)
    x = ReLU(name=prefix+'.2')(x)
    x = GlobalAveragePooling2D(name=prefix+'.3')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def _load_pretrained(model_name, model):
    if model_name not in _MODEL_URLS or _MODEL_URLS[model_name] is None:
        raise ValueError("No checkpoint is available for model type {}".format(model_name))
    checkpoint_url = _MODEL_URLS[model_name]['URL']
    checkpoint_hash = _MODEL_URLS[model_name]['HASH']
    weights_path = keras.utils.get_file('{}.h5'.format(model_name), checkpoint_url, cache_subdir='models', file_hash=checkpoint_hash)
    model.load_weights(weights_path)
    return model

def squeezenet1_0(input_shape=(224, 224, 3), num_classes=1000, pretrained=False):
    model = squeezenet(input_shape=input_shape, version='1_0', num_classes=num_classes)
    if pretrained:
        model = _load_pretrained('squeezenet1_0', model)
    return model

def squeezenet1_1(input_shape=(224, 224, 3), num_classes=1000, pretrained=False):
    model = squeezenet(input_shape=input_shape, version='1_1', num_classes=num_classes)
    if pretrained:
        model = _load_pretrained('squeezenet1_1', model)
    return model
