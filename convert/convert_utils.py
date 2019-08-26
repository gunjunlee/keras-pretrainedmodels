import torch

from tensorflow.keras.layers import (
    Conv2D, Dense, BatchNormalization, 
    DepthwiseConv2D, Input, ZeroPadding2D
)

def pth2keras(pth_model, keras_model):
    m = {}
    for k, v in pth_model.named_parameters():
        m[k] = v
    for k, v in pth_model.named_buffers(): # for batchnormalization
        m[k] = v

    # m : {'classifier.1.bias': np.array(...), ...}

    with torch.no_grad():
        for layer in keras_model.layers:
            if isinstance(layer, DepthwiseConv2D):
                print(layer.name)
                weights = []
                weights.append(m[layer.name+'.weight'].permute(2, 3, 0, 1).data.numpy()) # weight
                if layer.use_bias:
                    weights.append(m[layer.name+'.bias'].data.numpy()) # bias

                layer.set_weights(weights)
            elif isinstance(layer, Conv2D):
                print(layer.name)
                weights = []
                weights.append(m[layer.name+'.weight'].permute(2, 3, 1, 0).data.numpy()) # weight
                if layer.use_bias:
                    weights.append(m[layer.name+'.bias'].data.numpy()) # bias

                layer.set_weights(weights)
            elif isinstance(layer, BatchNormalization):
                print(layer.name)
                weights = []
                if layer.scale:
                    weights.append(m[layer.name+'.weight'].data.numpy()) # gamma
                if layer.center:
                    weights.append(m[layer.name+'.bias'].data.numpy()) # beta
                weights.append(m[layer.name+'.running_mean'].data.numpy()) # running_mean
                weights.append(m[layer.name+'.running_var'].data.numpy()) # running_var
                layer.set_weights(weights)
            
            elif isinstance(layer, Dense):
                print(layer.name)
                weights = []
                weights.append(m[layer.name+'.weight'].t().data.numpy())
                if layer.use_bias:
                    weights.append(m[layer.name+'.bias'].data.numpy())
                layer.set_weights(weights)

    return keras_model