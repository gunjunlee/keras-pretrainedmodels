import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import torch.nn as nn
from torchvision import models

import tensorflow as tf
from tensorflow import keras

# from kerasmodels import (
#     mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3,
#     shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,
#     mobilenet_v2, squeezenet1_0, squeezenet1_1
# )
import numpy as np

import kerasmodels as KM
from convert_utils import pth2keras

# models.mnasnet0_5(pretrained=True)
# # models.mnasnet0_75(pretrained=True)
# models.mnasnet1_0(pretrained=True)
# # models.mnasnet1_3(pretrained=True)
# models.shufflenet_v2_x0_5(pretrained=True)
# models.shufflenet_v2_x1_0(pretrained=True)
# # models.shufflenet_v2_x1_5(pretrained=True)
# # models.shufflenet_v2_x2_0(pretrained=True)
# models.mobilenet_v2(pretrained=True)
# models.squeezenet1_0(pretrained=True)
# models.squeezenet1_1(pretrained=True)

pretrained_models = [
    'mnasnet0_5',
    'mnasnet1_0',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'mobilenet_v2',
    'squeezenet1_0',
    'squeezenet1_1',
]

if not os.path.isdir('weights'):
    os.mkdir('weights')

def check(pth_model, keras_model):
    pth_input = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    keras_input = pth_input.permute(0, 2, 3, 1).numpy()

    pth_output = pth_model(pth_input).data.numpy()
    keras_output = keras_model.predict(keras_input)

    maxx = np.abs(pth_output-keras_output).max()
    assert maxx < 1e-2, 'Error'
    return True

for model_name in pretrained_models:
    pretrained_pth_model = getattr(models, model_name)(pretrained=True).eval()
    keras_model = getattr(KM, model_name)()
    pretrained_keras_model = pth2keras(pretrained_pth_model, keras_model)
    if check(pretrained_pth_model, pretrained_keras_model):
        pretrained_keras_model.save(f'weights/{model_name}.h5')
