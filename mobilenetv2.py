# insighted by https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py

import shy
shy.err_hook()

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

if __name__ == '__main__':
    model = mobilenetv2(n_class=1000, input_size=224, width_mult=1.0)
    # print(model(np.random.randn(1, 224, 224, 3).astype(np.float32)))
    model.summary()
    # h5_path = 'weights/mobilenetv2.h5'
    # save_model(model, h5_path)

    # converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
    # tflite_model = converter.convert()
    # open('tflite/mobilenetv2.tflite', 'wb').write(tflite_model)

























# # model = models.mobilenet_v2()
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.a = nn.Conv2d(3, 64, 1, 1, 0, groups=1)
#         self.b = nn.Conv2d(64, 64, 1, 1, 0, groups=64)
#         self.c = nn.Conv2d(64, 64, 1, 1, 0, groups=64)
#         self.d = nn.Conv2d(64, 64, 1, 1, 0, groups=64)
#         self.e = nn.Conv2d(64, 64, 1, 1, 0, groups=64)
    
#     def forward(self, x):
#         x = self.a(x)
#         x = self.b(x)
#         x = self.c(x)
#         x = self.d(x)
#         x = self.e(x)
#         return x
# model = Model()
# import pdb; pdb.set_trace()
# torch.onnx.export(model, torch.ones(1, 3, 224, 224), 'mobilenetv2.onnx')

# import onnx
# from onnx_tf.backend import prepare

# model = onnx.load('mobilenetv2.onnx')
# onnx.checker.check_model(model)

# tf_rep = prepare(model)
# print(tf_rep.inputs)
# print(tf_rep.outputs)
# print(tf_rep.tensor_dict)

# tf_rep.export_graph('mobilenetv2.pb')

# import tensorflow as tf

# with tf.Session(graph=tf.Graph()) as sess:
#     with tf.gfile.GFile('mobilenetv2.pb', "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
    
#     tf.import_graph_def(graph_def, name='')
    
#     # for op in sess.graph.get_operations():
#     #     print(op.name)

#     input_tensor = sess.graph.get_tensor_by_name(tf_rep.tensor_dict[tf_rep.inputs[0]].name)
#     output_tensor = sess.graph.get_tensor_by_name(tf_rep.tensor_dict[tf_rep.outputs[0]].name)
#     converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor], [output_tensor])
#     tflite_model = converter.convert()
#     open("mobilenetv2_pth.tflite", "wb").write(tflite_model)



















# class Inverted_Residual(tf.keras.Model):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(Inverted_Residual, self).__init__()
#         hidden_dim = inp*expand_ratio
#         use_res_connect = stride == 1 and inp == oup

#         self.ir_block = Sequential()

#         # pw
#         if expand_ratio != 1:
#             self.ir_block.add(Conv2D(hidden_dim, kernel_size=1, strides=1, padding='same'))
#             self.ir_block.add(BatchNormalization())
#             self.ir_block.add(ReLU(max_value=6))

#         # dw
#         self.ir_block.add(DepthwiseConv2D(kernel_size=3, strides=stride, padding='same'))
#         self.ir_block.add(BatchNormalization())
#         self.ir_block.add(ReLU(max_value=6))

#         # pw
#         self.ir_block.add(Conv2D(oup, kernel_size=1, strides=1, padding='same'))
#         self.ir_block.add(BatchNormalization())
    
#     def call(self, x):
#         x = self.ir_block(x)
#         return x

# class MobileNetV2(tf.keras.Model):
#     def __init__(self, n_class, input_size, width_mult=1.0):
#         super(MobileNetV2, self).__init__()
#         inverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1]
#         ]

#         input_channel = 32
#         last_channel = 1280
        
#         self.input_channel = int(input_channel * width_mult)
#         self.last_channel = max(last_channel, int(last_channel*width_mult))

#         block = Inverted_Residual

#         self.feature = Sequential()

#         self.feature.add(Conv2D(self.input_channel, kernel_size=3, strides=2, padding='same'))
#         self.feature.add(BatchNormalization())
#         self.feature.add(ReLU(max_value=6))
        
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = int(c*width_mult)
#             for i in range(n):
#                 if i == 0:
#                     self.feature.add(block(input_channel, output_channel, s, t))
#                 else:
#                     self.feature.add(block(input_channel, output_channel, 1, t))
#             input_channel = output_channel

#         self.feature.add(Conv2D(self.last_channel, kernel_size=1, strides=1, padding='same'))
#         self.feature.add(BatchNormalization())
#         self.feature.add(ReLU(max_value=6))
#         self.feature.add(GlobalAveragePooling2D())
#         self.feature.add(Flatten())

#         self.last_linear = Dense(n_class, activation='softmax')
    
#     @tf.function
#     def call(self, x):
#         x = self.feature(x)
#         x = self.last_linear(x)
#         return x

# model = MobileNetV2(n_class=1000, input_size=224, width_mult=1.0)
# # model._set_inputs(np.zeros((1, 224, 224, 3)))
# # tf.saved_model.save_keras_model(model, 'mobilenet_v2/', serving_only=True)
# # tf.saved_model.save(model, 'mobilenet_v2/',
# #     signatures=model.call.get_concrete_function(tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name='inp'))
# # )
# keras.experimental.export_saved_model(model, 'mobilenet_v2/')

# converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_v2/")
# tflite_model = converter.convert()
# open('mobilenet_v2_tf.tflite', 'wb').write(tflite_model)



# def mobilenetv2(*args, **kwargs):
#     inp = Input(shape=[224, 224, 3])
#     x = Conv2D(64, kernel_size=1, strides=1, padding='same')(inp)
#     x = DepthwiseConv2D(kernel_size=1, strides=1, padding='same')(x)
#     x = DepthwiseConv2D(kernel_size=1, strides=1, padding='same')(x)
#     x = DepthwiseConv2D(kernel_size=1, strides=1, padding='same')(x)
#     x = DepthwiseConv2D(kernel_size=1, strides=1, padding='same')(x)
#     model = Model(inputs=inp, outputs=x)
#     return model
