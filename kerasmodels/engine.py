import tensorflow as tf
import keras
backend = tf.keras
# backend = keras

def backend_wrapper(func):
    def wrapped_func(*args, **kwargs):
        if 'tensorflow' in backend.__name__: # tf.keras
            func.__globals__['keras'] = tf.keras
            func.__globals__['L'] = tf.keras.layers
            func.__globals__['Model'] = tf.keras.models.Model
        elif backend == keras: # keras
            func.__globals__['keras'] = keras
            func.__globals__['L'] = keras.layers
            func.__globals__['Model'] = keras.models.Model
        else:
            raise ValueError('backend error')
        return func(*args, **kwargs)
    return wrapped_func