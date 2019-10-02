# kerasmodels

## install

### pip

```
pip install kerasmodels
```


### pip+git

```
pip install git+https://github.com/kevin970401/keras-pretrainedmodels
```

## Example

backend default: tf.keras

```
import kerasmodels as KM
model = KM.shufflenet_v2_x0_5(pretrained=True)
```

using keras as backend

```
import kerasmodels as KM
import keras
KM.engine.backend = keras
model = KM.shufflenet_v2_x0_5(pretrained=True)
```