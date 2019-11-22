# TF2.0 DeblurGAN

TensorFlow 2.0 implementation of [raphaelmeudec/deblur-gan](https://www.github.com/raphaelmeudec/deblur-gan)


## Installation

```
virtualenv venv -p python3.6
. venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Tensorflow JS

- Export the Python Model to a JS one with `scripts/convert_to_tfjs.py`
  - Problem: ReflectionPadding is defined as a custom model and cannot be loaded in JS
- Install environment `npm install`
- Start a server with CORS enabled: `http-server . --cors -o`
