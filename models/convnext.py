import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.initializers import TruncatedNormal

def trunc_normal(stddev):
    return TruncatedNormal(stddev=stddev)

def layer_norm(epsilon=1e-6):
    return layers.LayerNormalization(epsilon=epsilon, center=True, scale=True)

def drop_path(rate):
    return layers.Dropout(rate, noise_shape=(None, 1, 1, 1))

class DepthwiseConv2D(layers.Layer):
    def __init__(self, kernel_size, strides, padding, **kwargs):
        super().__init__(**kwargs)
        self.dw_conv = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False)

    def call(self, inputs):
        return self.dw_conv(inputs)

class PointwiseConv2D(layers.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.pw_conv = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(self.out_channels,), initializer='ones', trainable=True)

    def call(self, inputs):
        x = self.pw_conv(inputs)
        return self.gamma * x

class ConvNeXtBlock(Model):
    def __init__(self, dim, drop_path=0., kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='same')
        self.norm = layer_norm()
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.pwconv2 = PointwiseConv2D(4 * dim, dim)
        self.drop_path = drop_path(rate=drop_path) if drop_path > 0. else lambda x: x

    def call(self, inputs):
        x = self.dwconv(inputs)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return layers.add([x, inputs])

class ConvNeXt(Model):
    def __init__(self, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.stem = layers.Conv2D(dims[0], kernel_size=4, strides=4, use_bias=False, kernel_initializer=trunc_normal(0.02))
        self.stem_norm = layer_norm()

        self.stages = []
        for i in range(len(depths)):
            blocks = [ConvNeXtBlock(dim=dims[i], kernel_size=kernel_size, drop_path=drop_path_rate * float(j) / sum(depths)) for j in range(depths[i])]
            self.stages.append(blocks)
            if i < len(depths) - 1:
                # Adding downsampling layer
                self.stages.append([
                    layers.Conv2D(dims[i + 1], kernel_size=2, strides=2, use_bias=False, kernel_initializer=trunc_normal(0.02)),
                    layer_norm()
                ])

        self.head_norm = layer_norm()
        self.head = layers.Dense(num_classes, kernel_initializer=trunc_normal(0.02))

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.stem_norm(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.head_norm(x)
        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
        x = self.head(x)
        return x

# Example of model creation
model = ConvNeXt(num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
model.build(input_shape=(None, 224, 224, 3))
model.summary()

