import tensorflow as tf
from tensorflow.keras import Model, layers, applications, initializers

def trunc_normal(std=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=std)

class InceptionDWConv2d(layers.Layer):
    """ Inception depthwise convolution layer """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125, **kwargs):
        super().__init__(**kwargs)
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = layers.DepthwiseConv2D(square_kernel_size, padding='same', depth_multiplier=1)
        self.dwconv_w = layers.DepthwiseConv2D((1, band_kernel_size), padding='same', depth_multiplier=1)
        self.dwconv_h = layers.DepthwiseConv2D((band_kernel_size, 1), padding='same', depth_multiplier=1)
        self.gate_channels = gc

    def call(self, inputs):
        channels = inputs.shape[-1]
        id_channels = channels - 3 * self.gate_channels
        
        x_id, x_hw, x_w, x_h = tf.split(inputs, [id_channels, self.gate_channels, self.gate_channels, self.gate_channels], axis=-1)
        return tf.concat([
            x_id,
            self.dwconv_hw(x_hw),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
        ], axis=-1)

class ConvMlp(layers.Layer):
    """ MLP using 1x1 convolutions """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=layers.ReLU, norm_layer=layers.LayerNormalization, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = layers.Conv2D(hidden_features, 1, use_bias=True)
        self.norm = norm_layer(axis=[1, 2]) if norm_layer else layers.LayerNormalization(axis=[1, 2])
        self.act = act_layer()
        self.drop = layers.Dropout(dropout_rate)
        self.fc2 = layers.Conv2D(out_features, 1, use_bias=True)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MetaNeXtBlock(layers.Layer):
    """ MetaNeXtBlock layer """
    def __init__(self, dim, token_mixer=None, norm_layer=layers.BatchNormalization, mlp_ratio=4, act_layer=layers.Activation('gelu'), ls_init_value=1e-6, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.token_mixer = token_mixer(dim) if token_mixer else layers.LayerNormalization(axis=[1, 2])
        self.norm = norm_layer(axis=[1, 2])
        self.mlp = ConvMlp(dim, int(mlp_ratio * dim), act_layer=act_layer, norm_layer=None, dropout_rate=dropout_rate)
        self.gamma = self.add_weight(name='gamma', shape=(dim,), initializer=initializers.Constant(ls_init_value), trainable=True)

    def call(self, inputs, training=None):
        shortcut = inputs
        x = self.token_mixer(inputs)
        x = self.norm(x)
        x = self.mlp(x)
        x = x * self.gamma
        return x + shortcut

class MetaNeXtStage(Model):
    """ MetaNeXt Stage model """
    def __init__(self, in_chs, out_chs, depth=2, token_mixer=None, act_layer=layers.Activation('gelu'), norm_layer=layers.BatchNormalization, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [MetaNeXtBlock(out_chs, token_mixer=token_mixer, norm_layer=norm_layer, mlp_ratio=mlp_ratio, act_layer=act_layer) for _ in range(depth)]
        self.downsample = layers.Conv2D(out_chs, 1, strides=2) if in_chs != out_chs else layers.Lambda(lambda x: x)

    def call(self, inputs, training=None):
        x = self.downsample(inputs)
        for block in self.blocks:
            x = block(x, training=training)
        return x

class MetaNeXt(Model):
    """ MetaNeXt model """
    def __init__(self, in_chans=3, num_classes=1000, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), token_mixers=InceptionDWConv2d, norm_layer=layers.BatchNormalization, act_layer=layers.Activation('gelu'), mlp_ratios=(4, 4, 4, 3), **kwargs):
        super().__init__(**kwargs)
        self.stem = layers.Conv2D(dims[0], 4, strides=4, padding='same')
        self.stages = [MetaNeXtStage(dims[i - 1], dims[i], depth=depths[i], token_mixer=token_mixers, act_layer=act_layer, norm_layer=norm_layer, mlp_ratio=mlp_ratios[i]) for i in range(4)]
        self.head = ConvMlp(dims[-1], dims[-1] * mlp_ratios[-1], num_classes, act_layer=act_layer, norm_layer=None, dropout_rate=0.5)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        for stage in self.stages:
            x = stage(x, training=training)
        x = tf.reduce_mean(x, [1, 2])
        x = self.head(x)
        return x

# Example of initializing and using the model
model = MetaNeXt()
