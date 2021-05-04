import tensorflow as tf
# tf.config.run_functions_eagerly(True)
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.python.keras.backend import conv2d
from INlayers import IN, AdaIn

# Conv Layer
class conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(conv, self).__init__()
        self.conv2d = Conv2D(filters, kernel, stride, use_bias=False, padding='same')
        self.IN = IN()

    def call(self, inputs, relu = True):
        # Only apply/ activate with ReLu after IN
        x = self.conv2d(inputs)
        x = self.IN(x)
        if relu:
            x = tf.nn.relu(x)

        return x

class residual(tf.keras.layers.Layer):
    # Fuck Gradient Disappearing
    # A residual structure/block that contains two conv layers
    def __init__(self, filters, kernel, stride):
        super(residual, self).__init__()
        self.c1 = conv(filters, kernel, stride)
        self.c2 = conv(filters, kernel, stride)

    def call(self, inputs):
        x = self.c1(inputs)
        out = inputs + self.c2(x, relu=False)
        return tf.nn.relu(out)

# ConvTranspose Layer

class convT(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(convT, self).__init__()
        self.convT = conv(filters, kernel, stride)
        self.instance_norm = IN()
        self.stride = stride

    def call(self, inputs):
        new_h = inputs.shape[1] * self.stride * 2
        new_w = inputs.shape[2] * self.stride * 2
        x = tf.image.resize(inputs, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.convT(x)
        # return x

        # Redundant
        x = self.instance_norm(x)

        return tf.nn.relu(x)


class Net(tf.keras.Model):
    # In Encoder and Decoder Structure:
    def __init__(self):
        super(Net, self).__init__()
        # Encoder
        self.c1 = conv(32, 9, 1)
        self.c2 = conv(64, 3, 2)
        self.c3 = conv(128, 3, 2)
        self.r1 = residual(128, 3, 1)
        self.r2 = residual(128, 3, 1)
        self.r3 = residual(128, 3, 1)
        self.r4 = residual(128, 3, 1)
        self.r5 = residual(128, 3, 1)
        # Decoder
        self.cT1 = convT(64, 3, 2)
        self.cT2 = convT(32, 3, 2)
        self.c4 = conv(3, 9, 1)

    def call(self,inputs):
        print(inputs.shape)
        x = self.c1(inputs)
        print(x.shape)
        x = self.c2(x)
        print(x.shape)
        x = self.c3(x)
        print(x.shape)
        x = self.r1(x)
        print(x.shape)
        x = self.r2(x)
        print(x.shape)
        x = self.r3(x)
        print(x.shape)
        x = self.r4(x)
        print(x.shape)
        x = self.r5(x)
        print(x.shape)

        x = self.cT1(x)
        print(x.shape)
        x = self.cT2(x)
        print(x.shape)
        x = self.c4(x, relu=False)
        print(x.shape)
        return x

