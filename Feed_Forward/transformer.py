import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.layers import Conv2D, UpSampling2D
from vgg import VGG

    # Single
    # def load_img(file_path):
    #     img = tf.io.read_file(file_path)
    #     img = tf.image.decode_jpeg(img, channels=3)
    #     img = tf.cast(img, tf.float32)
    #     img = img[tf.newaxis, :]
    #     return img
    # content_layer = "block4_conv1"
    # style_layers = [
    #     "block1_conv1",
    #     "block2_conv1",
    #     "block3_conv1",
    #     "block4_conv1",
    # ]
    # style_paths = ["../style2.jpg"]
    # sty_img = tf.concat([load_img(style_paths[0])], axis = 0)
    # # vgg = VGG(content_layer, style_layers)
    # # # Extracte the style features form style image
    # # _, style_feature_map = vgg(sty_img)


    # This Encoder is extedned on tf.keras.models.Model
class Encoder(tf.keras.models.Model):
    def __init__(self, content_layer):
        super(Encoder, self).__init__()
        vgg = VGG19(include_top=False, weights="imagenet")

        self.vgg = tf.keras.Model(
            [vgg.input], [vgg.get_layer(content_layer).output]
        )
        self.vgg.trainable = False

    def call(self, inputs):
        # get the features of content_layer
        preprocessed_input = vgg19.preprocess_input(inputs)
        feature_map = self.vgg(preprocessed_input)
        return feature_map

class TransferNet(tf.keras.Model):
    def __init__(self, content_layer):
        super(TransferNet, self).__init__()
        self.encoder = Encoder(content_layer)
        self.decoder = decoder()
        self.beta = tf.Variable(initial_value=tf.constant_initializer([0.])(shape=[1]),  trainable=True, name= 'beta' )  
        self.gamma = tf.Variable(initial_value=tf.constant_initializer([1.])(shape=[1]), trainable=True, name= 'gamma')

    def encode(self, content_image, style_image, alpha):
        content_feature_map = self.encoder(content_image)
        style_feature_map = self.encoder(style_image)

        t = instance_normalization(content_feature_map, style_feature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    def encode_IN(self, content_image, alpha):
        content_feature_map = self.encoder(content_image)
        #style_feature_map = self.encoder(style_image)

        t = instance_normalization(content_feature_map, self.beta, self.gamma)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    def encode_CIN(self, content_image, style_image, alpha):
        content_feature_map = self.encoder(content_image)
        style_feature_map = self.encoder(style_image)

        t = instance_normalization(content_feature_map, style_feature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    def encode_ADAIN(self, content_image, style_image, alpha):
        content_feature_map = self.encoder(content_image)
        style_contentfeature_map = self.encoder(style_image)

        t = instance_normalization_ADAIN(content_feature_map, style_contentfeature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    #   def encode(self, content_image, style_image, alpha):
    #       content_feature_map = self.encoder(content_image)
    #     #   style_feature_map = self.encoder(style_image)

    #     #   t = self.normalization(content_feature_map, style_feature_map)
    #       t = self.normalization(content_feature_map)
    #       t = alpha * t + (1 - alpha) * content_feature_map
    #       return t

    def decode(self, t):
        return self.decoder(t)

    def call(self, content_image, style_image, IN_METHOD, alpha=1.0):
        if IN_METHOD == 0:
            t = self.encode_IN(content_image, alpha)
        elif IN_METHOD == 1:
            t = self.encode_CIN(content_image, style_image, alpha)
        else :
            t = self.encode_ADAIN(content_image, style_image, alpha)
        g_t = self.decode(t)
        return g_t


 

def instance_normalization(content_feature_map, beta, gamma, epsilon=1e-5):

        # axes = [1, 2] means instancenorm
    content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)
    
   
    # beta = tf.Variable(initial_value=tf.constant_initializer([0.])(shape=[1]),  trainable=True, name= 'beta' )  
    # gamma = tf.Variable(initial_value=tf.constant_initializer([1.])(shape=[1]), trainable=True, name= 'gamma')

    content_feature_map_norm = tf.nn.batch_normalization(
        content_feature_map,
        mean=content_mean,
        variance=content_variance,
        offset= beta,
        scale= gamma,
        variance_epsilon=epsilon,
    )
    return content_feature_map_norm


def instance_normalization_ADAIN(content_feature_map, style_contentfeature_map, epsilon=1e-5):

        # axes = [1, 2] means instancenorm
    content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)
    
    style_mean, style_variance = tf.nn.moments(style_contentfeature_map, axes=[1, 2], keepdims=True)

    style_std = tf.math.sqrt(style_variance + epsilon)

    content_feature_map_norm = tf.nn.batch_normalization(
        content_feature_map,
        mean=content_mean,
        variance=content_variance,
        offset=style_std,
        scale=style_mean,
        variance_epsilon=epsilon,
    )
    return content_feature_map_norm   

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

    def call(self, x):
        return tf.pad(
            x,
            [
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            "REFLECT",
        )

def decoder():
    return tf.keras.Sequential(
        [
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            UpSampling2D(size=2),
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(256, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(128, (3, 3), activation="relu"),
            UpSampling2D(size=2),
            ReflectionPadding2D(),
            Conv2D(128, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(64, (3, 3), activation="relu"),
            UpSampling2D(size=2),
            ReflectionPadding2D(),
            Conv2D(64, (3, 3), activation="relu"),
            ReflectionPadding2D(),
            Conv2D(3, (3, 3)),
        ]
    )


