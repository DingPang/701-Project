import tensorflow as tf
# tf.config.run_functions_eagerly(True)
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose

from INlayers import IN, AdaIn

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
    def __init__(self, content_layer, INMETHOD):
        super(TransferNet, self).__init__()
        self.encoder = Encoder(content_layer)
        self.decoder = decoder()
        if INMETHOD == 0:
            self.INlayer = IN()
        elif INMETHOD == 1:
            self.INlayer = IN()
        else:
            self.INlayer = AdaIn()


    def encode_IN(self, content_image, alpha):
        content_feature_map = self.encoder(content_image)

        t = self.INlayer(content_feature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    def encode_CIN(self, content_image, style_image, alpha):
        # content_feature_map = self.encoder(content_image)
        # style_feature_map = self.encoder(style_image)

        # t = instance_normalization(content_feature_map, style_feature_map)
        # t = alpha * t + (1 - alpha) * content_feature_map
        # return t
        return 0

    def encode_ADAIN(self, content_image, style_image, alpha):
        content_feature_map = self.encoder(content_image)
        style_contentfeature_map = self.encoder(style_image)

        t = self.INlayer(content_feature_map, style_contentfeature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t


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




# def instance_normalization(content_feature_map, beta, gamma, epsilon=1e-5):

#         # axes = [1, 2] means instancenorm
#     content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)

#     content_feature_map_norm = tf.nn.batch_normalization(
#         content_feature_map,
#         mean=content_mean,
#         variance=content_variance,
#         offset= beta,
#         scale= gamma,
#         variance_epsilon=epsilon,
#     )
#     return content_feature_map_norm


# def instance_normalization_ADAIN(content_feature_map, style_contentfeature_map, epsilon=1e-5):

#         # axes = [1, 2] means instancenorm
#     content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)
#     print("content_mean: {}".format(tf.shape(content_mean)))
#     print("content_variance: {}".format(tf.shape(content_variance)))

#     style_mean, style_variance = tf.nn.moments(style_contentfeature_map, axes=[1, 2], keepdims=True)
#     print("style_mean: {}".format(tf.shape(style_mean)))
#     print("style_variance: {}".format(tf.shape(style_variance)))

#     style_std = tf.math.sqrt(style_variance + epsilon)
#     print("style_std: {}".format(tf.shape(style_std)))
#     content_feature_map_norm = tf.nn.batch_normalization(
#         content_feature_map,
#         mean=content_mean,
#         variance=content_variance,
#         offset= style_mean,
#         scale= style_std,
#         variance_epsilon=epsilon,
#     )
#     print("content_feature_map_norm: {}".format(tf.shape(content_feature_map_norm)))

#     return content_feature_map_norm

# class ReflectionPadding2D(tf.keras.layers.Layer):
#     def __init__(self, padding=1, **kwargs):
#         super(ReflectionPadding2D, self).__init__(**kwargs)
#         self.padding = padding

#     def compute_output_shape(self, s):
#         return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

#     def call(self, x):
#         return tf.pad(
#             x,
#             [
#                 [0, 0],
#                 [self.padding, self.padding],
#                 [self.padding, self.padding],
#                 [0, 0],
#             ],
#             "REFLECT",
#         )

# def decoder():
#     return tf.keras.Sequential(
#         [
#             ReflectionPadding2D(),
#             Conv2D(256, (3, 3), activation="relu"),
#             UpSampling2D(size=2),
#             ReflectionPadding2D(),
#             Conv2D(256, (3, 3), activation="relu"),
#             ReflectionPadding2D(),
#             Conv2D(256, (3, 3), activation="relu"),
#             ReflectionPadding2D(),
#             Conv2D(256, (3, 3), activation="relu"),
#             ReflectionPadding2D(),
#             Conv2D(128, (3, 3), activation="relu"),
#             UpSampling2D(size=2),
#             ReflectionPadding2D(),
#             Conv2D(128, (3, 3), activation="relu"),
#             ReflectionPadding2D(),
#             Conv2D(64, (3, 3), activation="relu"),
#             UpSampling2D(size=2),
#             ReflectionPadding2D(),
#             Conv2D(64, (3, 3), activation="relu"),
#             ReflectionPadding2D(),
#             Conv2D(3, (3, 3)),
#         ]
#     )

def decoder():
        return tf.keras.Sequential([
            Conv2DTranspose(512, (3, 3), activation="relu", padding='same'),
            UpSampling2D(size = (2,2)),
            Conv2DTranspose(256, (3, 3), activation="relu", padding='same'),
            Conv2DTranspose(256, (3, 3), activation="relu", padding='same'),
            Conv2DTranspose(256, (3, 3), activation="relu", padding='same'),
            Conv2DTranspose(256, (3, 3), activation="relu", padding='same'),
            UpSampling2D(size = (2,2)),
            Conv2DTranspose(128, (3, 3), activation="relu", padding='same'),
            Conv2DTranspose(128, (3, 3), activation="relu", padding='same'),
            UpSampling2D(size = (2,2)),
            Conv2DTranspose(64, (3, 3), activation="relu", padding='same'),
            Conv2DTranspose(64, (3, 3), activation="relu", padding='same'),
            Conv2DTranspose(3, (1, 1))
        ])
        # # print(t.shape)

        # y = Conv2DTranspose(512, (3, 3), activation="relu", padding='same')(t)
        # # print(y.shape)

        # y = UpSampling2D(size = (2,2))(y)
        # # print(y.shape)
        # y = Conv2DTranspose(256, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)
        # y = Conv2DTranspose(256, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)
        # y = Conv2DTranspose(256, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)
        # y = Conv2DTranspose(256, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)

        # y = UpSampling2D(size = (2,2))(y)
        # # print(y.shape)
        # y = Conv2DTranspose(128, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)
        # # y = UpSampling2D(size = (2,2))(y)
        # y = Conv2DTranspose(128, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)

        # y = UpSampling2D(size = (2,2))(y)
        # # print(y.shape)
        # y = Conv2DTranspose(64, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)
        # # y = UpSampling2D(size = (2,2))(y)
        # y = Conv2DTranspose(64, (3, 3), activation="relu", padding='same')(y)
        # # print(y.shape)

        # y = Conv2DTranspose(3, (1, 1))(y)
        # # print(y.shape)
        # return y




