import tensorflow as tf
# tf.config.run_functions_eagerly(True)
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose

from INlayers import IN, AdaIn

# This Encoder is extedned on tf.keras.models.Model, and VGG19 Based
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

# Encoder Decoder Structure for Arbitrary NST
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


    # This function is not in use
    def encode_IN(self, content_image, alpha):
        content_feature_map = self.encoder(content_image)

        t = self.INlayer(content_feature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    # This function is not in use
    def encode_CIN(self, content_image, style_image, alpha):
        # content_feature_map = self.encoder(content_image)
        # style_feature_map = self.encoder(style_image)

        # t = instance_normalization(content_feature_map, style_feature_map)
        # t = alpha * t + (1 - alpha) * content_feature_map
        # return t
        return 0

    # Utilize Adain to encode input images
    def encode_ADAIN(self, content_image, style_image, alpha):
        content_feature_map = self.encoder(content_image)
        style_contentfeature_map = self.encoder(style_image)

        t = self.INlayer(content_feature_map, style_contentfeature_map)
        t = alpha * t + (1 - alpha) * content_feature_map
        return t

    # Decode the feature maps
    def decode(self, t):
        return self.decoder(t)

    def call(self, content_image, style_image, IN_METHOD, alpha=1.0):
        if IN_METHOD == 0:
            # Not in use
            t = self.encode_IN(content_image, alpha)
        elif IN_METHOD == 1:
            # Not in use
            t = self.encode_CIN(content_image, style_image, alpha)
        else :
            # For Arbitrary NST
            t = self.encode_ADAIN(content_image, style_image, alpha)
        g_t = self.decode(t)
        return g_t

# Use Sequential to compose layers
# Symmetrical to VGG19 Structure
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





