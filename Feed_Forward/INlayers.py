import tensorflow as tf
# tf.config.run_functions_eagerly(True)
NUM_STYLES = 4

class IN(tf.keras.layers.Layer):
    def __init__(self):
        super(IN, self).__init__()


    def build(self, input_shape):
        self.beta = tf.Variable(initial_value= tf.zeros(input_shape[3]),  trainable=True, name= 'beta' )
        self.gamma = tf.Variable(initial_value= tf.ones(input_shape[3]), trainable=True, name= 'gamma')


    def call(self, content_feature_map, epsilon=1e-5):
        content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)

        content_feature_map_norm = tf.nn.batch_normalization(
            content_feature_map,
            mean=content_mean,
            variance=content_variance,
            offset= self.beta,
            scale= self.gamma,
            variance_epsilon=epsilon,
        )
        # print("content_feature_map_norm: {}".format(tf.shape(content_feature_map_norm)))
        return content_feature_map_norm

class CIN(tf.keras.layers.Layer):

    def __init__(self ):
        super(CIN, self).__init__()


    def build(self, input_shape):
        var_shape = input_shape[3]
        # self.betas = []
        # self.gammas = []

        for i in range(NUM_STYLES):
            setattr(self, "beta"+str(i), tf.Variable(initial_value= tf.zeros(var_shape), trainable=True, name= 'beta' + str(i)))
            setattr(self, "gamma"+str(i), tf.Variable(initial_value= tf.ones(var_shape), trainable=True, name= 'gamma' + str(i)))



    def call(self, content_feature_map, style_indexs = None, epsilon=1e-5):
        content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)
        betas = [getattr(self, "beta"+str(i)) for i in range(NUM_STYLES)]
        gammas = [getattr(self, "gamma"+str(i)) for i in range(NUM_STYLES)]
        offset = tf.math.add_n([tf.math.scalar_mul(w , beta) for beta , w in zip(betas, style_indexs)])
        scale = tf.math.add_n([tf.math.scalar_mul(w , gamma) for gamma , w in zip(gammas, style_indexs)])
        content_feature_map_norm = tf.nn.batch_normalization(
            content_feature_map,
            mean=content_mean,
            variance=content_variance,
            offset= offset,
            scale= scale,
            variance_epsilon=epsilon,
        )
        # print("content_feature_map_norm: {}".format(tf.shape(content_feature_map_norm)))
        return content_feature_map_norm

class AdaIn(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaIn, self).__init__()

    def call(self, content_feature_map, style_contentfeature_map, epsilon=1e-5):
                # axes = [1, 2] means instancenorm
        content_mean, content_variance = tf.nn.moments(content_feature_map, axes=[1, 2], keepdims=True)
        # print("content_mean: {}".format(tf.shape(content_mean)))
        # print("content_variance: {}".format(tf.shape(content_variance)))

        style_mean, style_variance = tf.nn.moments(style_contentfeature_map, axes=[1, 2], keepdims=True)
        # print("style_mean: {}".format(style_mean))
        # print("style_variance: {}".format(tf.shape(style_variance)))
        # print(style_mean)

        style_std = tf.math.sqrt(style_variance + epsilon)
        # print("style_std: {}".format(style_std))
        # print(style_std)
        content_feature_map_norm = tf.nn.batch_normalization(
            content_feature_map,
            mean=content_mean,
            variance=content_variance,
            offset= style_mean,
            scale= style_std,
            variance_epsilon=epsilon,
        )
        # print("content_feature_map_norm: {}".format(tf.shape(content_feature_map_norm)))

        return content_feature_map_norm


