import tensorflow as tf



def mean_standard_loss(feature, feature_styled, epsilon = 1e-5):
    featured_mean, featured_variance = tf.nn.moments(feature, axes = [1,2])
    featured_styled_mean, featured_styled_variance = tf.nn.moments(
        feature_styled, axes = [1,2]
    )
    featured_std = tf.math.sqrt(featured_variance + epsilon)
    featured_styled_std = tf.math.sqrt(featured_styled_variance + epsilon)
    loss = tf.losses.mse(featured_styled_mean, featured_mean) + tf.losses.mse (
        featured_styled_std, featured_std
    )
    return loss
def style_loss(feature, feature_styled):
    return tf.reduce_sum(
        [
            mean_standard_loss(f, f_styled)
            for f, f_styled in zip(feature, feature_styled)
        ]
    )

def content_loss(feature, feature_styled):
    return tf.reduce_mean(tf.square(feature - feature_styled), axis = [1,2,3])