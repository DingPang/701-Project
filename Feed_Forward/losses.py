import tensorflow as tf

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('kijc,kijd->kcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_positions = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_positions)
    # batch_size , height, width, filters = input_tensor.shape
    # features = tf.reshape(input_tensor, (batch_size, height*width, filters))

    # tran_f = tf.transpose(features, perm=[0,2,1])
    # gram = tf.matmul(tran_f, features)
    # gram /= tf.cast(height*width, tf.float32)

    # return gram

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


def content_loss(feature, feature_styled):
    return tf.reduce_mean(tf.square(feature - feature_styled), axis = [1,2,3])


def style_loss_arb(feature, feature_styled):
    style_loss = tf.reduce_sum(
        [
            mean_standard_loss(f, f_styled)
            for f, f_styled in zip(feature, feature_styled)
        ]
    )
    return style_loss

def style_loss(feature, feature_styled):
    style_loss = tf.add_n(
        [
            tf.reduce_mean((gram_matrix(f)-gram_matrix(f_styled))**2)
            for f, f_styled in zip(feature, feature_styled)
        ]
    )

    return style_loss / len(feature_styled)




