import tensorflow as tf

def gram_matrix(input_tensor):
  # Using einsum for gram matrix:
  # k = 1 this never changes
  # i, j are the postion in c/d filter
  result = tf.linalg.einsum('kijc,kijd->kcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_positions = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_positions)

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

def style_loss(feature, feature_styled):
    # print([gram_matrix(v).shape for v in feature])
    # print([gram_matrix(v).shape for v in feature_styled])
    return tf.reduce_sum(
        [
            tf.norm(gram_matrix(f)-gram_matrix(f_styled), ord='fro', axis=(1,2))
            for f, f_styled in zip(feature, feature_styled)
        ]
    )

# def style_loss(feature, feature_styled):
#     return tf.reduce_sum(
#         [
#             mean_standard_loss(f, f_styled)
#             for f, f_styled in zip(feature, feature_styled)
#         ]
#     )






# def entropy_loss(feature, feature_styled):
#     bce = tf.keras.losses.BinaryCrossentropy()
#     return bce(feature, feature_styled)

# def style_loss(feature, feature_styled):
#     return tf.reduce_sum(
#         [
#             entropy_loss(f, f_styled)
#             for f, f_styled in zip(feature, feature_styled)
#         ]
#     )

# def content_loss(feature, feature_styled):
#     return entropy_loss(feature, feature_styled)
