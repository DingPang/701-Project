'''
Author: Mark Wu; Ding Pang
This is adapted from tensorflow's official tutorial on NST:
https://www.tensorflow.org/tutorials/generative/style_transfer

'''
import os
from numpy.matrixlib.defmatrix import matrix
import tensorflow as tf
import IPython.display as display
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show TensorFlow warnings and errors
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import time
import PIL.Image
from dataclasses import dataclass
from tqdm import tqdm

# Local Paths for Images
Style_Path= "./style3.jpeg"
Content_Path = "./content1.jpeg"

# A function converts a tensor into image
def tensor_to_image(tensor):
  # Tensor is in range from 0 to 1, we convert it to 0-255 for colors.
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  # dim 3 is RGB
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


#Function to load in an image and limiting the image maximum dimention to 512 pixels.
def load_img(path_to_img):
  max_dim = 512
  # Read and get the image
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  # convert tensor's shape to float32 for easy calculation
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)

  # Find the "max" dimension
  long_dim = max(shape)
  # limit the "max" dimension to actual max_dim
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)

  # resize the image
  img = tf.image.resize(img, new_shape)
  # add one dim in image, so it is in (1, width, length, 3)
  img = img[tf.newaxis, :]
  return img

#Function to display image with title
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_image = load_img(Content_Path)
style_image = load_img(Style_Path)

plt.subplot(2, 2, 1)
imshow(content_image, 'Content Image')
plt.subplot(2, 2, 2)
imshow(style_image, 'Style Image')
# plt.show()

# Put content image in RGB values and pass to vgg19
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
# Resize x into 224 by 224, which is default size for vgg19
x = tf.image.resize(x, (224, 224))

# Load Pre-trained VGG19
# This would require a "good" certificate
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# We will get "features" from those belwo layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
  # Creates a vgg model that returns a list of feature values.
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg.trainable = False # Gatys Modle doesn't train or let the model learn

  outputs = []
  for name in layer_names:
    outputs.append(vgg.get_layer(name).output)

  # Defining a model using Keras API with specifying our outputs
  model = tf.keras.Model([vgg.input], outputs)
  return model


# each feature from one layer is in shape (batchsize=1, pos1, pos2, filters)
def gram_matrix(input_tensor):
  # Using einsum for gram matrix:
  # k = 1 this never changes
  # i, j are the postion in c/d filter
  result = tf.linalg.einsum('kijc,kijd->kcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_positions = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_positions)

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = num_style_layers
    self.vgg.trainable = False

#Function returns the style of style layers(in gram matrix) and content of content layer
  def call(self, content_image):
    "Expects float input in [0,1]"
    content_image = content_image*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(content_image)
    outputs = self.vgg(preprocessed_input)
    style_features = outputs[:self.num_style_layers]
    content_features = outputs[self.num_style_layers:]

    # Store style features in gram matrix form
    style_features_gram = []
    for style_feature in style_features:
      style_features_gram.append(gram_matrix(style_feature))

    # put style features in a store
    style_store = {}
    for name, value in zip(self.style_layers, style_features_gram):
      style_store[name] = value

    # put content features in a store
    content_store = {}
    for name, value in zip(self.content_layers, content_features):
      content_store[name] = value

    return {'content': content_store, 'style': style_store}



extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

# Original style/content features of style/content image
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# put content image as a variable, and we are going to update this variable
# during trainning
image = tf.Variable(content_image)

opt = tf.optimizers.Adam(learning_rate=0.05)

# we can play with the weights to get desirable result.
style_weight= 100
content_weight= 10

def style_content_loss(outputs):
    # Getting features
    style_features = outputs['style']
    content_features = outputs['content']

    # Calculate losses and apply weigths
    # we have more layers for style, so we are going to devide the weight by num_style_layers,
    # which kinda "undo" the add_n
    style_loss = tf.add_n([tf.reduce_mean((style_features[name]-style_targets[name])**2)
                           for name in style_features.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_features[name]-content_targets[name])**2)
                             for name in content_features.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss
    return loss


@tf.function()
def train_step(image):
  # Using Gradient descent to update our image
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  # update the image
  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


train_step(image)
train_step(image)
train_step(image)
final = tensor_to_image(image)
plt.subplot(2, 2, 3)
plt.imshow(final)
plt.title("Final with 3 iterations")

start = time.time()

max_steps = 10

step = 0
print("Running Optimization")
for m in tqdm(range(max_steps)):
  step += 1
  train_step(image)
  display.clear_output(wait=True)
end = time.time()

print("Total time: {:.1f}".format(end-start))
finalOpt = tensor_to_image(image)
plt.subplot(2, 2, 4)
plt.imshow(finalOpt)
plt.title("Final with more iterations")
#plt.show()
plt.savefig('singleOptResult2.png')
