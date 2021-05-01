import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformer import TransferNet
from utils import load_img

IN_METHOD = 0      # 0: IN; 1: CIN; 2: ADAIN

model_dir = ["./models/single/", "./models/multiple/", "./models/arbitrary/"]
content_paths = ["./fruits-360/Training/Apple Braeburn/r_26_100.jpg", "../content1.jpeg"]
style_paths = ["./style_gallery/Abstract_image_119.jpg", "../style2.jpg"]
content_image = load_img(content_paths[0])
style_image = load_img(style_paths[0])

content_layer = "block4_conv1"
transformer = TransferNet(content_layer, IN_METHOD)
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt.restore(tf.train.latest_checkpoint(model_dir[IN_METHOD])).expect_partial()
print(content_image.shape)
print(style_image.shape)
styled_image = transformer(content_image, style_image, IN_METHOD, alpha=1.0)
print(styled_image.shape)
styled_image = tf.cast(
    tf.squeeze(styled_image), tf.uint8
).numpy()
print(styled_image.shape)

plt.subplot(2, 2, 1)
content = mpimg.imread(content_paths[0])
plt.imshow(content)
plt.title("Content")

plt.subplot(2, 2, 2)
style = mpimg.imread(style_paths[0])
plt.imshow(style)
plt.title("Style")

img = Image.fromarray(styled_image, mode="RGB")
img.save("./outputs/test1.jpg")
plt.subplot(2, 2, 3)
plt.imshow(img)
plt.title("Single Feed Forward")
#plt.show()
plt.savefig('./outputs/plot.png')
