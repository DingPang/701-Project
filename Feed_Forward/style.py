import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformer import TransferNet
from generator import Net
from utils import load_img

IN_METHOD = 2      # 0: IN; 1: CIN; 2: ADAIN

model_dir = ["./models/single/",
            "./models/multiple/",
            "./models/arbitrary/"]

output_dir = ["./outputs/sff",
            "./outputs/mff",
            "./outputs/aff"]

content_paths = ["./cars/Acura_ILX_2013_28_16_110_15_4_70_55_179_39_FWD_5_4_4dr_aWg.jpg",
                "./fruits-360/Training Copy/r_322_100.jpg",
                "../content1.jpeg"]

style_paths = ["../la_muse.jpeg",
            "../starry_night_full.jpeg",
            "../style3.jpeg"]

content_image = load_img(content_paths[0])

content_layer = "block4_conv1"

transformer = None
if IN_METHOD == 0 or IN_METHOD == 1:
    # single model
    transformer = Net()
else:
    # arbitrary model
    transformer = TransferNet(content_layer, IN_METHOD)

ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt.restore(tf.train.latest_checkpoint(model_dir[IN_METHOD])).expect_partial()

styled_image = None
if IN_METHOD == 0:
    # single model
    styled_image = transformer(content_image, [0,1,0,0])
elif IN_METHOD == 1:
    styled_image = transformer(content_image, [0,0,1,0])
    styled_image1 = transformer(content_image, [0.2,0.3,0.5,0])
    styled_image2 = transformer(content_image, [0,1,0,0])
else:
    # arbitrary model
    style_image = load_img("../style2.jpg")
    content_image = load_img("./fruits-360/test-multiple_fruits/apple_apricot_nectarine_peach_peach(flat)_pomegranate_pear_plum.jpg")
    styled_image = transformer(content_image, style_image, IN_METHOD, alpha=1.0)


styled_image = tf.cast(
    tf.squeeze(styled_image), tf.uint8
).numpy()

if IN_METHOD == 1:
    styled_image1 = tf.cast(
      tf.squeeze(styled_image1), tf.uint8
    ).numpy()
    img = Image.fromarray(styled_image1, mode="RGB")
    plt.subplot(2, 3, 5)
    plt.imshow(img)
    plt.title("with style mixed")

    styled_image2 = tf.cast(
      tf.squeeze(styled_image2), tf.uint8
    ).numpy()
    img = Image.fromarray(styled_image2, mode="RGB")
    plt.subplot(2, 3, 6)
    plt.imshow(img)
    plt.title("with style2")

    style = mpimg.imread(style_paths[1])
    plt.subplot(2, 3, 3)
    plt.imshow(style)
    plt.title("Style 2")

    img = Image.fromarray(styled_image, mode="RGB")
    plt.subplot(2, 3, 4)
    plt.imshow(img)
    plt.title("with style1")
elif IN_METHOD == 0:
    img = Image.fromarray(styled_image, mode="RGB")
    plt.subplot(2, 3, 3)
    plt.imshow(img)
    plt.title("Single FeedForward")
else:
    img = Image.fromarray(styled_image, mode="RGB")
    plt.subplot(2, 3, 3)
    plt.imshow(img)
    plt.title("ARB")


if IN_METHOD == 2:
    plt.subplot(2, 3, 2)
    style = mpimg.imread("../style2.jpg")
    plt.imshow(style)
    plt.title("Style")
    plt.subplot(2, 3, 1)
    content = mpimg.imread("./fruits-360/test-multiple_fruits/apple_apricot_nectarine_peach_peach(flat)_pomegranate_pear_plum.jpg")
    plt.imshow(content)
    plt.title("Content")
else:
    plt.subplot(2, 3, 2)
    style = mpimg.imread(style_paths[2])
    plt.imshow(style)
    plt.title("Style")
    plt.subplot(2, 3, 1)
    content = mpimg.imread(content_paths[0])
    plt.imshow(content)
    plt.title("Content")




plt.savefig(output_dir[IN_METHOD]+'/plot.png')
