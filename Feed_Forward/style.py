import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from transformer import TransferNet

IN_METHOD = 0      # 0: IN; 1: CIN; 2: ADAIN

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    return img


def load_img(file_path):
    img = process_path(file_path)
    img = img[tf.newaxis, :]
    return img


content_paths = ["../content1.jpeg"]
style_paths = ["../style2.jpg"]
content_image = load_img(content_paths[0])
style_image = load_img(style_paths[0])

content_layer = "block4_conv1"
transformer = TransferNet(content_layer)
ckpt = tf.train.Checkpoint(transformer=transformer)
# ckpt = tf.train.Checkpoint()
ckpt.restore(tf.train.latest_checkpoint("./models/" )).expect_partial()

styled_image = transformer(content_image, style_image,IN_METHOD, alpha=1.0)
styled_image = tf.cast(
    tf.squeeze(styled_image), tf.uint8
).numpy()

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
