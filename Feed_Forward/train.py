import os
from argparse import ArgumentParser
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
from losses import style_loss, content_loss

from vgg import VGG
from transformer import TransferNet

AUTOTUNE = tf.data.experimental.AUTOTUNE


IN_METHOD = 0  # 0: Single style; 1: Multiple style; 2: Arbitrary style

#Paths
content_paths = ["../content1.jpeg"]
style_paths = ["../style2.jpg"]

def load_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

#Layers
content_layer = "block4_conv1"
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
]

image_size = 256

fruitpath = "./fruits-360/test-multiple_fruits/"
content_images = tf.concat([load_img(content_paths[0])], axis = 0)
sty_img = tf.concat([load_img(style_paths[0])], axis = 0)

model_dir = "./models/"
vgg = VGG(content_layer, style_layers)
# Extracte the style features form style image
# _, style_feature_map = vgg(sty_img)
transformer = TransferNet(content_layer)


def resize(img, min_size):
    width, height, _ = tf.unstack(tf.shape(img), num=3)
    if height < width:
        new_height = min_size
        new_width = int (width * new_height / height)
    else:
        new_width = min_size
        new_height = int (height * new_width / width)
    img = tf.image.resize(img, size = (new_width, new_height))
    return img

def content_pre_proc(file_path, min_size=image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = resize(img, min_size = min_size)
    img = tf.image.random_crop(
        img,
        size = (image_size, image_size, 3)
    )
    img = tf.cast(img, tf.float32)
    return img



ds_fruit = (
    tf.data.Dataset.list_files(os.path.join(fruitpath, "*.jpg"))
    .map(content_pre_proc, num_parallel_calls= AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .repeat()
    .batch(8)
    .prefetch(AUTOTUNE)
)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
checkpt = tf.train.Checkpoint(optimizer = optimizer, transformer = transformer)
manager = tf.train.CheckpointManager(checkpt, model_dir, max_to_keep = 1)
checkpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print( "Restored from {}".format(manager.latest_checkpoint) )
else:
    print("From scratch.")


avg_train_loss = tf.keras.metrics.Mean(name = "avg_train_loss")
avg_train_style_loss = tf.keras.metrics.Mean(name = "avg_train_style_loss")
avg_train_content_loss = tf.keras.metrics.Mean(name = "avg_train_content_loss")


@tf.function
def train_step(content_image, style_image):
    trans = transformer.encode(content_image, style_image, alpha = 1.0)

    with tf.GradientTape() as tape:
        styled_img = transformer.decode(trans)

        _, style_feature_map = vgg(style_image)
        content_feature_styled, style_feature_styled = vgg(styled_img)

        total_content_loss = 10 * content_loss(
            trans, content_feature_styled
        )
        total_style_loss = 1 * style_loss(
            style_feature_map, style_feature_styled
        )
        loss = total_content_loss + total_style_loss

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )
    avg_train_loss(loss)
    avg_train_style_loss(total_style_loss)
    avg_train_content_loss(total_content_loss)


@tf.function
def train_step_IN(content_image, style_feature_map):
    #trans = transformer.encode_IN(content_image, alpha = 1.0)
    
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        trans = transformer.encode_IN(content_image, alpha = 1.0)
        
        styled_img = transformer.decode(trans)


        content_feature_styled, style_feature_styled = vgg(styled_img)

        total_content_loss = 10 * content_loss(
            trans, content_feature_styled
        )
        total_style_loss = 1 * style_loss(
            style_feature_map, style_feature_styled
        )
        loss = total_content_loss + total_style_loss
    
    gradients = tape.gradient(loss, transformer.trainable_variables)
    
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )
    avg_train_loss(loss)
    avg_train_style_loss(total_style_loss)
    avg_train_content_loss(total_content_loss)


if IN_METHOD == 0 :                  # when we have a single style transfer, 
    print ("begin single style transfer")
    _, style_feature_map_single = vgg(sty_img)
    for step, content_images in tqdm(enumerate(ds_fruit)):
        train_step_IN(content_images, style_feature_map_single)

        if step % 10 == 0:
            print(
                f"Step {step}, "
                f"Loss: {avg_train_loss.result()}, "
                f"Style Loss: {avg_train_style_loss.result()}, "
                f"Content Loss: {avg_train_content_loss.result()}"
            )
            print(f"Saved checkpoint: {manager.save()}")
            avg_train_loss.reset_states()
            avg_train_style_loss.reset_states()
            avg_train_content_loss.reset_states()

elif IN_METHOD == 1:             # when we have a multiple style transfer,
    print ("begin multiple style transfer")
    for step, content_images in tqdm(enumerate(ds_fruit)):
        train_step(content_images, sty_img)
        if step % 10 == 0:
            print(
                f"Step {step}, "
                f"Loss: {avg_train_loss.result()}, "
                f"Style Loss: {avg_train_style_loss.result()}, "
                f"Content Loss: {avg_train_content_loss.result()}"
            )
            print(f"Saved checkpoint: {manager.save()}")
            avg_train_loss.reset_states()
            avg_train_style_loss.reset_states()
            avg_train_content_loss.reset_states()

else:
    print ("begin arbitrary style transfer")