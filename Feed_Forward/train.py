import os
from argparse import ArgumentParser
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.run_functions_eagerly(True)

from losses import style_loss, content_loss, style_loss_arb
from vgg import VGG
from transformer import TransferNet
from utils import load_img, content_pre_proc
from generator import Net


AUTOTUNE = tf.data.experimental.AUTOTUNE


IN_METHOD = 2 # 0: Single style; 1: Multiple style; 2: Arbitrary style

#Paths
style_paths = ["../style2.jpg", "./style_gallery/Abstract_image_119.jpg"]

#Layers
content_layer = "block4_conv1"
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1'
]

# Image size
image_size = 256

# Training Content Path
# fruitpath = "./fruits-360/test-multiple_fruits/"
# fruitpath = "./fruits-360/Training Copy"
trainpath = "./cars"

# Single Trainning Image
sty_img = tf.concat([load_img(style_paths[0])], axis = 0)

# Multiple Training Image
sty_imgs = [ tf.concat([load_img(i)] , axis = 0)  for i in style_paths]

# Arbitrary Trainning Image Dataset
style_gallery = "./style_gallery/"

# Saved Modle Dirs
model_dir = ["./models/single/", "./models/multiple/", "./models/arbitrary/"]

# pretrained Feature Extractor model (VGG)
vgg = VGG(content_layer, style_layers)

transformer = None
if IN_METHOD == 0 or IN_METHOD == 1:
    # single model or multiple style
    transformer = Net()
else:
    # arbitrary model
    transformer = TransferNet(content_layer, IN_METHOD)


# Content Image Dataset
ds_fruit = (
    #tf.data.Dataset.list_files(os.path.join(fruitpath, "*.jpg"))
    tf.data.Dataset.list_files(os.path.join(trainpath, "*.jpg"))
    .map(content_pre_proc, num_parallel_calls= AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .repeat()
    .batch(1)
    .prefetch(AUTOTUNE)
)

# Style Image Dataset
ds_style_gallery = (
    # tf.data.Dataset.list_files(os.path.join(style_gallery, "Abstract_image_119.jpg"))
    tf.data.Dataset.list_files("../style2.jpg")
    .map(content_pre_proc, num_parallel_calls= AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .repeat()
    .batch(1)
    .prefetch(AUTOTUNE)
)

# Combined Image Dataset
ds = tf.data.Dataset.zip((ds_fruit, ds_style_gallery))

# Setup Adam and checkpt Manager
optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4)

# transformer
checkpt = tf.train.Checkpoint(optimizer = optimizer, transformer = transformer)
manager = tf.train.CheckpointManager(checkpt, model_dir[IN_METHOD], max_to_keep = 1)   #to store the checkpoint in IN_METHOD specific directory
checkpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print( "Restored from {}".format(manager.latest_checkpoint) )
else:
    print("From scratch.")

# Declare losses for easy printing
avg_train_loss = tf.keras.metrics.Mean(name = "avg_train_loss")
avg_train_style_loss = tf.keras.metrics.Mean(name = "avg_train_style_loss")
avg_train_content_loss = tf.keras.metrics.Mean(name = "avg_train_content_loss")



# init single model trainstep function
@tf.function
def train_step_IN(content_image, style_feature_map, style_indexs):
    # trans = transformer.encode_IN(content_image, alpha = 1.0)

    with tf.GradientTape() as tape:
        #TEST
        # styled_img = net(content_image)
        content_feature, _ = vgg(content_image)
        styled_img = transformer(content_image, style_indexs)
        content_feature_styled, style_feature_styled = vgg(styled_img)

        # print("break")
        # print([v.shape for v in style_feature_map])
        # print([v.shape for v in style_feature_styled])
        # print(styled_img.shape)

        total_content_loss = content_loss(
            content_feature, content_feature_styled
        )
        total_style_loss = style_loss(
            style_feature_map, style_feature_styled
        )
        loss = 6e0 * total_content_loss + 2e-3 * total_style_loss

    # print([v.name for v in net.trainable_variables])
    gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )
    avg_train_loss(loss)
    avg_train_style_loss(total_style_loss)
    avg_train_content_loss(total_content_loss)

# Arbitrary Model trainstep
@tf.function
def train_step_ADAIN(content_image, style_image):
    trans = transformer.encode_ADAIN(content_image, style_image, alpha = 1.0)

    with tf.GradientTape() as tape:
        styled_img = transformer.decode(trans)

        _, style_feature_map = vgg(style_image)
        content_feature_styled, style_feature_styled = vgg(styled_img)

        total_content_loss =  content_loss(
            trans, content_feature_styled
        )
        total_style_loss = style_loss_arb(
            style_feature_map, style_feature_styled
        )
        loss = total_content_loss + 1e-3 * total_style_loss

    # gradients = tape.gradient(loss, [v for v in transformer.trainable_variables if not (v.name.startswith("gamma") or v.name.startswith("beta"))] )
    # print([v.name for v in transformer.trainable_variables])
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
    #TEST
    for step, content_images in tqdm(enumerate(ds_fruit)):
        # print(content_images)

        train_step_IN(content_images, style_feature_map_single, [1, 0, 0, 0])
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
            # print([v for v in transformer.trainable_variables if (v.name.startswith("gamma") or v.name.startswith("beta"))])

elif IN_METHOD == 1:             # when we have a multiple style transfer,
    print ("begin multiple style transfer")
    for i in range(len(sty_imgs)):
        _, style_feature_map_single = vgg(sty_imgs[i])
        style_indexs = [0, 0, 0, 0]
        style_indexs[i] = 1
        for step, content_images in tqdm(enumerate(ds_fruit)):
            train_step_IN(content_images, style_feature_map_single, style_indexs)
            if step > 1000:
                manager.save()
                break
            if step % 10 == 0:
                print ("printing the " + str(i) + "th image")
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
    print ("begin arbitrary style transfer") # when we have a multiple style transfer,
    for step, (content_images, style_images) in tqdm(enumerate(ds)):
        train_step_ADAIN(content_images, style_images)
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
