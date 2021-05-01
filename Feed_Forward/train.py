import os
from argparse import ArgumentParser
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.run_functions_eagerly(True)

from losses import style_loss, content_loss
from vgg import VGG
from transformer import TransferNet
from utils import load_img, content_pre_proc, resize


AUTOTUNE = tf.data.experimental.AUTOTUNE


IN_METHOD = 0  # 0: Single style; 1: Multiple style; 2: Arbitrary style

#Paths
style_paths = ["./style_gallery/Abstract_image_119.jpg", "../style2.jpg"]

#Layers
content_layer = "block4_conv1"
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1'
]

image_size = 256

# fruitpath = "./fruits-360/test-multiple_fruits/"
fruitpath = "./fruits-360/Training/Apple Braeburn"

sty_img = tf.concat([load_img(style_paths[0])], axis = 0)

style_gallery = "./style_gallery/"
model_dir = ["./models/single/", "./models/multiple/", "./models/arbitrary/"]
vgg = VGG(content_layer, style_layers)
transformer = TransferNet(content_layer, IN_METHOD)



ds_fruit = (
    tf.data.Dataset.list_files(os.path.join(fruitpath, "*.jpg"))
    .map(content_pre_proc, num_parallel_calls= AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .repeat()
    .batch(1)
    .prefetch(AUTOTUNE)
)


ds_style_gallery = (
    tf.data.Dataset.list_files(os.path.join(style_gallery, "Abstract_image_119.jpg"))
    .map(content_pre_proc, num_parallel_calls= AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .repeat()
    .batch(1)
    .prefetch(AUTOTUNE)
)

# ds_coco = (
#     tfds.load("coco/2014", split="train")
#     .repeat()
#     .batch(1)
#     .prefetch(AUTOTUNE)
# )

ds = tf.data.Dataset.zip((ds_fruit, ds_style_gallery))



optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4)
checkpt = tf.train.Checkpoint(optimizer = optimizer, transformer = transformer)
manager = tf.train.CheckpointManager(checkpt, model_dir[IN_METHOD], max_to_keep = 1)   #to store the checkpoint in IN_METHOD specific directory
checkpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print( "Restored from {}".format(manager.latest_checkpoint) )
else:
    print("From scratch.")




avg_train_loss = tf.keras.metrics.Mean(name = "avg_train_loss")
avg_train_style_loss = tf.keras.metrics.Mean(name = "avg_train_style_loss")
avg_train_content_loss = tf.keras.metrics.Mean(name = "avg_train_content_loss")




@tf.function
def train_step_IN(content_image, style_feature_map):
    # trans = transformer.encode_IN(content_image, alpha = 1.0)

    with tf.GradientTape() as tape:
        #TEST
        trans = transformer.encode_IN(content_image, alpha = 1.0)
        styled_img = transformer.decode(trans)
        content_feature_styled, style_feature_styled = vgg(styled_img)

        # print("break")
        # print([v.shape for v in style_feature_map])
        # print([v.shape for v in style_feature_styled])
        # print(styled_img.shape)

        total_content_loss = 10 * content_loss(
            trans, content_feature_styled
        )
        total_style_loss = 1 * style_loss(
            style_feature_map, style_feature_styled
        )
        loss = total_content_loss + total_style_loss
        loss = loss

    # print([v.name for v in transformer.trainable_variables])
    gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )
    avg_train_loss(loss)
    avg_train_style_loss(total_style_loss)
    avg_train_content_loss(total_content_loss)

@tf.function
def train_step_ADAIN(content_image, style_image):
    trans = transformer.encode_ADAIN(content_image, style_image, alpha = 1.0)

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
            # print([v for v in transformer.trainable_variables if (v.name.startswith("gamma") or v.name.startswith("beta"))])

elif IN_METHOD == 1:             # when we have a multiple style transfer,
    print ("begin multiple style transfer")


else:
    print ("begin arbitrary style transfer")
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
