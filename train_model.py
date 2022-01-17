import random
import collections

import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf; print(tf.__version__)
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 1  # Number of the batch size
accum_step = 3  # Gradient accumulation steps
input_height = 120
input_width = 120
input_channel = 4  # Total number of channel, e.g. 4
input_depth = 30  # Total number of slices from each modality, e.g. 30
total_fold = 5
fold = 0
global_seed = 7

modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

AUTO = tf.data.AUTOTUNE


skf = StratifiedKFold(n_splits=total_fold, shuffle=True, random_state=global_seed)

for index, (train_index, val_index) in enumerate(skf.split(X=train_df.index,
                                                           y=train_df.MGMT_value)):
    train_df.loc[val_index, 'fold'] = index

print('Ground Truth Distribution Fold-Wise..')
print(train_df.groupby(['fold', train_df.MGMT_value]).size())


# Generate brain tumour image data from path
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dicom_path, data, is_train=True):
        self.is_train = is_train  # to control training/validation/inference part
        self.data = data
        self.dicom_path = dicom_path
        self.label = self.data['MGMT_value']

    def __len__(self):
        return len(self.data['BraTS21ID'])

    def __getitem__(self, index):
        patient_ids = f"{self.dicom_path}/{str(self.data['BraTS21ID'][index]).zfill(5)}/"

        flair = []
        t1w = []
        t1wce = []
        t2w = []

        # Iterating over each modality
        for m, t in enumerate(modalities):
            t_paths = sorted(
                glob.glob(os.path.join(patient_ids, t, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )

            # Pick input_depth times slices -
            # - from middle range possible
            strt_idx = (len(t_paths) // 2) - (input_depth // 2)
            end_idx = (len(t_paths) // 2) + (input_depth // 2)
            # slicing extracting elements with 1 intervals
            picked_slices = t_paths[strt_idx:end_idx:1]

            # Preprocess picked slices (remove black border + bind together)
            for i in picked_slices:
                # Reading pixel file from dicom file
                image = self.read_dicom_xray(i)

                # Iterate and randomly replace an image that is fully black
                j = 0
                while True:
                    # if it's a black image, try to pick any random slice of non-black
                    # otherwise move on with black image.
                    if image.mean() == 0:
                        image = self.read_dicom_xray(random.choice(t_paths))
                        j += 1
                        if j == 100:
                            break
                    else:
                        break

                # Now, we remove black areas; remove black borders from brain image
                rows = np.where(np.max(image, 0) > 0)[0]
                cols = np.where(np.max(image, 1) > 0)[0]
                if rows.size:
                    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
                else:
                    image = image[:1, :1]

                # Add frames / slices of individual modalities
                if m == 0:
                    # Adding flair
                    flair.append(cv2.resize(image, (input_height, input_width)))
                elif m == 1:
                    # Adding t1w
                    t1w.append(cv2.resize(image, (input_height, input_width)))
                elif m == 2:
                    # Adding t1wce
                    t1wce.append(cv2.resize(image, (input_height, input_width)))
                elif m == 3:
                    # Adding t2w
                    t2w.append(cv2.resize(image, (input_height, input_width)))

        # input_shape: (None, h, w, depth, channel)
        # Resample modality arrays that have less than input depth no. of slices

        # for flair
        while True:
            if len(flair) < input_depth and flair:
                flair.append(cv2.convertScaleAbs(random.choice(flair), alpha=1.2, beta=0))
            else:
                break

        # for t1w
        while True:
            if len(t1w) < input_depth and t1w:
                t1w.append(cv2.convertScaleAbs(random.choice(t1w), alpha=1.1, beta=0))
            else:
                break

        # for t1wce
        while True:
            if len(t1wce) < input_depth and t1wce:
                t1wce.append(cv2.convertScaleAbs(random.choice(t1wce), alpha=1.2, beta=0))
            else:
                break

        # for t2w
        while True:
            if len(t2w) < input_depth and t2w:
                t2w.append(cv2.convertScaleAbs(random.choice(t2w), alpha=1.1, beta=0))
            else:
                break

        return np.array((flair, t1w, t1wce, t2w),
                        dtype="object").T, self.label.iloc[index,]

    # Function to read dicom file
    def read_dicom_xray(self, path):
        data = pydicom.read_file(path).pixel_array
        if data.mean() == 0:
            # If all black, return data and find non-black if possible.
            return data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data


def fold_generator(fold):
    train_labels = train_df[train_df.fold != fold].reset_index(drop=True)
    print(train_labels)
    val_labels = train_df[train_df.fold == fold].reset_index(drop=True)
    return (
        DataGenerator(train_sample_path, train_labels),
        DataGenerator(train_sample_path, val_labels)
    )


# Get fold set
train_gen, val_gen = fold_generator(fold)


train_data = tf.data.Dataset.from_generator(
    lambda: map(tuple, train_gen),
    (tf.float32, tf.float32),
    (
        tf.TensorShape([input_height, input_width, input_depth, input_channel]),
        tf.TensorShape([]),
    ),
)

val_data = tf.data.Dataset.from_generator(
    lambda: map(tuple, val_gen),
    (tf.float32, tf.float32),
    (
        tf.TensorShape([input_height, input_width, input_depth, input_channel]),
        tf.TensorShape([]),
    ),
)


def tf_image_augmentation(image):
    splitted_modalities = tf.split(tf.cast(image, tf.float32), input_channel, axis=-1)

    flair_augment_img = []
    t1w_augment_img = []
    t1wce_augment_img = []
    t2w_augment_img = []

    # Remove last axis so we go from (h, w, input_depth, 1) to (h, w, input_depth)
    splitted_modalities = [tf.squeeze(i, axis=-1) for i in splitted_modalities]

    # iterate over each modality, e.g: flair, t1w, t1wce, t2w
    for j, modality in enumerate(splitted_modalities):
        # now splitting each frame from one modality
        splitted_frames = tf.split(tf.cast(modality, tf.float32), modality.shape[-1], axis=-1)

        # iterate over each frame to conduct same augmentation on each frame
        for i, img in enumerate(splitted_frames):
            # Get deterministic augmentation results of each modality.
            tf.random.set_seed(j)
            np.random.seed(j)

            # In tf.image.stateless_random_* , the seed is a Tensor of shape (2,) whose values are any integers.
            img = tf.image.stateless_random_flip_left_right(img, seed=(j, 2))
            img = tf.image.stateless_random_flip_up_down(img, seed=(j, 2))
            img = tf.image.stateless_random_contrast(img, 0.4, 0.8, seed=(j, 2))
            img = tf.image.stateless_random_brightness(img, 0.3, seed=(j, 2))

            # Some operations require channel == 3
            img = tf.image.stateless_random_saturation(tf.image.grayscale_to_rgb(img),
                                                       0.9, 1.8, seed=(j, 2))
            img = tf.image.stateless_random_hue(img, 0.4, seed=(j, 2))

            # Some operations we don't need channel == 3, just 1 is enough
            img = tf.image.rgb_to_grayscale(img)
            img = tf.cast(
                tf.image.stateless_random_jpeg_quality(
                    tf.cast(img, tf.uint8),
                    min_jpeg_quality=20, max_jpeg_quality=40, seed=(j, 2)
                ), tf.float32)

            # Ensuring same augmentation for each modalities
            if tf.random.uniform((), seed=j) > 0.7:
                kimg = np.random.choice([1, 2, 3, 4])
                kgma = np.random.choice([0.7, 0.9, 1.2])

                img = tf.image.rot90(img, k=kimg)  # Random rotation of any 90, 180, 270, 360
                img = tf.image.adjust_gamma(img, gamma=kgma)  # Adjust the gamma
                noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.2,
                                         dtype=tf.float32, seed=j)
                img = img + noise  # additive gaussian noise to image

            # The mask_size should be divisible by 2.
            if tf.random.uniform((), seed=j) > 0.6:
                img = tfa.image.random_cutout(tf.expand_dims(img, 0),
                                              mask_size=(int(input_height * 0.2),
                                                         int(input_width * 0.2)),
                                              constant_values=0)
                img = tf.squeeze(img, axis=0)

            # Clipping. We'll rescale later.
            img = tf.clip_by_value(img, 0, 255)

            # Gathering all frames
            if j == 0:  # 1st modality
                flair_augment_img.append(img)
            elif j == 1:  # 2nd modality
                t1w_augment_img.append(img)
            elif j == 2:  # 3rd modality
                t1wce_augment_img.append(img)
            elif j == 3:  # 4th modality
                t2w_augment_img.append(img)

    image = tf.transpose([flair_augment_img, t1w_augment_img,
                          t1wce_augment_img, t2w_augment_img])
    image = tf.reshape(image, [input_height, input_width, input_depth, input_channel])
    return image


class TFDataGenerator:
    def __init__(self,
                 data,
                 shuffle,
                 aug_lib,
                 batch_size,
                 rescale):
        self.data = data  # data files
        self.shuffle = shuffle  # true for training
        self.aug_lib = aug_lib  # type of augmentation library
        self.batch_size = batch_size  # batch size number
        self.rescale = rescale  # normalize or not

    def get_3D_data(self):
        # augmentation on 3D data set
        if self.aug_lib == 'tf' and self.shuffle:
            self.data = self.data.map(lambda x, y: (tf_image_augmentation(x), y), num_parallel_calls=AUTO)
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle)
        else:
            # true for evaluation and inference, no augmentation
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle)

        # rescaling the data for faster convergence
        if self.rescale:
            self.data = self.data.map(lambda x, y: (layers.Rescaling(scale=1. / 255, offset=0.0)(x), y),
                                      num_parallel_calls=AUTO)

        # prefetching the data
        return self.data.prefetch(-1)



tf_gen = TFDataGenerator(
    train_data,
    shuffle=True,
    aug_lib='tf',
    batch_size=batch_size,
    rescale=True
)

train_generator = tf_gen.get_3D_data()


tf_gen = TFDataGenerator(
    val_data,
    shuffle=False,
    aug_lib=None,
    batch_size=batch_size,
    rescale=True
)

valid_generator = tf_gen.get_3D_data()


class BrainTumorModel3D(keras.Model):
    def __init__(self,
                 model,  # Subclass Model
                 n_gradients=1,  # e.g total_batch_size = batch_size * n_gradients
                 *args, **kwargs):
        super(BrainTumorModel3D, self).__init__(*args, **kwargs)
        self.model = model
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32),
                                                  trainable=False)
                                      for v in self.model.trainable_variables]

    # The training step, forward and backward propagation
    def train_step(self, data):
        # Adding 1 to num_acum_step till n_gradients and start GA
        self.n_acum_step.assign_add(1)
        # Unpack the data
        images, labels = data

        # Open a GradientTape
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer or model.
            # Record operations done by layer onto input on the gradient tape.
            predictions = self.model(images, training=True)
            # Compute the loss value for this minibatch.
            loss = self.compiled_loss(labels, predictions)

        # Compute batch gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Accumulating the batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients),
                self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    # Function for applying Gradient Accum.
    def apply_accu_gradients(self):
        # Apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation,
                                           self.model.trainable_variables))

        # Reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.model.trainable_variables[i], dtype=tf.float32)
            )

    # The test step for evaluation and inference
    def test_step(self, data):
        # Unpack the data
        images, labels = data

        # Run model on inference mode
        predictions = self.model(images, training=False)

        # Compute the loss value for this minibatch.
        loss = self.compiled_loss(labels, predictions)

        # Update metrics
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    # A call funciton needs to be implemented
    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)

    # A custom l2 regularization loss for model to tackle overfit
    def reg_l2_loss(self, weight_decay=1e-5):
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.model.trainable_variables
        ])

from classification_models_3D.tfkeras import Classifiers

# build models
input_tensor = keras.Input((input_height, input_width,
                            input_depth, input_channel), name='input3D')
mapping3feat = keras.layers.Conv3D(3, (3, 3, 3),
                                   strides=(1, 1, 1),
                                   padding='same',
                                   use_bias=True)(input_tensor)

resnet50, _ = Classifiers.get('resnet50')
feat_ext = resnet50(input_shape=(input_height, input_width, input_depth, 3),
                    include_top=False, weights='imagenet')

output = feat_ext(mapping3feat)
output = keras.layers.GlobalAveragePooling3D(keepdims=False)(output)
output = keras.layers.Dense(1, activation='sigmoid')(output)
model = keras.Model(input_tensor, output)
model.summary()

keras.backend.clear_session()
model3D = BrainTumorModel3D(model, n_gradients=batch_size * accum_step)

# compiling
model3D.compile(
    loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.SUM),
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.AUC(), keras.metrics.BinaryAccuracy(name='acc')],
)

class CustomModelCheckpoint(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.val_auc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        current_val_auc = logs.get('val_auc')
        current_val_acc = logs.get('val_acc')
        self.val_loss.append(current_val_loss)
        self.val_auc.append(current_val_auc)
        self.val_acc.append(current_val_acc)

        # save based on lowest validation loss
        if current_val_loss <= min(self.val_loss):
            print('Found lowest val_loss. Saving model weight.')
            self.model.save_weights('model_at_val_loss.h5')

            # save based on highest validation auc
        if current_val_auc >= max(self.val_auc):
            print('Found highest val_auc. Saving model weight.')
            self.model.save_weights('model_at_val_auc.h5')

            # save based on highest validation acc
        if current_val_acc >= max(self.val_acc):
            print('Found highest val_acc. Saving model weight.')
            self.model.save_weights('model_at_val_acc.h5')



# epoch params
epochs = 5

model3D.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[CustomModelCheckpoint(),
               tf.keras.callbacks.CSVLogger('history.csv')],
    verbose=1
)