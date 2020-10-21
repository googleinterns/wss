# Lint as: python2, python3
# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow.compat.v1 as tf
from third_party.deeplab.core import feature_extractor
from core import preprocess_utils


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5
_PROB_OF_JITTER = 1.0


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None,
                               strong_weak=False,
                               output_valid=False):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.
    strong_weak: Generate a pair of (strong, weak) augmented images for
      consistency. Also, need to get the valid region exluding padding.
    output_valid: Output valid region excluding padding or not.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image

  processed_image = tf.cast(image, tf.float32)

  if label is not None:
    label = tf.cast(label, tf.int32)

  # Resize image and label to the desired range.
  if min_resize_value or max_resize_value:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  # Data augmentation by randomly scaling the inputs.
  if is_training:
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)
    processed_image.set_shape([None, None, 3])

  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  if strong_weak:
    # # Color distortion (operates in [0, 1])
    strong = processed_image / 255.
    strong = preprocess_utils.random_color_jitter(
        strong, _PROB_OF_JITTER)
    strong = strong * 255.

  # Pad image with mean pixel value.
  mean_pixel = tf.reshape(
      feature_extractor.mean_pixel(model_variant), [1, 1, 3])
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)
  if strong_weak:
    strong = preprocess_utils.pad_to_bounding_box(
        strong, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, ignore_label)

  # Randomly crop the image and label.
  if is_training and label is not None:
    if not strong_weak:
      processed_image, label = preprocess_utils.random_crop(
          [processed_image, label], crop_height, crop_width)
    else:
      processed_image, label, strong = preprocess_utils.random_crop(
          [processed_image, label, strong], crop_height, crop_width)

  processed_image.set_shape([crop_height, crop_width, 3])

  if label is not None:
    label.set_shape([crop_height, crop_width, 1])

  if not is_training and output_valid:
    # Construct the valid mask excluding the boundary padding
    xs, ys = tf.meshgrid(tf.range(0, crop_width), tf.range(0, crop_height))
    valid_x = tf.cast(tf.less(xs, image_width), tf.int32)
    valid_y = tf.cast(tf.less(ys, image_height), tf.int32)
    valid = tf.reshape(valid_x * valid_y, [crop_height, crop_width, 1])

  if is_training:
    if strong_weak or output_valid:
      # Construct the valid mask excluding the boundary padding
      xs, ys = tf.meshgrid(tf.range(0, crop_width), tf.range(0, crop_height))
      valid_x = tf.cast(tf.less(xs, image_width), tf.int32)
      valid_y = tf.cast(tf.less(ys, image_height), tf.int32)
      valid = tf.reshape(valid_x * valid_y, [crop_height, crop_width, 1])

    if strong_weak:
      # Ignore CutOut region
      strong, label, valid = preprocess_utils.cutout_with_mask(
          strong,
          label,
          pad_size=50,
          mean_pixel=mean_pixel,
          ignore_label=ignore_label,
          valid=valid)

      # Randomly left-right flip the image and label.
      processed_image, strong, valid, label, _ = preprocess_utils.flip_dim(
          [processed_image, strong, valid, label], _PROB_OF_FLIP, dim=1)

      return original_image, processed_image, label, strong, valid

    # Randomly left-right flip the image and label.
    if not output_valid:
      processed_image, label, _ = preprocess_utils.flip_dim(
          [processed_image, label], _PROB_OF_FLIP, dim=1)
    else:
      processed_image, valid, label, _ = preprocess_utils.flip_dim(
          [processed_image, valid, label], _PROB_OF_FLIP, dim=1)

  if not output_valid:
    return original_image, processed_image, label
  else:
    return original_image, processed_image, label, valid
