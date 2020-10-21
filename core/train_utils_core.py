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
"""Utility functions for training."""

import math
import tensorflow as tf
from tensorflow.contrib import slim
# Custom import
from third_party.deeplab.core import preprocess_utils


def _div_maybe_zero(total_loss, num_present):
  """Normalizes the total loss with the number of present pixels."""
  return tf.cast(num_present > 0, tf.float32) * tf.math.divide(
      total_loss, tf.maximum(1e-5, num_present))


net_to_stride_to_endpoints_name = {
    'xception_65': {
        4: 'xception_65/entry_flow/block1',
        8: 'xception_65/entry_flow/block2',
        ## All stride=16 below
        13: 'xception_65/entry_flow/block3',
        14: 'xception_65/middle_flow/block1',
        15: 'xception_65/exit_flow/block1',
        16: 'xception_65/exit_flow/block2',
    },
    'resnet_v1_50': {
        8: 'resnet_v1_50/block1',
        ## All stride=16 below
        14: 'resnet_v1_50/block2',
        15: 'resnet_v1_50/block3',
        16: 'resnet_v1_50/block4',
    },
    'resnet_v1_101': {
        8: 'resnet_v1_101/block1',
        ## All stride=16 below
        14: 'resnet_v1_101/block2',
        15: 'resnet_v1_101/block3',
        16: 'resnet_v1_101/block4',
    },
}


def compute_cam_v2(
    end_points,
    logits,
    cls_label,
    num_class=21,
    use_attention=True,
    attention_dim=128,
    strides=(15, 16),
    is_training=True,
    valid_mask=None,
    net='xception_65',
):
  """Compute Grad-CAM.

  Args:
    end_points: Network end_points (dict).
    logits: Cls logits with shape [N, #classes-1] (multi-label, no bg)
    cls_label: Ground truth image-level label
    num_class: Number of classes including background
    use_attention: Using self-attention to refine or not. If not, then no
    learnable parameters
    attention_dim: Embedding space dimension for key and query used in the
    self-attention module
    strides: Use feature maps from which stride to compute pixel similarity for
    Grad-CAM refinement
    is_training: Indicate training or inference mode
    valid_mask: To identity valid region of the input. It is used to avoid
    attending to padding regions
    net: Specify which network is used

  Returns:
    A list of computed Grad-CAMs or refined ones.
  """
  # Sanity check: Make sure strides are sorted
  strides = sorted(list(strides))[::-1]
  # Always use the last stride layer to compute Grad-CAM
  conv_layer = end_points[net_to_stride_to_endpoints_name[net][strides[0]]]
  cams = []
  # Can we speed up this part?
  for c in range(num_class-1):
    grads = tf.gradients(logits[:, c], conv_layer)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    weighted_grads = weights * conv_layer
    curr_cams = tf.nn.relu(tf.reduce_sum(weighted_grads, axis=3))
    cams.append(curr_cams)
  cams = tf.stack(cams, axis=-1)
  cls_label = tf.reshape(cls_label, [-1, 1, 1, num_class - 1])
  cams = cls_label * cams
  # Normalize to [0, 1]
  cams = _div_maybe_zero(
      cams, tf.reduce_max(cams, axis=(1, 2), keepdims=True))
  out_cam = tf.stop_gradient(cams, name='stride_{}/cam'.format(strides[0]))

  if not use_attention:
    out_att_cam = None
  else:
    valid_mask = tf.compat.v1.image.resize_nearest_neighbor(
        valid_mask, preprocess_utils.resolve_shape(out_cam, 4)[1:3])
    out_att_cam = compute_self_att_v2(
        end_points,
        out_cam,
        num_class,
        attention_dim,
        strides,
        is_training,
        linformer=False,
        valid_mask=valid_mask,
        net=net)

  # Add bg score
  bg = 1 - tf.reduce_max(out_cam, axis=3, keepdims=True)
  out_cam = tf.concat([bg, out_cam], axis=-1)

  return out_cam, out_att_cam


def compute_self_att_v2(
    end_points,
    logits,
    num_class=21,
    attention_dim=128,
    strides=(15, 16),
    is_training=True,
    linformer=True,
    valid_mask=None,
    factor=8,
    downsample_type='nearest',
    net='xception_65'):
  """Compute self-attention for segmentation head.

  Args:
    end_points: Network end_points (dict).
    logits: The input seed for refinement. Used as ``value'' in self-attention.
    Can be either logits, probability, or score map.
    num_class: Number of classes including background
    attention_dim: Embedding space dimension for key and query used in the
    self-attention module
    strides: Use feature maps from which stride to compute pixel similarity
    is_training: Indicate training or inference mode
    linformer: Adopt the idea from https://arxiv.org/abs/2006.04768 to reduce
    memory usage in self-attention computation. But instead of learning the
    downsample function, we use deterministic image downsample functions
    valid_mask: To identity valid region of the input. It is used to avoid
    attending to padding regions
    factor: Downsample factor used in linformer mode
    downsample_type: Use which downsample method to reduce the memory usage. Can
    be either 'nearest' or 'bilinear'. Default: 'nearest'
    net: Specify which network is used

  Returns:
    A list of computed Grad-CAMs or refined ones.
  """
  # Sanity check: Make sure strides are sorted
  strides = sorted(list(strides))[::-1]
  conv_layer_list = []
  for stride in strides:
    conv_layer = end_points[net_to_stride_to_endpoints_name[net][stride]]
    conv_layer_list.append(conv_layer)

  # Resize to seed resolution first
  h, w = preprocess_utils.resolve_shape(logits, 4)[1:3]
  conv_layer_list = [
      tf.compat.v1.image.resize_bilinear(
          conv, (h, w), align_corners=True)
      for conv in conv_layer_list
  ]
  conv_layer_merged = tf.concat(conv_layer_list, axis=-1)
  conv_layer_merged = tf.stop_gradient(conv_layer_merged)
  score = tf.stop_gradient(logits)
  # This tells us what input it is (decoder logits or Grad-CAM)
  value_dim = tf.shape(score)[-1]

  # Only valid when we use Linformer style to reduce size for key and value
  if downsample_type == 'bilinear':
    resize_fn = tf.compat.v1.image.resize_bilinear
  else:
    resize_fn = tf.compat.v1.image.resize_nearest_neighbor

  scope = 'hyper_column'
  with tf.variable_scope(scope):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
                        normalizer_fn=None,
                        biases_initializer=None,
                        reuse=tf.AUTO_REUSE):
      k = slim.conv2d(
          conv_layer_merged, attention_dim, [1, 1], scope='key')
      q = slim.conv2d(
          conv_layer_merged, attention_dim, [1, 1], scope='query')
  q = tf.reshape(q, [-1, h * w, attention_dim])
  if valid_mask is not None:
    valid_mask_q = tf.reshape(valid_mask, [-1, h * w, 1])

  # Adopt idea from Linformer (https://arxiv.org/abs/2006.04768) to reduce the
  # memory usage. Instead of learning a downsample function, we use determinstic
  # image downsample methods (nearest neighbor or bilinear) to reduce the size
  # of key and value.
  if linformer:
    k = resize_fn(
        k, ((h // factor + 1), (w // factor + 1)), align_corners=True)
    k = tf.reshape(k,
                   [-1, (h // factor + 1) * (w // factor + 1), attention_dim])

    if valid_mask is not None:
      valid_mask_k = tf.compat.v1.image.resize_nearest_neighbor(
          valid_mask, ((h // factor + 1), (w // factor + 1)))
      valid_mask_k = tf.reshape(
          tf.cast(valid_mask_k, tf.float32),
          [-1, (h // factor + 1) * (w // factor + 1), 1])
  else:
    k = tf.reshape(k, [-1, h * w, attention_dim])
    valid_mask_k = tf.reshape(valid_mask, [-1, h * w, 1])

  matmul_qk = tf.matmul(q, k, transpose_b=True)
  scaled_att_logits = matmul_qk / math.sqrt(attention_dim)
  # Masking
  if valid_mask is not None:
    final_mask = tf.matmul(valid_mask_q, valid_mask_k, transpose_b=True)
    scaled_att_logits += (1 - final_mask) * -1e9
  att_weights = tf.nn.softmax(scaled_att_logits, axis=-1)

  if linformer:
    value = resize_fn(
        score, ((h // factor + 1), (w // factor + 1)), align_corners=True)
    value = tf.reshape(value,
                       [-1, (h // factor + 1) * (w // factor + 1), value_dim])
  else:
    value = tf.reshape(score, [-1, h * w, value_dim])
  att_score = tf.matmul(att_weights, value)
  att_score = tf.reshape(att_score, tf.shape(score))

  ## Add skip-connection and 1x1 conv to convert score back to logit
  att_score += score
  if value_dim != num_class:
    # Set an initial score for the background class. Since the score range of a
    # class is [0, 2] after skip-connection, we use 2 minus the max class
    # probability to set the initial background score for each pixel.
    bg = 2 - tf.reduce_max(att_score, axis=3, keepdims=True)
    att_score = tf.concat([bg, att_score], axis=-1)

  out_att_logits = slim.conv2d(
      att_score,
      num_class, [1, 1],
      scope='pixel_normalization',
      activation_fn=None,
      normalizer_fn=slim.batch_norm,
      normalizer_params={'is_training': is_training},
      reuse=tf.AUTO_REUSE)

  return out_att_logits


def compute_cam(
    end_points,
    logits,
    cls_label,
    num_class=21,
    use_attention=True,
    attention_dim=128,
    strides=(15, 16),
    is_training=True,
    valid_mask=None,
    net='xception_65'
):
  """Compute Grad-CAM.

  Args:
    end_points: Network end_points (dict).
    logits: Cls logits with shape [N, #classes-1] (multi-label, no bg)
    cls_label: Ground truth image-level label
    num_class: Number of classes including background
    use_attention: Using self-attention to refine or not. If not, then no
    learnable parameters
    attention_dim: Embedding space dimension for key and query used in the
    self-attention module
    strides: Use feature maps from which stride to compute pixel similarity for
    Grad-CAM refinement
    is_training: Indicate training or inference mode (for compatibility only,
    not used in this function)
    valid_mask: To identity valid region of the input. It is used to avoid
    attending to padding regions
    net: Specify which network is used

  Returns:
    A list of computed Grad-CAMs or refined ones.
  """
  # Sanity check: Make sure strides are sorted
  strides = sorted(list(strides))[::-1]
  # Always use the last stride layer to compute Grad-CAM
  conv_layer = end_points[net_to_stride_to_endpoints_name[net][strides[0]]]
  cams = []
  # Can we speed up this part?
  for c in range(num_class-1):
    grads = tf.gradients(logits[:, c], conv_layer)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    weighted_grads = weights * conv_layer
    curr_cams = tf.nn.relu(tf.reduce_sum(weighted_grads, axis=3))
    cams.append(curr_cams)
  cams = tf.stack(cams, axis=-1)
  cls_label = tf.reshape(cls_label, [-1, 1, 1, num_class - 1])
  cams = cls_label * cams
  # Normalize to [0, 1]
  cams = _div_maybe_zero(
      cams, tf.reduce_max(cams, axis=(1, 2), keepdims=True))
  out_cam = tf.stop_gradient(cams, name='stride_{}/cam'.format(strides[0]))

  if not use_attention:
    out_att_cam = None
  else:
    valid_mask = tf.compat.v1.image.resize_nearest_neighbor(
        valid_mask, preprocess_utils.resolve_shape(out_cam, 4)[1:3])

    conv_layer_list = []
    for stride in strides:
      conv_layer = end_points[net_to_stride_to_endpoints_name[net][stride]]
      conv_layer_list.append(conv_layer)

    # Resize to seed resolution first
    h, w = preprocess_utils.resolve_shape(out_cam, 4)[1:3]
    conv_layer_list = [
        tf.compat.v1.image.resize_bilinear(
            conv, (h, w), align_corners=True)
        for conv in conv_layer_list
    ]
    conv_layer_merged = tf.concat(conv_layer_list, axis=-1)
    conv_layer_merged = tf.stop_gradient(conv_layer_merged)
    score = tf.stop_gradient(out_cam)

    scope = 'hyper_column'
    with tf.variable_scope(scope):
      with slim.arg_scope([slim.conv2d],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=None,
                          reuse=tf.AUTO_REUSE):
        k = slim.conv2d(
            conv_layer_merged, attention_dim, [1, 1], scope='key')
        q = slim.conv2d(
            conv_layer_merged, attention_dim, [1, 1], scope='query')
    q = tf.reshape(q, [-1, h * w, attention_dim])
    if valid_mask is not None:
      valid_mask_q = tf.reshape(valid_mask, [-1, h * w, 1])

    k = tf.reshape(k, [-1, h * w, attention_dim])
    valid_mask_k = tf.reshape(valid_mask, [-1, h * w, 1])

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    scaled_att_logits = matmul_qk / math.sqrt(attention_dim)
    # Masking
    if valid_mask is not None:
      final_mask = tf.matmul(valid_mask_q, valid_mask_k, transpose_b=True)
      scaled_att_logits += (1 - final_mask) * -1e9
    att_weights = tf.nn.softmax(scaled_att_logits, axis=-1)

    value = tf.reshape(score, [-1, h * w, num_class-1])
    att_score = tf.matmul(att_weights, value)
    att_score = tf.reshape(att_score, tf.shape(score))

    ## Add 1x1 conv to convert score back to logit
    bg = 1 - tf.reduce_max(att_score, axis=3, keepdims=True)
    att_score = tf.concat([bg, att_score], axis=-1)

    out_att_cam = slim.conv2d(
        att_score,
        num_class, [1, 1],
        scope='pixel_normalization',
        activation_fn=None,
        normalizer_fn=None,
        reuse=tf.AUTO_REUSE)

  return out_cam, out_att_cam


