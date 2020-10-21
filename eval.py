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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

from absl import flags
import numpy as np
import six
import tensorflow.compat.v1 as tf
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from tensorflow.contrib import training as contrib_training
from third_party.deeplab import common
from third_party.deeplab.core import feature_extractor
# Custom import
from core import data_generator
from core import model

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

## DeepLab options
# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('eval_crop_size', '513,513',
                  'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

## Pseudo_seg options
flags.DEFINE_boolean('weakly', False, 'Using image-level labeled data or not')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.eval_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.eval_batch_size,
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      num_readers=2,
      is_training=False,
      should_shuffle=False,
      should_repeat=False,
      with_cls=True,
      cls_only=False,
      output_valid=True)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])
    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      raise NotImplementedError('Multi-scale is not supported yet!')

    metric_map = {}
    ## Extract cls logits
    if FLAGS.weakly:
      _, end_points = feature_extractor.extract_features(
          samples[common.IMAGE],
          output_stride=model_options.output_stride,
          multi_grid=model_options.multi_grid,
          model_variant=model_options.model_variant,
          depth_multiplier=model_options.depth_multiplier,
          divisible_by=model_options.divisible_by,
          reuse=tf.AUTO_REUSE,
          is_training=False,
          preprocessed_images_dtype=model_options.preprocessed_images_dtype,
          global_pool=True,
          num_classes=dataset.num_of_classes - 1)
      # ResNet beta version has an additional suffix in FLAGS.model_variant, but
      # it shares the same variable names with original version. Add a special
      # handling here for beta version ResNet.
      logits = end_points['{}/logits'.format(FLAGS.model_variant).replace(
          '_beta', '')]
      logits = tf.reshape(logits, [-1, dataset.num_of_classes - 1])
      cls_pred = tf.sigmoid(logits)

      # Multi-label classification evaluation
      cls_label = samples['cls_label']
      cls_pred = tf.cast(
          tf.greater_equal(cls_pred, 0.5), tf.int32)

      ## For classification
      metric_map['eval/cls_overall'] = tf.metrics.accuracy(
          labels=cls_label, predictions=cls_pred)
      metric_map['eval/cls_precision'] = tf.metrics.precision(
          labels=cls_label, predictions=cls_pred)
      metric_map['eval/cls_recall'] = tf.metrics.recall(
          labels=cls_label, predictions=cls_pred)

    ## For segmentation branch eval
    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    # Define the evaluation metric.
    num_classes = dataset.num_of_classes

    ## For segmentation
    metric_map['eval/%s_overall' % predictions_tag] = tf.metrics.mean_iou(
        labels=labels, predictions=predictions, num_classes=num_classes,
        weights=weights)
    # IoU for each class.
    one_hot_predictions = tf.one_hot(predictions, num_classes)
    one_hot_predictions = tf.reshape(one_hot_predictions, [-1, num_classes])
    one_hot_labels = tf.one_hot(labels, num_classes)
    one_hot_labels = tf.reshape(one_hot_labels, [-1, num_classes])
    for c in range(num_classes):
      predictions_tag_c = '%s_class_%d' % (predictions_tag, c)
      tp, tp_op = tf.metrics.true_positives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      fp, fp_op = tf.metrics.false_positives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      fn, fn_op = tf.metrics.false_negatives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      tp_fp_fn_op = tf.group(tp_op, fp_op, fn_op)
      iou = tf.where(tf.greater(tp + fn, 0.0),
                     tp / (tp + fn + fp),
                     tf.constant(np.NaN))
      metric_map['eval/%s' % predictions_tag_c] = (iou, tp_fp_fn_op)

    (metrics_to_values,
     metrics_to_updates) = contrib_metrics.aggregate_metric_map(metric_map)

    summary_ops = []
    for metric_name, metric_value in six.iteritems(metrics_to_values):
      op = tf.summary.scalar(metric_name, metric_value)
      op = tf.Print(op, [metric_value], metric_name)
      summary_ops.append(op)

    summary_op = tf.summary.merge(summary_ops)
    summary_hook = contrib_training.SummaryAtEndHook(
        log_dir=FLAGS.eval_logdir, summary_op=summary_op)
    hooks = [summary_hook]

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations

    if FLAGS.quantize_delay_step >= 0:
      contrib_quantize.create_eval_graph()

    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer
        .TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
    contrib_training.evaluate_repeatedly(
        checkpoint_dir=FLAGS.checkpoint_dir,
        master=FLAGS.master,
        eval_ops=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        hooks=hooks,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
