# [PseudoSeg: Designing Pseudo Labels for Semantic Segmentation](https://arxiv.org/pdf/2010.09713v1.pdf)

PseudoSeg is a simple consistency training framework for semi-supervised image
semantic segmentation, which has a simple and novel re-design of pseudo-labeling
to generate well-calibrated structured pseudo labels for training with unlabeled
or weakly-labeled data. It is implemented by [Yuliang Zou](https://yuliang.vision/) (research intern) in 2020 Summer.

__This is not an official Google product.__

## Instruction

### Installation

- Use a virtual environment

```bash
virtualenv -p python3 --system-site-packages env
source env/bin/activate
```

- Install packages

```bash
pip install -r requirements.txt
```

### Dataset

Create a `dataset` folder under the ROOT directory, then download the pre-created tfrecords for [voc12](https://filebox.ece.vt.edu/~ylzou/summer2020pseudoseg/pascal_voc_seg.tar) and [coco](https://filebox.ece.vt.edu/~ylzou/summer2020pseudoseg/coco.tar), and extract them in `dataset` folder. You may also want to check the filenames for each split under `data_splits` folder.


### Training

**NOTE:** 
- We train all our models using 16 V100 GPUs.
- The ImageNet pre-trained models can be download [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md#model-details-3).
- For VOC12, `${SPLIT}` can be `2_clean, 4_clean, 8_clean, 16_clean_3` (representing 1/2, 1/4, 1/8, and 1/16 splits), `NUM_ITERATIONS` should be set to 30000.
- For COCO, `${SPLIT}` can be `32_all, 64_all, 128_all, 256_all, 512_all` (representing 1/32, 1/64, 1/128, 1/256, and 1/512 splits), `NUM_ITERATIONS` should be set to 200000.

#### Supervised baseline

```bash
python train_sup.py \
  --logtostderr \
  --train_split="${SPLIT}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --num_clones=16 \
  --train_batch_size=64 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception_65/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"
```

#### PseudoSeg (w/ unlabeled data)

```bash
python train_wss.py \
  --logtostderr \
  --train_split="${SPLIT}" \
  --train_split_cls="train_aug" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --num_clones=16 \
  --train_batch_size=64 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception_65/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"
```

#### PseudoSeg (w/ image-level labeled data)

```bash
python train_wss.py \
  --logtostderr \
  --train_split="${SPLIT}" \
  --train_split_cls="train_aug" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --num_clones=16 \
  --train_batch_size=64 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception_65/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --weakly=true
```


### Evaluation

**NOTE:** `${EVAL_CROP_SIZE}` should be `513,513` for VOC12, `641,641` for COCO.

```bash
python eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size="${EVAL_CROP_SIZE}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --max_number_of_evaluations=1
```

### Visualization

**NOTE:** `${VIS_CROP_SIZE}` should be `513,513` for VOC12, `641,641` for COCO.

```bash
python vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size="${VIS_CROP_SIZE}" \
  --checkpoint_dir="${CKPT}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --also_save_raw_predictions=true
```

### Citation
If you use this work for your research, please cite our paper.

```
@article{zou2020pseudoseg,
  title={PseudoSeg: Designing Pseudo Labels for Semantic Segmentation},
  author={Zou, Yuliang and Zhang, Zizhao and Zhang, Han and Li, Chun-Liang and Bian, Xiao and Huang, Jia-Bin and Pfister, Tomas},
  journal={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
