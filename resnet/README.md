Overview

A PyTorch ResNet50 classifier for traffic signs. Data are organized under data/train and data/validation. The notebook includes data augmentation, Dropout to mitigate overfitting, Adam + CosineAnnealingLR, TensorBoard logging, and best-checkpoint saving (by lowest val loss).

Features

ResNet50 with a custom fc head: 2048→1024→Dropout(0.5)→num_classes

Strong augmentations (crop/flip/rotate/color jitter/translate + Normalize)

Adam optimizer with weight decay and cosine LR schedule

TensorBoard logs to logs/

Robust image loading & corrupted image skipping

Best checkpoint saved to parameters_Resnet50.cpt (by min val loss)

Quickstart

Open resnet50_traffic.ipynb and run all cells.

Default: epochs=100, batch_size=16, lr=1e-4.

Use TensorBoard via tensorboard --logdir logs.

Data Layout
data/
  train/<class_name>/*.jpg|png|jpeg|bmp
  validation/<class_name>/*.jpg|png|jpeg|bmp


Make sure class folders match the MyDataset.classes list and the fc output dim.

Evaluate & Inference

Use the last cells of the notebook for overall and per-class accuracy.
To infer: load checkpoint with load_net_from_hdf(...), preprocess to 224, then argmax.

Common Edits

Change classes → edit MyDataset.classes and self.net.fc.

Augmentations → edit T.Compose.

LR/weight decay/scheduler → edit in train().
