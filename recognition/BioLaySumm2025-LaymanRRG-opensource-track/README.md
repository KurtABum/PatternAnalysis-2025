---
dataset_info:
  features:
  - name: source
    dtype: string
  - name: images_path
    dtype: string
  - name: radiology_report
    dtype: string
  - name: layman_report
    dtype: string
  splits:
  - name: train
    num_bytes: 68578938
    num_examples: 150454
  - name: validation
    num_bytes: 4567677
    num_examples: 10000
  - name: test
    num_bytes: 2526810.5332
    num_examples: 10537
  download_size: 29283249
  dataset_size: 75673425.5332
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
