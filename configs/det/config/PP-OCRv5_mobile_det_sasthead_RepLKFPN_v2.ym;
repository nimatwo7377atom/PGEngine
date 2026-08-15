Global:
  model_name: PP-OCRv5_mobile_det_sasthead_RepLKFPN_v2
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 100
  save_model_dir: ./output/PP-OCRv5_mobile_det_sasthead_RepLKFPN_v2
  save_epoch_step: 5
  eval_batch_step: [1000, 500]        # نه [0, ...] - جلوگیری از انفجار NMS روی هد تصادفی
  cal_metric_during_train: false
  checkpoints: /content/models/iter_epoch_35
  pretrained_model: /content/models/iter_epoch_35.pdparams
  save_inference_dir: null
  use_visualdl: false
  infer_img: doc/imgs_en/img_10.jpg
  save_res_path: ./checkpoints/det_sast/predicts_sast.txt
  d2s_train_image_shape: [3, 960, 960]
  distributed: false

Architecture:
  model_type: det
  algorithm: SAST
  Transform: null
  Backbone:
    name: PPLCNetV3        # وزن‌ها کامل لود می‌شوند
    scale: 0.75
    det: True
  Neck:
    name: RepLKFPN_Det    # جدید: میدان دریافت 7x7 با هزینه‌ی inference یکسان
    out_channels: 96
    shortcut: True
    dilated_kernel_size: 7 # اگر نسخه‌ی شما این پارامتر را نپذیرفت، این خط را حذف کنید
  Head:
    name: SASTHead         # از صفر (کوچک است)

Loss:
  name: SASTLoss

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001  # نک هم از صفر است؛ اگر loss تا epoch 5 تند پایین نیامد، 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 5.0e-05

PostProcess:
  name: SASTPostProcess
  score_thresh: 0.7
  sample_pts_num: 2
  nms_thresh: 0.5
  expand_scale: 1.0
  shrink_ratio_of_width: 0.4

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /content/my_dataset/
    label_file_list:
      - /content/my_dataset/train_oversampled.txt   # خروجی فایل ۳
    ratio_list: [1.0]
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - IaaAugment:
        augmenter_args:
        - type: Fliplr
          args:
            p: 0.5
        - type: Affine
          args:
            rotate: [-10, 10]
        - type: Resize
          args:
            size: [0.5, 1.5]     # jitter مقیاس دوطرفه (v2 فقط [1.0,1.5] داشت)
    - SmallPlateZoomCrop:
        p: 0.5
        target_short: 64
        image_shape: 960
    - SASTProcessTrain:
        image_shape: [960, 960]
        min_crop_side_ratio: 0.3
        min_crop_size: 24
        min_text_size: 4
        max_text_size: 640
    - KeepKeys:
        keep_keys: ['image', 'score_map', 'border_map', 'training_mask', 'tvo_map', 'tco_map']
  loader:
    shuffle: true
    drop_last: false
    batch_size_per_card: 6
    num_workers: 8

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /content/my_dataset/
    label_file_list:
      - /content/my_dataset/train.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - DetResizeForTest:
        resize_long: 960
    - NormalizeImage:
        scale: 1./255.
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: 'hwc'
    - ToCHWImage: null
    - KeepKeys:
        keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
