# PaddleOCR-Only Implementation Guide

This guide focuses on **just the core OCR functionality** (text detection + recognition), skipping advanced features like tables, layout analysis, and key information extraction.

## 🎯 What This Guide Covers

✅ **Included** (Core OCR):
- Text Detection (finding text boxes)
- Text Recognition (reading text)
- Text Direction Classification (optional, for rotated text)
- Basic inference and training

❌ **Excluded** (Advanced features):
- PaddleStructure (tables, layout analysis)
- Key Information Extraction (KIE)
- End-to-end text spotting
- Super-resolution
- Specialized applications

## 📂 Minimal Folder Structure

If you only need basic OCR, you can focus on these folders:

```
PaddleOCR/
├── ppocr/                     # Core library (USE THIS)
│   ├── modeling/
│   │   ├── architectures/
│   │   ├── backbones/
│   │   │   ├── det_*.py      ✓ Detection backbones
│   │   │   └── rec_*.py      ✓ Recognition backbones
│   │   ├── necks/
│   │   │   ├── db_fpn.py     ✓ Detection neck
│   │   │   └── rnn.py        ✓ Recognition neck
│   │   ├── heads/
│   │   │   ├── det_*.py      ✓ Detection heads
│   │   │   └── rec_*.py      ✓ Recognition heads
│   │   └── transforms/        (Optional: TPS)
│   ├── data/                  ✓ Data loading
│   ├── losses/
│   │   ├── det_*.py          ✓ Detection losses
│   │   └── rec_*.py          ✓ Recognition losses
│   ├── metrics/
│   │   ├── det_metric.py     ✓ Detection metrics
│   │   └── rec_metric.py     ✓ Recognition metrics
│   ├── postprocess/
│   │   ├── db_postprocess.py ✓ Detection postprocess
│   │   └── rec_postprocess.py ✓ Recognition postprocess
│   └── optimizer/             ✓ Training optimization
│
├── tools/                     ✓ Training/inference scripts
│   ├── train.py              ✓ Main training
│   ├── eval.py               ✓ Evaluation
│   ├── export_model.py       ✓ Model export
│   ├── infer_det.py          ✓ Detection inference
│   ├── infer_rec.py          ✓ Recognition inference
│   └── infer/                ✓ Predictor classes
│
├── configs/                   ✓ Configuration files
│   ├── det/                  ✓ Detection configs
│   │   ├── PP-OCRv3/
│   │   ├── PP-OCRv4/
│   │   └── PP-OCRv5/
│   ├── rec/                  ✓ Recognition configs
│   │   ├── PP-OCRv3/
│   │   ├── PP-OCRv4/
│   │   └── PP-OCRv5/
│   └── cls/                  ✓ Direction classifier configs
│
├── paddleocr/                 ✓ User API (simple interface)
│
└── deploy/                    ✓ Deployment (optional)
    ├── cpp_infer/
    ├── lite/
    └── hubserving/

❌ SKIP THESE (not needed for basic OCR):
├── ppstructure/              ❌ Tables, layout, KIE
├── benchmark/                ❌ Benchmarking tools
├── test_tipc/                ❌ Testing infrastructure
└── applications/             ❌ Advanced applications
```

---

## 🚀 Quick Start: Using Pre-trained Models

### Option 1: Simple Python API (Easiest)

```python
from paddleocr import PaddleOCR, draw_ocr

# Initialize OCR
ocr = PaddleOCR(
    use_angle_cls=True,  # Use text direction classifier
    lang='en'            # Language: 'en', 'ch', 'fr', etc.
)

# Run OCR on an image
result = ocr.ocr('image.jpg', cls=True)

# Print results
for line in result[0]:
    box = line[0]          # Bounding box coordinates
    text = line[1][0]      # Recognized text
    confidence = line[1][1]  # Confidence score
    print(f"Text: {text}, Confidence: {confidence:.2f}")

# Visualize results
from PIL import Image
image = Image.open('image.jpg')
boxes = [line[0] for line in result[0]]
texts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

im_show = draw_ocr(image, boxes, texts, scores, font_path='path/to/font.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

**That's it!** This gives you a complete OCR pipeline with 3 lines of code.

---

### Option 2: Using Individual Components

If you need more control:

```python
from tools.infer.predict_system import TextSystem

# Initialize detection + recognition system
text_sys = TextSystem(args)

# Args specify:
# - Detection model path
# - Recognition model path
# - Classifier model path (optional)

# Run OCR
result = text_sys(img)
```

---

## 🛠️ Training Your Own Models

### Step 1: Prepare Data

#### Detection Data Format

```
dataset/
├── train_images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── train_labels.txt
```

**Label file format** (`train_labels.txt`):
```
train_images/img1.jpg	[{"transcription": "hello", "points": [[100, 50], [200, 50], [200, 80], [100, 80]]}, {"transcription": "world", "points": [[100, 100], [200, 100], [200, 130], [100, 130]]}]
```

#### Recognition Data Format

```
dataset/
├── train_images/
│   ├── word1.jpg  (contains "hello")
│   ├── word2.jpg  (contains "world")
│   └── ...
└── train_labels.txt
```

**Label file format** (`train_labels.txt`):
```
train_images/word1.jpg	hello
train_images/word2.jpg	world
```

---

### Step 2: Configure Training

#### Detection Config Example

Create/modify `configs/det/my_det_config.yml`:

```yaml
Global:
  use_gpu: true
  epoch_num: 500
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/db_mv3/
  save_epoch_step: 1200
  eval_batch_step: [0, 2000]
  cal_metric_during_train: False
  pretrained_model: ./pretrain_models/MobileNetV3_large_x0_5_pretrained
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_en/img_10.jpg
  save_res_path: ./output/det_db/predicts_db.txt

Architecture:
  model_type: det
  algorithm: DB
  Transform:
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: large
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 0.00001

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/train_labels.txt
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - DetLabelEncode:
      - IaaAugment:
          augmenter_args:
            - { 'type': Fliplr, 'args': { 'p': 0.5 } }
            - { 'type': Affine, 'args': { 'rotate': [-10, 10] } }
            - { 'type': Resize, 'args': { 'size': [0.5, 3] } }
      - EastRandomCropData:
          size: [640, 640]
          max_tries: 50
          keep_ratio: true
      - MakeBorderMap:
          shrink_ratio: 0.4
          thresh_min: 0.3
          thresh_max: 0.7
      - MakeShrinkMap:
          shrink_ratio: 0.4
          min_text_size: 8
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: 'hwc'
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 16
    num_workers: 8

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/val_labels.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - DetLabelEncode:
      - DetResizeForTest:
          limit_side_len: 736
          limit_type: 'min'
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: 'hwc'
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 1
    num_workers: 2
```

#### Recognition Config Example

Create/modify `configs/rec/my_rec_config.yml`:

```yaml
Global:
  use_gpu: true
  epoch_num: 500
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_model/
  save_epoch_step: 3
  eval_batch_step: [0, 2000]
  cal_metric_during_train: True
  pretrained_model:
  checkpoints:
  save_inference_dir:
  use_visualdl: False
  infer_img: doc/imgs_words/en/word_1.png
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  max_text_length: 25
  infer_mode: False
  use_space_char: True
  save_res_path: ./output/rec/predicts_rec.txt

Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
  Backbone:
    name: SVTRNet
    dims: [64, 128, 256]
    depth: [3, 6, 3]
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead

Loss:
  name: CTCLoss

Optimizer:
  name: AdamW
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 0.00001

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/rec_train_labels.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - RecAug:
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: True
    batch_size_per_card: 128
    drop_last: True
    num_workers: 8

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/rec_val_labels.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 128
    num_workers: 4
```

---

### Step 3: Start Training

#### Train Detection Model

```bash
# Single GPU
python tools/train.py -c configs/det/my_det_config.yml

# Multi-GPU (4 GPUs)
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    tools/train.py -c configs/det/my_det_config.yml
```

#### Train Recognition Model

```bash
# Single GPU
python tools/train.py -c configs/rec/my_rec_config.yml

# Multi-GPU
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    tools/train.py -c configs/rec/my_rec_config.yml
```

---

### Step 4: Evaluate Models

```bash
# Evaluate detection
python tools/eval.py -c configs/det/my_det_config.yml \
    -o Global.checkpoints=./output/db_mv3/best_accuracy

# Evaluate recognition
python tools/eval.py -c configs/rec/my_rec_config.yml \
    -o Global.checkpoints=./output/rec_model/best_accuracy
```

---

### Step 5: Export Models for Inference

```bash
# Export detection model
python tools/export_model.py -c configs/det/my_det_config.yml \
    -o Global.pretrained_model=./output/db_mv3/best_accuracy \
       Global.save_inference_dir=./inference/det_db/

# Export recognition model
python tools/export_model.py -c configs/rec/my_rec_config.yml \
    -o Global.pretrained_model=./output/rec_model/best_accuracy \
       Global.save_inference_dir=./inference/rec_svtr/
```

---

## 🔍 Inference with Custom Models

### Using Custom Detection Model

```python
from tools.infer.predict_det import TextDetector

# Initialize detector with custom model
det_args = {
    'det_model_dir': './inference/det_db/',
    'det_limit_side_len': 960,
    'det_limit_type': 'max',
}

detector = TextDetector(det_args)

# Run detection
img = cv2.imread('test.jpg')
dt_boxes, elapse = detector(img)

print(f"Detected {len(dt_boxes)} text boxes")
for box in dt_boxes:
    print(box)
```

### Using Custom Recognition Model

```python
from tools.infer.predict_rec import TextRecognizer

# Initialize recognizer with custom model
rec_args = {
    'rec_model_dir': './inference/rec_svtr/',
    'rec_char_dict_path': 'ppocr/utils/ppocr_keys_v1.txt',
    'rec_image_shape': '3, 32, 320',
}

recognizer = TextRecognizer(rec_args)

# Run recognition on cropped text
img_crop = cv2.imread('word.jpg')
rec_result, elapse = recognizer([img_crop])

print(f"Recognized text: {rec_result[0][0]}")
print(f"Confidence: {rec_result[0][1]}")
```

### Complete OCR Pipeline with Custom Models

```python
from tools.infer.predict_system import TextSystem

# Initialize system with custom models
args = {
    'det_model_dir': './inference/det_db/',
    'rec_model_dir': './inference/rec_svtr/',
    'rec_char_dict_path': 'ppocr/utils/ppocr_keys_v1.txt',
    'use_angle_cls': False,  # Set to True if using direction classifier
}

text_sys = TextSystem(args)

# Run complete OCR
img = cv2.imread('test.jpg')
result = text_sys(img)

for line in result:
    box = line[0]
    text = line[1][0]
    confidence = line[1][1]
    print(f"Box: {box}, Text: {text}, Conf: {confidence:.2f}")
```

---

## 📝 What You Can Safely Ignore

If you only need basic OCR, you can **completely ignore** these parts:

### 1. PaddleStructure Folder
```
ppstructure/  ❌ Skip entirely
├── layout/
├── table/
├── kie/
└── recovery/
```

### 2. Advanced Model Files

**In `ppocr/modeling/backbones/`**:
- `table_*.py` ❌ (table recognition)
- `kie_*.py` ❌ (key information extraction)

**In `ppocr/modeling/heads/`**:
- `table_*.py` ❌
- `kie_*.py` ❌
- `e2e_*.py` ❌ (end-to-end text spotting)

### 3. Advanced Configs
```
configs/
├── table/  ❌
├── kie/    ❌
├── e2e/    ❌
└── sr/     ❌ (super-resolution)
```

### 4. Advanced Inference Scripts
```
tools/
├── infer_table.py       ❌
├── infer_kie*.py        ❌
├── infer_sr.py          ❌
└── infer_vqa*.py        ❌
```

---

## 🎯 Minimal Code Understanding

### Essential Files to Understand

#### For Detection:
1. **Model**: `ppocr/modeling/architectures/base_model.py`
2. **Backbone**: `ppocr/modeling/backbones/det_mobilenet_v3.py`
3. **Neck**: `ppocr/modeling/necks/db_fpn.py`
4. **Head**: `ppocr/modeling/heads/det_db_head.py`
5. **Loss**: `ppocr/losses/det_db_loss.py`
6. **Postprocess**: `ppocr/postprocess/db_postprocess.py`
7. **Training**: `tools/train.py`, `tools/program.py`

#### For Recognition:
1. **Model**: `ppocr/modeling/architectures/base_model.py`
2. **Backbone**: `ppocr/modeling/backbones/rec_svtrnet.py`
3. **Head**: `ppocr/modeling/heads/rec_ctc_head.py`
4. **Loss**: `ppocr/losses/rec_ctc_loss.py`
5. **Postprocess**: `ppocr/postprocess/rec_postprocess.py`
6. **Training**: `tools/train.py`, `tools/program.py`

---

## 🚀 Quick Experiments

### Experiment 1: Try Different Detection Models

```yaml
# configs/det/experiment1.yml
Architecture:
  Backbone:
    name: ResNet  # Change from MobileNetV3
    layers: 50
```

### Experiment 2: Try Different Recognition Models

```yaml
# configs/rec/experiment2.yml
Architecture:
  Backbone:
    name: PPHGNetV2  # Change from SVTRNet for higher accuracy
```

### Experiment 3: Adjust Image Size

```yaml
# For detection
Train:
  dataset:
    transforms:
      - EastRandomCropData:
          size: [960, 960]  # Increase from [640, 640] for better detection

# For recognition
Train:
  dataset:
    transforms:
      - RecResizeImg:
          image_shape: [3, 32, 320]  # Increase width for longer text
```

---

## 💡 Tips for Basic OCR

### 1. Start with Pre-trained Models
Don't train from scratch! Use PP-OCRv4 or v5 pre-trained weights.

### 2. Fine-tune on Your Data
If pre-trained models work poorly on your specific domain:
- Collect 1000-5000 images
- Label them
- Fine-tune for 10-20 epochs

### 3. Data Augmentation is Key
For limited data, use aggressive augmentation:
- Rotation, scaling, perspective
- Color jittering, blur
- Copy-paste augmentation

### 4. Multi-language Support
```python
# For multiple languages, use appropriate character dictionaries
ocr = PaddleOCR(lang='en')   # English
ocr = PaddleOCR(lang='ch')   # Chinese
ocr = PaddleOCR(lang='french')  # French
# ... 80+ languages supported
```

---

## 📊 Summary: OCR-Only Workflow

```
1. Install PaddleOCR
   ↓
2. Try pre-trained models (paddleocr package)
   ↓
3. If accuracy is good → Deploy!
   ↓
4. If accuracy is poor → Collect custom data
   ↓
5. Fine-tune detection model
   ↓
6. Fine-tune recognition model
   ↓
7. Export models
   ↓
8. Integrate into your application
   ↓
9. Deploy to production
```

---

## 🔗 Next Steps

- **Configuration System**: See [07_Configuration_System.md](./07_Configuration_System.md)
- **Training Details**: See [08_Training_Pipeline.md](./08_Training_Pipeline.md)
- **Deployment**: See [09_Deployment_Options.md](./09_Deployment_Options.md)
- **Model Details**: See [06_Models_Explained.md](./06_Models_Explained.md)

---

**Key Takeaway**: For basic OCR, focus on `ppocr/`, `tools/`, `configs/det/`, `configs/rec/`, and the `paddleocr/` API. Ignore everything related to tables, KIE, and document structure!
