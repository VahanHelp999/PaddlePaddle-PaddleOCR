# Training Pipeline: From Data to Deployment

This guide explains the complete training pipeline in PaddleOCR, from data preparation to model deployment.

## 🔄 Complete Training Flow

```
1. Prepare Data → 2. Configure → 3. Train → 4. Evaluate → 5. Export → 6. Deploy
     📁              ⚙️            🏋️          📊           📦          🚀
```

---

## 📁 Stage 1: Data Preparation

### Detection Data Format

#### Directory Structure
```
dataset/
├── train_images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── val_images/
│   ├── val1.jpg
│   └── ...
├── train_list.txt
└── val_list.txt
```

#### Label File Format

**Format 1: JSON (Recommended)**
```
# train_list.txt
train_images/img1.jpg	[{"transcription": "Hello", "points": [[100, 50], [200, 50], [200, 80], [100, 80]]}, {"transcription": "World", "points": [[100, 100], [200, 100], [200, 130], [100, 130]]}]
train_images/img2.jpg	[{"transcription": "Text", "points": [[50, 60], [150, 60], [150, 90], [50, 90]]}]
```

**Format 2: Simple (Legacy)**
```
# Format: image_path\tx1,y1,x2,y2,x3,y3,x4,y4,text
train_images/img1.jpg	100,50,200,50,200,80,100,80,Hello	100,100,200,100,200,130,100,130,World
```

**Key points**:
- Each line = one image
- Each polygon = 4 points (x,y) for corners
- Points in clockwise order
- Tab-separated format

### Recognition Data Format

#### Directory Structure
```
dataset/
├── train_images/
│   ├── word1.jpg  (contains "Hello")
│   ├── word2.jpg  (contains "World")
│   └── ...
├── val_images/
│   ├── val_word1.jpg
│   └── ...
├── train_list.txt
└── val_list.txt
```

#### Label File Format
```
# train_list.txt
# Format: image_path\ttext
train_images/word1.jpg	Hello
train_images/word2.jpg	World
train_images/word3.jpg	PaddleOCR
```

**Key points**:
- One word per image
- Fixed height (e.g., 32 pixels), variable width
- Clear text, properly cropped

### Data Collection Tips

#### 1. **Public Datasets**

**Detection**:
- ICDAR 2015/2017/2019
- MSRA-TD500
- CTW1500 (curved text)
- Total-Text
- COCO-Text

**Recognition**:
- MJSynth (9M synthetic images)
- SynthText
- IIIT5K
- SVT (Street View Text)
- ICDAR 2015 Word

#### 2. **Synthetic Data Generation**

Use `StyleText` for generation:
```bash
cd StyleText
python tools/synth_image.py -c configs/config.yml
```

Benefits:
- Unlimited data
- No labeling cost
- Controllable variations

#### 3. **Data Annotation Tools**

- **PPOCRLabel**: Official PaddleOCR annotation tool
- **LabelImg**: For bounding boxes
- **VGG Image Annotator**: Web-based

---

## ⚙️ Stage 2: Configuration

### Create Config File

```yaml
# configs/det/my_training.yml

Global:
  use_gpu: true
  epoch_num: 500
  save_model_dir: ./output/my_det_model/
  pretrained_model: ./pretrain_models/ch_PP-OCRv4_det_train/best_accuracy

Architecture:
  model_type: det
  algorithm: DB
  Backbone:
    name: MobileNetV3
    scale: 0.5
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

Optimizer:
  name: Adam
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 2

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./my_dataset/
    label_file_list:
      - ./my_dataset/train_list.txt
  loader:
    batch_size_per_card: 16
    num_workers: 8

Eval:
  dataset:
    data_dir: ./my_dataset/
    label_file_list:
      - ./my_dataset/val_list.txt
  loader:
    batch_size_per_card: 1
    num_workers: 2
```

---

## 🏋️ Stage 3: Training

### Single GPU Training

```bash
python tools/train.py -c configs/det/my_training.yml
```

### Multi-GPU Training

```bash
# 4 GPUs
python -m paddle.distributed.launch --gpus '0,1,2,3' \
    tools/train.py -c configs/det/my_training.yml

# All available GPUs
python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7' \
    tools/train.py -c configs/det/my_training.yml
```

### Resume Training from Checkpoint

```bash
python tools/train.py \
    -c configs/det/my_training.yml \
    -o Global.checkpoints=./output/my_det_model/latest
```

### Training Output

```
output/my_det_model/
├── best_accuracy.pdparams         # Best model (by metric)
├── best_accuracy.pdopt            # Optimizer state
├── latest.pdparams                # Latest checkpoint
├── latest.pdopt
├── iter_epoch_1.pdparams          # Periodic checkpoints
├── iter_epoch_2.pdparams
└── train.log                      # Training log
```

---

## 📊 Stage 4: Evaluation

### Evaluate Trained Model

```bash
python tools/eval.py \
    -c configs/det/my_training.yml \
    -o Global.checkpoints=./output/my_det_model/best_accuracy
```

### Evaluation Output

**Detection**:
```
[2023/09/10 12:00:00] Eval: recall:0.85, precision:0.82, hmean:0.83
```

**Metrics**:
- **Recall**: How many actual text regions were found
- **Precision**: How many detected regions are correct
- **Hmean (F1-score)**: Harmonic mean of recall and precision

**Recognition**:
```
[2023/09/10 12:00:00] Eval: acc:0.88, norm_edit_dis:0.92
```

**Metrics**:
- **Accuracy**: % of correctly recognized text
- **Norm Edit Distance**: Normalized edit distance (closer to 1 = better)

---

## 📦 Stage 5: Model Export

### Why Export?

Training models include extra layers (dropout, batch norm stats) not needed for inference. Export creates an optimized inference model.

### Export Command

**Detection**:
```bash
python tools/export_model.py \
    -c configs/det/my_training.yml \
    -o Global.pretrained_model=./output/my_det_model/best_accuracy \
       Global.save_inference_dir=./inference/my_det_model/
```

**Recognition**:
```bash
python tools/export_model.py \
    -c configs/rec/my_rec_training.yml \
    -o Global.pretrained_model=./output/my_rec_model/best_accuracy \
       Global.save_inference_dir=./inference/my_rec_model/
```

### Export Output

```
inference/my_det_model/
├── inference.pdmodel              # Model structure
├── inference.pdiparams            # Model weights
└── inference.pdiparams.info       # Parameter info
```

---

## 🚀 Stage 6: Inference

### Python Inference

**Detection**:
```python
from tools.infer.predict_det import TextDetector

args = {
    'det_model_dir': './inference/my_det_model/',
}
detector = TextDetector(args)

import cv2
img = cv2.imread('test.jpg')
boxes, time = detector(img)

print(f"Detected {len(boxes)} text regions in {time:.2f}s")
```

**Recognition**:
```python
from tools.infer.predict_rec import TextRecognizer

args = {
    'rec_model_dir': './inference/my_rec_model/',
    'rec_char_dict_path': 'ppocr/utils/ppocr_keys_v1.txt',
}
recognizer = TextRecognizer(args)

img_list = [cv2.imread('word1.jpg'), cv2.imread('word2.jpg')]
results, time = recognizer(img_list)

for text, score in results:
    print(f"Text: {text}, Confidence: {score:.2f}")
```

**Complete OCR System**:
```python
from tools.infer.predict_system import TextSystem

args = {
    'det_model_dir': './inference/my_det_model/',
    'rec_model_dir': './inference/my_rec_model/',
}
text_sys = TextSystem(args)

img = cv2.imread('document.jpg')
results = text_sys(img)

for box, (text, score) in results:
    print(f"Box: {box}, Text: {text}, Score: {score:.2f}")
```

---

## 🔧 Training Details

### What Happens During Training?

```python
# Simplified from tools/program.py

def train(config, train_dataloader, model, loss_fn, optimizer, eval_fn):
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # 1. Forward pass
            images = batch['image']
            labels = batch['label']
            preds = model(images)

            # 2. Compute loss
            loss = loss_fn(preds, labels)

            # 3. Backward pass
            loss.backward()

            # 4. Update weights
            optimizer.step()
            optimizer.clear_grad()

            # 5. Log metrics
            if step % log_interval == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        # 6. Evaluate
        if epoch % eval_interval == 0:
            metrics = eval_fn(model, eval_dataloader)
            print(f"Epoch {epoch}, Eval: {metrics}")

        # 7. Save checkpoint
        if epoch % save_interval == 0:
            save_model(model, optimizer, epoch)
```

### Training Monitoring

#### 1. **Console Output**
```
[2023/09/10 12:00:00] Epoch: 1/500, Step: 100/5000, loss: 2.543, lr: 0.00095
[2023/09/10 12:01:00] Epoch: 1/500, Step: 200/5000, loss: 2.234, lr: 0.00096
```

#### 2. **VisualDL (TensorBoard-like)**
```yaml
# Enable in config
Global:
  use_visualdl: True
```

```bash
# Start VisualDL
visualdl --logdir ./output/my_det_model/ --port 8040

# Open browser: http://localhost:8040
```

View:
- Loss curves
- Learning rate schedule
- Evaluation metrics
- Sample predictions

#### 3. **Weights & Biases (W&B)**
```python
# Add to ppocr/utils/loggers/wandb_logger.py
import wandb
wandb.init(project="my-ocr-project")
```

---

## 🎯 Training Tips & Best Practices

### 1. **Use Pre-trained Weights**

**Always start with pre-trained weights!**

```yaml
Global:
  pretrained_model: ./pretrain_models/ch_PP-OCRv4_det_train/best_accuracy
```

**Why?**
- Converges faster (10-50 epochs vs 500+)
- Better final accuracy
- Requires less data

### 2. **Learning Rate Tuning**

**Rule of thumb**:
- Training from scratch: `0.001`
- Fine-tuning: `0.0001` (10x lower)
- Large batch size: Scale up LR proportionally

**Find optimal LR**:
```python
# LR range test (manual)
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
for lr in lrs:
    train_for_epochs(10, lr=lr)
    # Check which gives lowest loss
```

### 3. **Batch Size Selection**

| GPU Memory | Batch Size (Detection) | Batch Size (Recognition) |
|------------|------------------------|--------------------------|
| 8 GB | 4-8 | 64-128 |
| 16 GB | 16-32 | 128-256 |
| 24 GB | 32-64 | 256-512 |

**If OOM (Out of Memory)**:
- Reduce batch size
- Reduce image size
- Use smaller model (scale=0.5 instead of 1.0)
- Enable gradient accumulation

### 4. **Data Augmentation**

**For Detection**:
```yaml
Train:
  dataset:
    transforms:
      - IaaAugment:
          augmenter_args:
            - { 'type': Fliplr, 'args': { 'p': 0.5 } }          # Horizontal flip
            - { 'type': Affine, 'args': { 'rotate': [-10, 10] } }  # Rotation
            - { 'type': GaussianBlur, 'args': { 'sigma': [0, 1.5] } }  # Blur
      - RandomCrop:
          size: [640, 640]
```

**For Recognition**:
```yaml
Train:
  dataset:
    transforms:
      - RecAug:           # Recognition augmentation (built-in)
```

**Balance**:
- Too little augmentation → Overfitting
- Too much augmentation → Slow convergence

### 5. **Evaluation Strategy**

**During training**:
```yaml
Global:
  eval_batch_step: [0, 2000]  # Evaluate every 2000 steps
  cal_metric_during_train: True
```

**Benefits**:
- Catch overfitting early
- Track progress
- Select best checkpoint

**Cost**: Slower training (evaluation takes time)

### 6. **When to Stop Training?**

**Good indicators**:
- Validation metric stops improving for 50+ epochs
- Training loss near zero but validation metric plateaus (overfitting)
- Target accuracy reached

**Bad indicators**:
- Stopping too early (check if still improving)
- Using only training loss (ignores generalization)

### 7. **Handling Overfitting**

**Symptoms**:
- Training loss decreases
- Validation loss increases or plateaus

**Solutions**:
1. More training data
2. Stronger data augmentation
3. Weight regularization (L2)
4. Smaller model
5. Early stopping

```yaml
Optimizer:
  regularizer:
    name: L2
    factor: 0.00001        # Increase to 0.0001 for stronger regularization
```

### 8. **Handling Underfitting**

**Symptoms**:
- Both training and validation loss high
- Accuracy below expectations

**Solutions**:
1. Larger model (scale=1.0 instead of 0.5)
2. More training epochs
3. Higher learning rate
4. Check data quality (labels correct?)
5. Reduce regularization

---

## 🛠️ Advanced Training Techniques

### 1. Knowledge Distillation

Train a small model (student) from a large model (teacher):

```yaml
Architecture:
  model_type: distillation
  algorithm: Distillation
  Models:
    Teacher:
      pretrained: ./pretrain_models/large_model
      freeze_params: true
    Student:
      # Student model config
```

**Benefits**:
- Small model with large model accuracy
- Faster inference

### 2. Mixed Precision Training (FP16)

```yaml
Global:
  use_amp: true          # Automatic Mixed Precision
```

**Benefits**:
- 2x faster training
- Reduced memory usage

**Requirements**: GPU with Tensor Cores (V100, A100, etc.)

### 3. Gradient Accumulation

For large batch sizes with limited GPU memory:

```yaml
Global:
  accumulation_steps: 4  # Accumulate gradients for 4 steps
```

Effective batch size = `batch_size_per_card × accumulation_steps`

### 4. Multi-Scale Training (Detection)

Train with different image scales:

```yaml
Train:
  dataset:
    transforms:
      - RandomScale:
          scale_range: [0.5, 2.0]  # Random scale between 0.5x and 2.0x
```

**Benefits**: Better generalization to different text sizes

---

## 📈 Monitoring Training Progress

### Key Metrics to Watch

**Detection**:
- **Loss**: Should decrease smoothly
- **Hmean**: Should increase (target: >0.85)
- **Precision/Recall**: Balance (not one >> other)

**Recognition**:
- **Loss**: Should decrease
- **Accuracy**: Should increase (target: >0.85)
- **Norm Edit Distance**: Should approach 1.0

### Healthy Training Curves

```
Loss:         |           Metric:
             \|                 /
              \               /
               \            /
                \         /
                 \_____/           ← Plateau = converged
Epochs →                  Epochs →
```

### Unhealthy Training Curves

```
Overfitting:      Training loss ↓, Val loss ↑
Underfitting:     Both losses high
Divergence:       Loss increases (lower LR!)
Oscillation:      Loss jumps up/down (lower LR!)
```

---

## 🔍 Debugging Failed Training

### Problem 1: Loss is NaN

**Causes**:
- Learning rate too high
- Gradient explosion

**Solutions**:
```yaml
Optimizer:
  lr:
    learning_rate: 0.0001  # Lower LR
  clip_norm: 10.0          # Gradient clipping
```

### Problem 2: Loss Not Decreasing

**Causes**:
- Learning rate too low
- Data issue (wrong labels?)
- Model issue (check config)

**Solutions**:
- Increase LR
- Verify data (visualize a few samples)
- Check if model loads pre-trained weights

### Problem 3: Out of Memory (OOM)

**Solutions**:
```yaml
Train:
  loader:
    batch_size_per_card: 4  # Reduce batch size

# Or reduce image size
Train:
  dataset:
    transforms:
      - RandomCrop:
          size: [480, 480]  # Reduce from [640, 640]
```

### Problem 4: Training Too Slow

**Solutions**:
- Increase `num_workers` (data loading parallelism)
- Use smaller images
- Use smaller model (during prototyping)
- Enable mixed precision (`use_amp: true`)
- Use multiple GPUs

---

## 📝 Training Checklist

Before training:
- [ ] Data prepared and verified
- [ ] Config file created
- [ ] Pre-trained weights downloaded
- [ ] GPU available and tested
- [ ] Enough disk space for checkpoints

During training:
- [ ] Monitor loss curves
- [ ] Check evaluation metrics
- [ ] Watch for overfitting/underfitting
- [ ] Save best checkpoints

After training:
- [ ] Evaluate on test set
- [ ] Export inference model
- [ ] Test inference speed
- [ ] Document experiment results

---

## 🔗 Next Steps

- **Deploy your model**: See [09_Deployment_Options.md](./09_Deployment_Options.md)
- **Optimize model**: Quantization, pruning (see deploy/slim/)
- **Troubleshooting**: See [10_FAQ.md](./10_FAQ.md)

---

**Summary**: Training follows a clear pipeline: prepare data → configure → train → evaluate → export → deploy. Use pre-trained weights, monitor metrics, and iterate based on results.
