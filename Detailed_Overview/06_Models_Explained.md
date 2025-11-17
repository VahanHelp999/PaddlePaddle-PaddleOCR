# Models Explained: Understanding Different OCR Models

This guide explains the different models available in PaddleOCR, when to use each one, and how they evolved.

## 📊 Model Categories

PaddleOCR includes models for three main tasks:

```
1. Text Detection → Find where text is located
2. Text Recognition → Read what the text says
3. Text Direction Classification → Determine if text is rotated
```

---

## 🔍 Detection Models

### Overview of Detection Algorithms

| Algorithm | Speed | Accuracy | Characteristics | Use Case |
|-----------|-------|----------|-----------------|----------|
| **DB (Differentiable Binarization)** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | Default in PP-OCR | General purpose |
| **EAST** | ⚡⚡ Medium | ⭐⭐ Fair | Simple | Legacy |
| **PSE** | ⚡ Slow | ⭐⭐⭐ Good | Curved text | Special cases |
| **FCE** | ⚡ Slow | ⭐⭐⭐⭐ Excellent | Arbitrary shapes | Complex scenes |

### 1. DB (Differentiable Binarization) - Recommended ✅

**What it does**: Converts detection into a segmentation problem with adaptive binarization.

**How it works**:
```
Image → CNN → Probability Map + Threshold Map → Binarization → Boxes
```

**Versions in PaddleOCR**:

#### PP-OCRv3 Detection
- **Backbone**: MobileNetV3
- **Neck**: DBFPN (Feature Pyramid Network)
- **Head**: DBHead
- **Size**: ~3 MB (mobile), ~50 MB (server)
- **Config**: `configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_*.yml`

**Characteristics**:
- Fast inference (~50ms CPU, ~10ms GPU)
- Good accuracy on most scenes
- Lightweight (mobile-friendly)

#### PP-OCRv4 Detection
- **Backbone**: PPLCNetV3 (more efficient)
- **Improvements**: Better small text detection
- **Size**: ~4 MB (mobile)
- **Config**: `configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_*.yml`

#### PP-OCRv5 Detection (Latest)
- **Further optimized** for speed and accuracy
- **Config**: `configs/det/ch_PP-OCRv5/`

**When to use DB**:
- ✅ General documents (receipts, forms, books)
- ✅ Natural scenes (street signs, storefronts)
- ✅ Need speed and accuracy balance
- ❌ Extremely curved text (use FCE/PSE instead)

**Code reference**:
- Model: `ppocr/modeling/heads/det_db_head.py`
- Loss: `ppocr/losses/det_db_loss.py`
- Postprocess: `ppocr/postprocess/db_postprocess.py`

---

### 2. EAST (Efficient and Accurate Scene Text)

**What it does**: Directly predicts rotated rectangles for text regions.

**How it works**:
```
Image → CNN → Geometry Map (box parameters) + Score Map → Boxes
```

**When to use**:
- Legacy projects
- Simpler to understand (good for learning)
- Less commonly used now (DB is better)

**Config**: `configs/det/det_r50_vd_east.yml`

---

### 3. PSE (Progressive Scale Expansion)

**What it does**: Detects text using progressive instance segmentation.

**Special feature**: Good for **curved text** and touching text instances.

**When to use**:
- ✅ Curved text (circular signs, product labels)
- ✅ Dense text (overlapping regions)
- ❌ Real-time applications (slower than DB)

**Config**: `configs/det/det_r50_vd_pse.yml`

---

### 4. FCE (Fourier Contour Embedding)

**What it does**: Uses Fourier transform to represent arbitrary-shaped text.

**Special feature**: Best for **irregular and curved text**.

**When to use**:
- ✅ Highly irregular text shapes
- ✅ Artistic text
- ✅ Extreme perspective distortion
- ❌ Real-time applications (very slow)

**Config**: `configs/det/det_r50_vd_fce.yml`

---

### Detection Model Selection Guide

```
Need speed? → PP-OCRv4/v5 DB (MobileNetV3)
Need accuracy? → PP-OCRv4/v5 DB (ResNet50)
Curved text? → PSE or FCE
Irregular shapes? → FCE
Learning/Research? → DB (most popular)
```

---

## 📖 Recognition Models

### Overview of Recognition Algorithms

| Algorithm | Speed | Accuracy | Characteristics | Use Case |
|-----------|-------|----------|-----------------|----------|
| **CRNN (CTC)** | ⚡⚡⚡ Fast | ⭐⭐ Fair | Classic | Legacy |
| **SVTR** | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | Transformer-based | PP-OCRv3 |
| **PPHGNetV2** | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | Latest backbone | PP-OCRv4/v5 |

### 1. CRNN (Convolutional Recurrent Neural Network)

**What it is**: Classic OCR recognition model (2015).

**Architecture**:
```
Image → CNN (features) → RNN (sequence) → CTC (decode) → Text
```

**How it works**:
- CNN extracts visual features
- RNN models left-to-right sequence
- CTC loss handles alignment

**When to use**:
- ✅ Learning OCR (well-documented)
- ✅ Resource-constrained devices
- ✅ Simple English/digit recognition
- ❌ Complex scripts (use SVTR instead)

**Code reference**:
- Backbone: `ppocr/modeling/backbones/rec_resnet_vd.py`
- Neck: `ppocr/modeling/necks/rnn.py`
- Head: `ppocr/modeling/heads/rec_ctc_head.py`

**Config**: `configs/rec/rec_r34_vd_none_bilstm_ctc.yml`

---

### 2. SVTR (Scene Text Recognition with a Single Visual Model)

**What it is**: Modern recognition model using vision transformers (2022).

**Architecture**:
```
Image → SVTR Blocks (attention) → Sequence Encoder → CTC/Attention → Text
```

**Key innovation**:
- No RNN needed
- Pure vision transformer
- Better for multi-language and complex text

**Versions**:

#### SVTR-Tiny
- **Size**: ~6 MB
- **Speed**: Fast
- **Accuracy**: Good
- **Use**: Mobile devices

#### SVTR-Base
- **Size**: ~20 MB
- **Speed**: Medium
- **Accuracy**: Excellent
- **Use**: Server (PP-OCRv3)

#### SVTR-Large
- **Size**: ~40 MB
- **Speed**: Slower
- **Accuracy**: Best
- **Use**: Highest accuracy scenarios

**When to use SVTR**:
- ✅ Multi-language recognition
- ✅ Complex scripts (Chinese, Arabic, etc.)
- ✅ PP-OCRv3 (default backbone)
- ✅ Better than CRNN in most cases

**Code reference**: `ppocr/modeling/backbones/rec_svtrnet.py`

**Config**: `configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml`

---

### 3. PP-OCRv4 Recognition (PPHGNetV2 backbone)

**What's new**: Uses PPHGNetV2 (High-Performance GPU-Friendly Network) as backbone.

**Improvements over PP-OCRv3**:
- Higher accuracy (+2-5%)
- Better small text recognition
- Improved multi-language support

**Models**:

#### Mobile Model
- **Backbone**: PPHGNetV2-B0
- **Size**: ~10 MB
- **Config**: `configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_mobile.yml`

#### Server Model
- **Backbone**: PPHGNetV2-B4/B5
- **Size**: ~80-100 MB
- **Config**: `configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_server.yml`

**When to use PP-OCRv4**:
- ✅ Latest production systems
- ✅ Need best accuracy
- ✅ Multi-language scenarios
- ❌ Extremely resource-constrained (use PP-OCRv3 mobile)

**Code reference**: `ppocr/modeling/backbones/rec_pphgnetv2.py`

---

### 4. Attention-based Recognition

**Alternative to CTC**: Uses attention mechanism instead of CTC for decoding.

**Architecture**:
```
Image → CNN → Encoder → Attention Decoder → Text
```

**Advantages**:
- Better for irregular text
- Can handle 2D layouts
- No independence assumption (CTC assumes characters are independent)

**Disadvantages**:
- Slower than CTC
- More complex to train

**Algorithms**:
- **RARE** (Robust Attention Recognition)
- **SAR** (Show, Attend and Read)
- **NRTR** (No Recurrence Text Recognition)

**When to use**:
- ✅ Irregular text (curved, perspective)
- ✅ Research purposes
- ❌ Production (CTC is simpler and faster)

**Code reference**: `ppocr/modeling/heads/rec_att_head.py`

---

### Recognition Model Selection Guide

```
Need speed + decent accuracy? → PP-OCRv3 Mobile (SVTR-Tiny)
Need best accuracy? → PP-OCRv4 Server (PPHGNetV2)
Multi-language? → PP-OCRv3/v4 (SVTR/PPHGNet)
English only, simple? → CRNN
Learning? → CRNN (simpler to understand)
Research? → Attention-based models
```

---

## 🔄 Text Direction Classification

**Purpose**: Detect if text is upside-down (180° rotation).

**Why needed?**
- Some text images might be incorrectly oriented
- Rotating before recognition improves accuracy

**Model**:
- Simple CNN classifier
- Input: Text crop
- Output: 0° or 180°

**Architecture**:
```
Image → MobileNetV3 → Pooling → FC → [0°, 180°]
```

**When to use**:
- ✅ Mixed document orientations
- ✅ User-uploaded images (unknown orientation)
- ❌ Controlled environments (documents always upright)

**Code reference**: `ppocr/modeling/heads/cls_head.py`

**Config**: `configs/cls/ch_ppocr_mobile_v2.0_cls.yml`

**Usage**:
```python
ocr = PaddleOCR(use_angle_cls=True)  # Enable classifier
result = ocr.ocr('image.jpg', cls=True)
```

---

## 🏆 PP-OCR Evolution (v1 → v5)

### PP-OCRv1 (2020)
**Focus**: Lightweight models for mobile

**Detection**:
- DB + MobileNetV3
- 3 MB mobile model

**Recognition**:
- CRNN + CTC
- 5 MB mobile model

---

### PP-OCRv2 (2021)
**Focus**: Accuracy improvements through distillation

**Key improvements**:
- Knowledge distillation (teacher-student training)
- Better data augmentation
- +3% accuracy

**Detection**: DB + MobileNetV3 (distilled)
**Recognition**: CRNN (distilled)

---

### PP-OCRv3 (2022)
**Focus**: SVTR for recognition, better detection

**Key improvements**:
- SVTR-based recognition (+5% accuracy)
- Improved FPN for detection
- Better multi-language support

**Detection**: DB + MobileNetV3 + Improved FPN
**Recognition**: SVTR + CTC

**Major upgrade**: Recognition accuracy significantly improved

---

### PP-OCRv4 (2023)
**Focus**: Latest backbones and optimizations

**Key improvements**:
- PPHGNetV2 backbone (more accurate)
- PPLCNetV3 for detection (more efficient)
- Better small text handling
- +2-5% accuracy improvement

**Detection**: DB + PPLCNetV3
**Recognition**: PPHGNetV2 + CTC

---

### PP-OCRv5 (2024)
**Focus**: Continued optimization

**Latest state-of-the-art** models with ongoing improvements.

---

## 📏 Model Size Comparison

### Detection Models

| Model | Mobile | Server | Use Case |
|-------|--------|--------|----------|
| PP-OCRv3 Det | 3 MB | 47 MB | Balanced |
| PP-OCRv4 Det | 4 MB | 55 MB | Better accuracy |
| DB-ResNet50 | - | 50 MB | Research |

### Recognition Models

| Model | Mobile | Server | Use Case |
|-------|--------|--------|----------|
| CRNN | 5 MB | 15 MB | Legacy |
| PP-OCRv3 Rec | 12 MB | 25 MB | Good balance |
| PP-OCRv4 Rec | 10 MB | 100 MB | Best accuracy |

### Complete OCR System

| Version | Mobile (Det+Rec) | Server (Det+Rec) |
|---------|------------------|------------------|
| PP-OCRv3 | ~15 MB | ~70 MB |
| PP-OCRv4 | ~14 MB | ~150 MB |

---

## 🎯 Which Model Should You Use?

### For Mobile Apps
```
PP-OCRv3 Mobile
├── Detection: DB + MobileNetV3 (3 MB)
└── Recognition: SVTR-Tiny (12 MB)
Total: ~15 MB

Good balance of size, speed, and accuracy
```

### For Server/Cloud
```
PP-OCRv4 Server
├── Detection: DB + PPLCNetV3 (55 MB)
└── Recognition: PPHGNetV2 (100 MB)
Total: ~150 MB

Best accuracy, acceptable speed on GPU
```

### For Embedded Devices
```
PP-OCRv3 Slim (with quantization)
├── Detection: ~2 MB
└── Recognition: ~4 MB
Total: ~6 MB

Quantized models for resource-constrained devices
```

### For Research/Custom
```
Mix and match:
├── Detection: DB + ResNet50 (customizable)
└── Recognition: SVTR-Large (highest accuracy)

Experiment with different combinations
```

---

## 🔬 Advanced Models (Optional)

### End-to-End Models
**Single model** for detection + recognition together.

**Algorithms**:
- **PGNet** (Progressive Grid Network)
- **ABINet** (Autonomous Bidirectional Inference Network)

**Pros**: Simpler pipeline
**Cons**: Less flexible, usually lower accuracy

**Code**: `ppocr/modeling/heads/e2e_*.py`

---

### Table Recognition Models
For structured table extraction (if you need it).

**Models**:
- **TableMaster**
- **SLANet**

**Location**: `ppocr/modeling/backbones/table_*.py`

---

## 💡 Model Training Tips

### 1. Start with Pre-trained Weights
```yaml
Global:
  pretrained_model: ./pretrain_models/ch_PP-OCRv4_det_train/best_accuracy
```

Never train from scratch - always use pre-trained weights!

### 2. Fine-tune on Your Data
- Use PP-OCRv4 pre-trained weights
- Train on your custom dataset for 10-50 epochs
- Much better than training from scratch

### 3. Knowledge Distillation
Train a small model (student) from a large model (teacher):
```
Teacher (Large, accurate) → Student (Small, fast)
```

**Code**: `ppocr/modeling/architectures/distillation_model.py`

---

## 📊 Performance Benchmarks

### Detection (Average)

| Model | CPU (ms) | GPU (ms) | F1-Score |
|-------|----------|----------|----------|
| PP-OCRv3 Mobile | 50 | 10 | 0.88 |
| PP-OCRv4 Mobile | 45 | 8 | 0.90 |
| PP-OCRv4 Server | 100 | 15 | 0.93 |

### Recognition (per text line)

| Model | CPU (ms) | GPU (ms) | Accuracy |
|-------|----------|----------|----------|
| CRNN | 5 | 1 | 0.75 |
| PP-OCRv3 Mobile | 10 | 2 | 0.82 |
| PP-OCRv4 Server | 20 | 3 | 0.88 |

---

## 🔍 Model File Locations

### Detection
```
ppocr/modeling/
├── backbones/det_*.py       # Detection backbones
├── necks/db_fpn.py          # FPN for detection
├── heads/det_db_head.py     # DB head
└── losses/det_db_loss.py    # DB loss
```

### Recognition
```
ppocr/modeling/
├── backbones/rec_*.py       # Recognition backbones
├── necks/rnn.py             # RNN neck
├── heads/rec_ctc_head.py    # CTC head
└── losses/rec_ctc_loss.py   # CTC loss
```

---

## 📖 Further Reading

- **DB Paper**: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
- **CRNN Paper**: [An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717)
- **SVTR Paper**: [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)

---

**Summary**: For most users, **PP-OCRv4** (detection + recognition) offers the best balance. Use mobile models for apps, server models for cloud deployment. SVTR and PPHGNetV2 are the modern backbones replacing older CRNN-based approaches.
