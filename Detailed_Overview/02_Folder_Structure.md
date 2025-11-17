# Folder Structure Explained

This document explains **every folder** in PaddleOCR, why it exists, and what it contains.

## 📂 Top-Level Directory Structure

```
PaddleOCR/
├── ppocr/              # Core OCR library (models, training, data)
├── tools/              # Training and inference scripts
├── configs/            # Configuration files for all models
├── deploy/             # Deployment solutions
├── paddleocr/          # User-facing Python API/CLI
├── benchmark/          # Performance benchmarking tools
├── ppstructure/        # Document structure analysis
├── doc/                # Documentation (legacy)
├── docs/               # Documentation (main)
├── test_tipc/          # Testing infrastructure
├── tests/              # Unit tests
├── applications/       # Example applications
├── StyleText/          # Style text generation
├── requirements.txt    # Python dependencies
└── setup.py           # Package installation
```

---

## 1. 🧠 `ppocr/` - The Brain of PaddleOCR

**Purpose**: This is the **core library** containing all models, training logic, data processing, and utilities.

### Why it exists?
- Separates core functionality from user interfaces
- Reusable across different entry points (CLI, API, training scripts)
- Clean architecture following software engineering best practices

### Directory Structure

```
ppocr/
├── modeling/          # Model architectures
├── data/              # Data loading and augmentation
├── losses/            # Loss functions
├── metrics/           # Evaluation metrics
├── postprocess/       # Post-processing algorithms
├── optimizer/         # Optimizers and learning rate schedulers
├── utils/             # Utilities (logging, visualization, etc.)
└── ext_op/            # Custom operators
```

---

### 1.1 `ppocr/modeling/` - Model Architectures

**Purpose**: Contains ALL model implementations for detection, recognition, classification, table recognition, and KIE.

#### Structure

```
modeling/
├── architectures/     # Model builders and base classes
│   ├── base_model.py           # BaseModel orchestrates components
│   ├── distillation_model.py  # Knowledge distillation wrapper
│   └── __init__.py             # build_model() factory
│
├── backbones/         # Feature extraction (39 implementations)
│   ├── det_mobilenet_v3.py    # Lightweight detection backbone
│   ├── det_resnet_vd.py       # ResNet for detection
│   ├── rec_svtrnet.py         # SVTR for recognition
│   ├── rec_pphgnetv2.py       # PP-HGNetV2 for recognition
│   ├── table_master_resnet.py # Table recognition backbone
│   └── ... (30+ more)
│
├── necks/             # Feature refinement (15 implementations)
│   ├── db_fpn.py              # Feature Pyramid Network for DB
│   ├── rnn.py                 # RNN for sequence modeling
│   ├── csp_pan.py             # CSP-PAN for detection
│   └── ... (12 more)
│
├── heads/             # Task-specific output heads (37 implementations)
│   ├── det_db_head.py         # DB detection head
│   ├── rec_ctc_head.py        # CTC recognition head
│   ├── rec_att_head.py        # Attention recognition head
│   ├── table_master_head.py   # Table structure head
│   └── ... (33 more)
│
└── transforms/        # Spatial transformations
    └── tps.py                 # Thin Plate Spline transformation
```

#### Why This Structure?

**The Modular Pattern**: `Transform → Backbone → Neck → Head`

1. **Transform** (Optional): Spatial transformation (e.g., straighten curved text)
2. **Backbone**: Extract features from images (CNN-based)
3. **Neck**: Refine and combine features (FPN, RNN, etc.)
4. **Head**: Task-specific predictions (detection boxes, recognized text, etc.)

**Benefits**:
- **Reusability**: One backbone can be used for multiple tasks
- **Experimentation**: Easy to swap components (try different backbones)
- **Clarity**: Each component has a single responsibility

**Example Configuration**:
```yaml
Architecture:
  model_type: det
  algorithm: DB
  Transform: null              # No transformation
  Backbone:
    name: MobileNetV3          # Feature extraction
    scale: 0.5
  Neck:
    name: DBFPN                # Feature refinement
    out_channels: 256
  Head:
    name: DBHead               # Detection output
    k: 50
```

---

### 1.2 `ppocr/data/` - Data Loading & Augmentation

**Purpose**: Load datasets, apply augmentations, and create batches for training.

```
data/
├── __init__.py                # build_dataloader() factory
├── simple_dataset.py          # Standard dataset (reads label files)
├── lmdb_dataset.py            # LMDB format support (faster I/O)
├── collate_fn.py              # Batch collation
└── imaug/                     # Data augmentation
    ├── operators.py           # Basic ops (resize, normalize)
    ├── label_ops.py           # Label encoding/decoding (75KB)
    ├── rec_img_aug.py         # Recognition augmentations
    ├── randaugment.py         # RandAugment policy
    ├── copy_paste.py          # Copy-paste augmentation
    └── ... (20+ augmentation modules)
```

#### Why Data Augmentation is Complex?

OCR has unique challenges:
- **Text rotation**: Real-world text is often tilted
- **Perspective distortion**: Photos of documents have perspective issues
- **Blur and noise**: Low-quality images
- **Varying fonts and sizes**: Must generalize across styles

**Common Augmentations**:
- Rotation, scaling, cropping
- Color jittering
- Gaussian blur, motion blur
- Copy-paste (paste text regions onto new backgrounds)
- RandAugment (automatic augmentation policy search)

---

### 1.3 `ppocr/losses/` - Loss Functions

**Purpose**: Define training objectives for different tasks.

```
losses/
├── det_db_loss.py             # DB detection loss
├── det_east_loss.py           # EAST detection loss
├── rec_ctc_loss.py            # CTC recognition loss
├── rec_att_loss.py            # Attention recognition loss
├── table_att_loss.py          # Table attention loss
├── distillation_loss.py       # Knowledge distillation (41KB)
└── ... (44 loss implementations)
```

#### Why So Many Losses?

Different tasks need different loss functions:
- **Detection**: Segmentation loss (pixel-level text/non-text)
- **Recognition**: Sequence loss (CTC or attention-based)
- **Table**: Combined structure + content loss
- **Distillation**: Transfer knowledge from large model to small model

---

### 1.4 `ppocr/metrics/` - Evaluation Metrics

**Purpose**: Measure model performance during training and evaluation.

```
metrics/
├── det_metric.py              # IoU, Precision, Recall, F1-score
├── rec_metric.py              # Accuracy (character & word level)
├── cls_metric.py              # Classification accuracy
├── table_metric.py            # Table structure accuracy
└── ... (14 metrics)
```

---

### 1.5 `ppocr/postprocess/` - Post-Processing

**Purpose**: Convert raw model outputs to usable results.

```
postprocess/
├── db_postprocess.py          # Convert probability maps to boxes
├── rec_postprocess.py         # Decode predictions to text (58KB)
├── cls_postprocess.py         # Classification post-processing
└── ... (17 post-processing modules)
```

#### Example: Detection Post-Processing

**Model Output**: Probability map (each pixel = text likelihood)
```
[0.1, 0.9, 0.9, 0.1]
[0.1, 0.9, 0.9, 0.1]  → Post-process → Bounding box: [(1,0), (2,0), (2,1), (1,1)]
[0.1, 0.1, 0.1, 0.1]
```

#### Example: Recognition Post-Processing

**Model Output**: Sequence of character probabilities
```
[0.8: 'H', 0.1: 'A', ...]
[0.9: 'e', 0.05: 'o', ...]  → Post-process → "Hello"
[0.7: 'l', 0.2: 'i', ...]
```

---

### 1.6 `ppocr/optimizer/` - Training Optimization

**Purpose**: Learning rate schedulers and optimizer configurations.

```
optimizer/
├── lr_scheduler.py            # Cosine, Step, Warmup, etc.
├── optimizer.py               # Adam, SGD, etc.
└── regularizer.py             # L1, L2 regularization
```

---

### 1.7 `ppocr/utils/` - Utilities

**Purpose**: Helper functions for logging, checkpointing, visualization, etc.

```
utils/
├── dict/                      # Character dictionaries (80+ languages)
│   ├── en_dict.txt           # English characters
│   ├── ch_dict.txt           # Chinese characters
│   ├── arabic_dict.txt       # Arabic characters
│   └── ... (80+ language dicts)
│
├── loggers/                   # Logging integrations
│   ├── vdl_logger.py         # VisualDL logger
│   └── wandb_logger.py       # Weights & Biases logger
│
├── save_load.py              # Model checkpointing
├── stats.py                  # Training statistics
└── utility.py                # General utilities
```

---

## 2. 🛠️ `tools/` - Training & Inference Scripts

**Purpose**: **Entry points** for training, evaluation, and inference.

### Why Separate from `ppocr/`?

- `ppocr/` = Library (reusable code)
- `tools/` = Scripts (executable programs)

### Structure

```
tools/
├── train.py                   # Main training script (10KB)
├── eval.py                    # Evaluation script
├── export_model.py            # Export to inference format
├── program.py                 # Training loop implementation (34KB)
│
├── infer_det.py              # Detection inference
├── infer_rec.py              # Recognition inference
├── infer_cls.py              # Classification inference
├── infer_e2e.py              # End-to-end inference
├── infer_table.py            # Table recognition
│
└── infer/                    # Prediction utilities
    ├── predict_det.py        # Detection predictor class
    ├── predict_rec.py        # Recognition predictor class
    ├── predict_cls.py        # Classification predictor class
    ├── predict_system.py     # Complete OCR system
    └── utility.py            # Helper functions
```

### Key Scripts

#### `train.py` - Start Training
```bash
python tools/train.py -c configs/det/det_db.yml
```
- Loads config
- Builds dataloader, model, loss, optimizer
- Calls training loop

#### `eval.py` - Evaluate Model
```bash
python tools/eval.py -c configs/det/det_db.yml -o Global.checkpoints=output/model
```

#### `export_model.py` - Export for Deployment
```bash
python tools/export_model.py -c configs/det/det_db.yml -o Global.checkpoints=output/model
```
- Converts training model to inference model
- Removes training-specific layers

#### `infer_det.py` / `infer_rec.py` - Quick Inference
```bash
python tools/infer_det.py --image_dir=test.jpg
python tools/infer_rec.py --image_dir=text_crop.jpg
```

---

## 3. ⚙️ `configs/` - Configuration Files

**Purpose**: Define models, training settings, and hyperparameters using YAML.

### Why YAML Configs?

**Without configs** (hardcoded):
```python
model = DBNet(backbone='MobileNetV3', channels=96, scale=0.5)
```
- Requires code changes for experiments
- Hard to track what settings were used

**With configs**:
```yaml
Architecture:
  Backbone:
    name: MobileNetV3
    scale: 0.5
```
- Change settings without touching code
- Easy to version control experiments
- Reproducible research

### Structure

```
configs/
├── det/                       # Text Detection configs
│   ├── PP-OCRv3/
│   ├── PP-OCRv4/
│   ├── PP-OCRv5/
│   └── det_*.yml             # Other algorithms
│
├── rec/                       # Text Recognition configs
│   ├── PP-OCRv3/
│   ├── PP-OCRv4/
│   ├── PP-OCRv5/
│   ├── SVTRv2/
│   ├── multi_language/       # 80+ languages
│   └── ...
│
├── cls/                       # Text Angle Classification
├── table/                     # Table Recognition
├── kie/                       # Key Information Extraction
└── e2e/                       # End-to-End OCR
```

### Config Sections

Every config has these sections:
1. **Global**: Training settings (epochs, save path, GPU, etc.)
2. **Architecture**: Model definition (backbone, neck, head)
3. **Loss**: Loss function
4. **Optimizer**: Learning rate, optimizer type
5. **PostProcess**: Post-processing settings
6. **Metric**: Evaluation metric
7. **Train**: Training data configuration
8. **Eval**: Evaluation data configuration

See `07_Configuration_System.md` for details.

---

## 4. 🚀 `deploy/` - Deployment Solutions

**Purpose**: Deploy models to production environments.

### Structure

```
deploy/
├── cpp_infer/                # C++ inference engine
│   ├── src/                  # C++ source
│   ├── include/              # Headers
│   └── tools/                # Build scripts
│
├── android_demo/             # Android app
├── ios_demo/                 # iOS app
├── lite/                     # Paddle Lite (mobile/embedded)
│
├── hubserving/               # PaddleHub HTTP serving
│   ├── ocr_det/             # Detection service
│   ├── ocr_rec/             # Recognition service
│   ├── ocr_system/          # Complete OCR service
│   └── structure_*/         # Structure services
│
├── paddle2onnx/             # ONNX export
├── slim/                    # Model compression
│   ├── quantization/        # Quantization (INT8)
│   ├── prune/               # Pruning (remove weights)
│   └── auto_compression/    # Automatic compression
│
└── docker/                  # Docker deployment
```

### Why Multiple Deployment Options?

Different scenarios need different solutions:

| Deployment | Use Case | Language | Platform |
|------------|----------|----------|----------|
| Python API | Prototyping, Jupyter | Python | Any |
| C++ Inference | High performance | C++ | Server |
| Mobile (Lite) | Smartphones | Java/Swift | iOS/Android |
| ONNX | Cross-platform | Any | Any runtime |
| Docker | Cloud deployment | Any | Kubernetes |
| Hub Serving | HTTP API | Python | Server |

---

## 5. 🎯 `paddleocr/` - User-Facing API

**Purpose**: Simple Python interface for end users.

### Structure

```
paddleocr/
├── __init__.py              # Main PaddleOCR class
├── __main__.py              # CLI entry point
├── paddleocr.py             # Core implementation
└── tools/                   # Model download utilities
```

### Why Separate from `ppocr/`?

- `ppocr/`: Low-level library (for training & development)
- `paddleocr/`: High-level API (for users)

### Usage

```python
from paddleocr import PaddleOCR

# One-line initialization
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# One-line inference
result = ocr.ocr('image.jpg')
```

**Behind the scenes**:
1. Downloads pre-trained models (first time only)
2. Loads detection, recognition, and classification models
3. Runs complete OCR pipeline
4. Returns structured results

---

## 6. 📊 `benchmark/` - Performance Benchmarking

**Purpose**: Measure and optimize model performance.

### Structure

```
benchmark/
├── PaddleOCR_DBNet/         # DBNet benchmark
├── analysis.py              # Performance analysis
├── run_benchmark_det.sh     # Detection benchmark script
└── run_det.sh              # Detection inference
```

### Why Benchmark?

To measure:
- **Inference speed** (FPS, latency)
- **Memory usage**
- **Accuracy** vs speed trade-offs
- **Optimization effects** (quantization, pruning)

### When to Use?

- Comparing different models
- Optimizing for production
- Hardware-specific tuning (CPU vs GPU)

---

## 7. 📄 `ppstructure/` - Document Structure Analysis

**Purpose**: Advanced document understanding (beyond basic OCR).

### Structure

```
ppstructure/
├── layout/                   # Layout analysis (find regions)
├── table/                    # Table recognition
│   ├── table_metric/        # Table evaluation
│   └── tablepyxl/          # Excel export
├── kie/                     # Key Information Extraction
├── recovery/                # Document recovery (to Word/Markdown)
└── pdf2word/                # PDF to Word conversion
```

### Why Separate?

- **PaddleOCR**: Basic OCR (text detection + recognition)
- **PaddleStructure**: Advanced document understanding

Not everyone needs document structure analysis, so it's modular.

---

## 8. 📚 `doc/` and `docs/` - Documentation

**Purpose**: User guides, API references, tutorials.

```
doc/                         # Legacy documentation
docs/                        # Main documentation
├── quick_start.md          # Getting started
├── training.md             # Training guide
├── inference.md            # Inference guide
└── ...
```

---

## 9. ✅ `test_tipc/` and `tests/` - Testing

**Purpose**: Ensure code quality and catch bugs.

```
test_tipc/                   # Test in Production CI
tests/                       # Unit tests
```

---

## 10. 🎨 `StyleText/` - Styled Text Generation

**Purpose**: Generate synthetic training data with various text styles.

### Why?

Training OCR models requires **lots of labeled data**. StyleText can:
- Generate realistic text images
- Apply different fonts, colors, backgrounds
- Create augmented training data

---

## 11. 💼 `applications/` - Example Applications

**Purpose**: Real-world use case examples.

Examples might include:
- Invoice processing
- ID card recognition
- License plate recognition

---

## Summary Table

| Folder | Purpose | For Training? | For Inference? | For Users? |
|--------|---------|---------------|----------------|------------|
| `ppocr/` | Core library | ✅ Yes | ✅ Yes | ❌ No (low-level) |
| `tools/` | Scripts | ✅ Yes | ✅ Yes | ⚠️ Advanced users |
| `configs/` | Model configs | ✅ Yes | ⚠️ Some | ⚠️ Advanced users |
| `deploy/` | Deployment | ❌ No | ✅ Yes | ✅ Yes |
| `paddleocr/` | Simple API | ❌ No | ✅ Yes | ✅ Yes (main interface) |
| `benchmark/` | Benchmarking | ❌ No | ⚠️ Optimization | ⚠️ Advanced users |
| `ppstructure/` | Doc structure | ⚠️ Some | ✅ Yes | ✅ Yes (if needed) |
| `doc/docs/` | Documentation | ❌ No | ❌ No | ✅ Yes |
| `tests/` | Testing | ❌ No | ❌ No | ❌ No (developers) |

---

## Quick Reference

### I want to...

**Use PaddleOCR** → Start with `paddleocr/`
**Train a model** → Use `tools/train.py` + `configs/`
**Understand models** → Read `ppocr/modeling/`
**Deploy to production** → Check `deploy/`
**Benchmark performance** → Use `benchmark/`
**Process documents** → Explore `ppstructure/`

---

Next: [Architecture Deep Dive](./03_Architecture_Explained.md)
