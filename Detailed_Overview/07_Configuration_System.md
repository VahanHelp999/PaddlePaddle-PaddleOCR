# Configuration System: Understanding YAML Configs

This guide explains how PaddleOCR's configuration system works and how to create your own configs.

## 🎯 Why Use Configuration Files?

### Without Configs (Hardcoded) ❌
```python
model = DBNet(
    backbone='MobileNetV3',
    scale=0.5,
    fpn_channels=256,
    k=50
)
optimizer = Adam(lr=0.001)
train_for_epochs(500)
```
**Problems**:
- Requires code changes for experiments
- Hard to track what settings were used
- Not reproducible

### With Configs (YAML) ✅
```yaml
# config.yml
Architecture:
  Backbone:
    name: MobileNetV3
    scale: 0.5

Optimizer:
  name: Adam
  lr: 0.001

Global:
  epoch_num: 500
```
**Benefits**:
- No code changes needed
- Easy to version control
- Reproducible experiments
- Clear documentation of settings

---

## 📁 Config File Structure

Every PaddleOCR config has **8 main sections**:

```yaml
Global:           # Training settings (epochs, GPU, paths)
Architecture:     # Model definition (backbone, neck, head)
Loss:             # Loss function
Optimizer:        # Optimizer and learning rate schedule
PostProcess:      # Post-processing settings
Metric:           # Evaluation metric
Train:            # Training data configuration
Eval:             # Evaluation data configuration
```

---

## 🔧 Section 1: Global

**Purpose**: General training settings and runtime configuration.

```yaml
Global:
  use_gpu: true                          # Use GPU?
  epoch_num: 500                         # Number of training epochs
  log_smooth_window: 20                  # Logging smoothing window
  print_batch_step: 10                   # Print every N batches
  save_model_dir: ./output/db_mv3/       # Where to save models
  save_epoch_step: 3                     # Save checkpoint every N epochs
  eval_batch_step: [0, 2000]             # Evaluate at these steps
  cal_metric_during_train: True          # Calculate metrics during training?
  pretrained_model: ./pretrain_models/   # Pre-trained weights path
  checkpoints:                           # Resume from checkpoint (optional)
  save_inference_dir:                    # Inference model output directory
  use_visualdl: False                    # Use VisualDL for visualization?
  infer_img: doc/imgs_en/img_10.jpg      # Test inference image
  save_res_path: ./output/predicts.txt   # Save prediction results
  distributed: false                     # Distributed training?
```

### Common Parameters Explained

#### `use_gpu`
- `true`: Train on GPU (faster)
- `false`: Train on CPU (slower)

#### `epoch_num`
- Detection: 500-1200 epochs
- Recognition: 300-800 epochs
- Fine-tuning: 10-50 epochs

#### `save_epoch_step`
- How often to save checkpoints
- Smaller = more checkpoints (safer but more disk space)
- Larger = fewer checkpoints (risk losing progress)

#### `eval_batch_step`
- When to run evaluation
- `[0, 2000]` = Evaluate at step 0 and every 2000 steps
- `[0, 500, 1000, 1500]` = Evaluate at specific steps

#### `pretrained_model`
- Path to pre-trained weights
- **Always use pre-trained weights!** (Much better results)
- Download from PaddleOCR model zoo

---

## 🏗️ Section 2: Architecture

**Purpose**: Define the model structure.

### Detection Model Example

```yaml
Architecture:
  model_type: det                        # Type: det/rec/cls/table/kie
  algorithm: DB                          # Algorithm name
  Transform:                             # Optional transformation (TPS, etc.)
  Backbone:                              # Feature extraction
    name: MobileNetV3                    # Backbone name
    scale: 0.5                           # Model scale (0.5/1.0/etc.)
    model_name: large                    # Variant (large/small)
  Neck:                                  # Feature refinement
    name: DBFPN                          # Neck name
    out_channels: 256                    # Output channels
    use_asf: False                       # Additional feature fusion?
  Head:                                  # Prediction head
    name: DBHead                         # Head name
    k: 50                                # Expanding parameter
```

### Recognition Model Example

```yaml
Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:                             # Spatial transformation (optional)
    name: TPS                            # Thin Plate Spline
    num_fiducial: 20
  Backbone:
    name: SVTRNet                        # SVTR backbone
    dims: [64, 128, 256]                 # Hidden dimensions
    depth: [3, 6, 3]                     # Number of blocks at each stage
    num_heads: [2, 4, 8]                 # Attention heads
    mixer: ['Local', 'Local', 'Global']  # Mixer types
  Neck:
    name: SequenceEncoder                # Sequence encoder
    encoder_type: reshape                # Type: reshape/rnn/svtr
  Head:
    name: CTCHead                        # CTC head
```

### Key Parameters

#### `model_type`
- `det`: Text detection
- `rec`: Text recognition
- `cls`: Text direction classification
- `table`: Table recognition
- `kie`: Key information extraction

#### Backbone Options

**Detection**:
- `MobileNetV3` (lightweight)
- `ResNet` (accurate)
- `PPLCNetV3` (PP-OCRv4)

**Recognition**:
- `ResNet` (legacy)
- `SVTRNet` (PP-OCRv3)
- `PPHGNetV2` (PP-OCRv4)

#### Neck Options

**Detection**:
- `DBFPN` (DB algorithm)
- `EASTFPN` (EAST algorithm)
- `PSEFPN` (PSE algorithm)

**Recognition**:
- `SequenceEncoder` (reshape/rnn)
- `MultiHead` (attention)

#### Head Options

**Detection**:
- `DBHead` (DB)
- `EASTHead` (EAST)
- `PSEHead` (PSE)

**Recognition**:
- `CTCHead` (CTC decoding)
- `AttentionHead` (Attention decoding)
- `SRNHead` (SRN algorithm)

---

## 📉 Section 3: Loss

**Purpose**: Define the training objective.

### Detection Loss Example (DB)

```yaml
Loss:
  name: DBLoss                           # Loss function name
  balance_loss: true                     # Balance positive/negative samples?
  main_loss_type: DiceLoss               # Main loss type
  alpha: 5                               # Probability map loss weight
  beta: 10                               # Binary map loss weight
  ohem_ratio: 3                          # Hard example mining ratio
```

### Recognition Loss Example (CTC)

```yaml
Loss:
  name: CTCLoss                          # CTC loss
```

### Common Loss Functions

| Task | Loss | Description |
|------|------|-------------|
| Detection | DBLoss | DB algorithm loss (prob + threshold + binary) |
| Detection | EASTLoss | EAST algorithm loss |
| Recognition | CTCLoss | CTC loss for sequence recognition |
| Recognition | AttentionLoss | Attention-based loss |
| Classification | ClsLoss | Cross-entropy loss |

---

## 🎛️ Section 4: Optimizer

**Purpose**: Configure optimization and learning rate.

```yaml
Optimizer:
  name: Adam                             # Optimizer type
  beta1: 0.9                             # Adam beta1 parameter
  beta2: 0.999                           # Adam beta2 parameter
  lr:                                    # Learning rate schedule
    name: Cosine                         # LR scheduler type
    learning_rate: 0.001                 # Initial learning rate
    warmup_epoch: 2                      # Warmup epochs
  regularizer:                           # Weight regularization
    name: L2                             # L2 regularization
    factor: 0.00001                      # Regularization strength
```

### Optimizer Options

| Optimizer | Use Case |
|-----------|----------|
| `Adam` | General purpose (default) |
| `AdamW` | Adam with weight decay (better for transformers) |
| `SGD` | Traditional, with momentum |
| `RMSprop` | Good for RNNs |

### Learning Rate Schedulers

#### 1. **Cosine** (Recommended)
```yaml
lr:
  name: Cosine
  learning_rate: 0.001                   # Max LR
  warmup_epoch: 2                        # Warmup period
```
- Smoothly decreases LR over training
- Good for most cases

#### 2. **Step**
```yaml
lr:
  name: Step
  learning_rate: 0.001
  step_size: 100                         # Decrease every N epochs
  gamma: 0.1                             # Multiply LR by gamma
```
- Decreases LR at fixed intervals
- Traditional approach

#### 3. **Piecewise**
```yaml
lr:
  name: Piecewise
  learning_rate: 0.001
  decay_epochs: [100, 200, 300]          # Decrease at these epochs
  values: [0.001, 0.0005, 0.0001, 0.00001]  # LR values
```
- Manual control of LR
- Fine-grained tuning

### Learning Rate Guidelines

| Scenario | Learning Rate | Scheduler |
|----------|---------------|-----------|
| Training from scratch | 0.001 | Cosine |
| Fine-tuning | 0.0001 | Cosine or Piecewise |
| Small batch size | Lower (0.0005) | Cosine |
| Large batch size | Higher (0.002) | Cosine with warmup |

---

## 🔄 Section 5: PostProcess

**Purpose**: Configure post-processing to convert model outputs to usable results.

### Detection PostProcess

```yaml
PostProcess:
  name: DBPostProcess                    # Post-processor name
  thresh: 0.3                            # Probability threshold
  box_thresh: 0.6                        # Box confidence threshold
  max_candidates: 1000                   # Max number of candidates
  unclip_ratio: 1.5                      # Box expansion ratio
```

**Parameters explained**:
- `thresh`: Minimum probability to consider as text (lower = more sensitive)
- `box_thresh`: Minimum confidence for a box (lower = more boxes)
- `unclip_ratio`: Expand boxes by this ratio (account for shrinkage during training)

### Recognition PostProcess

```yaml
PostProcess:
  name: CTCLabelDecode                   # Decoder name
```

For attention-based recognition:
```yaml
PostProcess:
  name: AttentionLabelDecode
```

---

## 📊 Section 6: Metric

**Purpose**: Define evaluation metrics.

### Detection Metric

```yaml
Metric:
  name: DetMetric                        # Detection metric
  main_indicator: hmean                  # Main indicator: hmean/precision/recall
```

**Metrics computed**:
- Precision: % of predicted boxes that are correct
- Recall: % of ground truth boxes detected
- F1-score (hmean): Harmonic mean of precision and recall

### Recognition Metric

```yaml
Metric:
  name: RecMetric                        # Recognition metric
  main_indicator: acc                    # Main indicator: acc (accuracy)
  ignore_space: True                     # Ignore spaces in comparison?
```

---

## 📂 Section 7: Train

**Purpose**: Configure training data loading and augmentation.

```yaml
Train:
  dataset:
    name: SimpleDataSet                  # Dataset class
    data_dir: ./train_data/              # Data directory
    label_file_list:                     # Label files
      - ./train_data/train_list.txt
    ratio_list: [1.0]                    # Sampling ratio for each label file
    transforms:                          # Data augmentation pipeline
      - DecodeImage:                     # 1. Decode image
          img_mode: BGR
          channel_first: False
      - DetLabelEncode:                  # 2. Encode labels
      - IaaAugment:                      # 3. Apply augmentations
          augmenter_args:
            - { 'type': Fliplr, 'args': { 'p': 0.5 } }
            - { 'type': Affine, 'args': { 'rotate': [-10, 10] } }
      - EastRandomCropData:              # 4. Random crop
          size: [640, 640]
      - MakeBorderMap:                   # 5. Generate GT maps
          shrink_ratio: 0.4
      - MakeShrinkMap:
          shrink_ratio: 0.4
      - NormalizeImage:                  # 6. Normalize
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      - ToCHWImage:                      # 7. Convert to CHW format
      - KeepKeys:                        # 8. Keep only needed keys
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
  loader:
    shuffle: True                        # Shuffle data?
    drop_last: False                     # Drop last incomplete batch?
    batch_size_per_card: 16              # Batch size per GPU
    num_workers: 8                       # Data loading workers
```

### Data Augmentation Pipeline

The `transforms` list is executed sequentially:

1. **DecodeImage**: Load image from file
2. **LabelEncode**: Parse and encode labels
3. **Augmentation**: Apply random transformations
4. **Crop/Resize**: Adjust image size
5. **Generate GT**: Create ground truth maps (for detection)
6. **Normalize**: Normalize pixel values
7. **Format**: Convert to tensor format
8. **KeepKeys**: Select which data to return

### Common Augmentations

#### Detection
- `IaaAugment`: Flip, rotate, scale
- `RandomCrop`: Random crops
- `ColorJitter`: Color variations
- `Blur`: Gaussian blur

#### Recognition
- `RecAug`: Recognition-specific augmentations
- `RecResizeImg`: Resize to fixed height
- `Normalize`: Pixel normalization

---

## 📂 Section 8: Eval

**Purpose**: Configure evaluation data loading.

```yaml
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/val_list.txt
    transforms:                          # Usually less augmentation than training
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - DetLabelEncode:
      - DetResizeForTest:                # Fixed resize (no random crop)
          limit_side_len: 736
          limit_type: 'min'
      - NormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'shape', 'polys', 'ignore_tags']
  loader:
    shuffle: False                       # Don't shuffle evaluation data
    drop_last: False
    batch_size_per_card: 1               # Usually batch size 1 for eval
    num_workers: 2
```

**Key differences from Train**:
- No data augmentation (or minimal)
- `shuffle: False` (deterministic evaluation)
- Smaller batch size (often 1)

---

## 🔧 Creating Your Own Config

### Step 1: Start from a Template

```bash
# Copy an existing config as a template
cp configs/det/PP-OCRv4/ch_PP-OCRv4_det_mobile.yml configs/det/my_custom_config.yml
```

### Step 2: Modify for Your Task

```yaml
# my_custom_config.yml

Global:
  save_model_dir: ./output/my_model/     # Change output directory
  pretrained_model: ./pretrain/best.pdparams  # Your pre-trained model

Architecture:
  Backbone:
    scale: 1.0                           # Increase model size

Optimizer:
  lr:
    learning_rate: 0.0001                # Lower LR for fine-tuning

Train:
  dataset:
    data_dir: ./my_dataset/              # Your dataset path
    label_file_list:
      - ./my_dataset/train.txt           # Your labels
  loader:
    batch_size_per_card: 8               # Adjust batch size
```

### Step 3: Test Your Config

```bash
# Dry run (test config without training)
python tools/train.py -c configs/det/my_custom_config.yml --dry_run
```

---

## 🎛️ Command-Line Overrides

You can override config parameters from command line:

```bash
# Override single parameter
python tools/train.py \
    -c configs/det/det_db.yml \
    -o Global.epoch_num=1000

# Override multiple parameters
python tools/train.py \
    -c configs/det/det_db.yml \
    -o Global.epoch_num=1000 \
       Optimizer.lr.learning_rate=0.001 \
       Train.loader.batch_size_per_card=32

# Override nested parameters
python tools/train.py \
    -c configs/det/det_db.yml \
    -o Architecture.Backbone.scale=1.0
```

**Syntax**: `-o Section.SubSection.Parameter=Value`

---

## 📝 Config Best Practices

### 1. Version Control
```bash
git add configs/det/my_config.yml
git commit -m "Add custom detection config"
```

### 2. Naming Convention
```
{task}_{algorithm}_{variant}_{notes}.yml

Examples:
- det_db_mv3_finetune.yml
- rec_svtr_base_en.yml
- det_db_r50_icdar15.yml
```

### 3. Comments
```yaml
Architecture:
  Backbone:
    name: MobileNetV3
    scale: 0.5              # Reduced from 1.0 for faster inference
```

### 4. Organize Experiments
```
configs/
└── det/
    └── experiments/
        ├── exp1_baseline.yml
        ├── exp2_larger_model.yml
        ├── exp3_more_aug.yml
        └── exp4_best.yml
```

---

## 🔍 Config File Locations

```
configs/
├── det/                       # Detection configs
│   ├── PP-OCRv3/
│   ├── PP-OCRv4/
│   ├── PP-OCRv5/
│   └── det_*.yml
├── rec/                       # Recognition configs
│   ├── PP-OCRv3/
│   ├── PP-OCRv4/
│   ├── multi_language/
│   └── rec_*.yml
├── cls/                       # Classification configs
└── table/                     # Table configs
```

---

## 💡 Common Config Patterns

### Fine-tuning Config
```yaml
Global:
  epoch_num: 50                          # Fewer epochs
  pretrained_model: ./pretrain/best      # Load pre-trained weights

Optimizer:
  lr:
    learning_rate: 0.0001                # Lower learning rate
```

### Large Batch Training
```yaml
Train:
  loader:
    batch_size_per_card: 64              # Larger batch size

Optimizer:
  lr:
    learning_rate: 0.002                 # Scale up learning rate
    warmup_epoch: 5                      # Longer warmup
```

### Multi-GPU Training
```yaml
Global:
  distributed: true                      # Enable distributed training

Train:
  loader:
    batch_size_per_card: 16              # Batch size per GPU
    # Total batch size = 16 x num_GPUs
```

---

## 🔗 Further Reading

- See [08_Training_Pipeline.md](./08_Training_Pipeline.md) for how configs are used during training
- See [06_Models_Explained.md](./06_Models_Explained.md) for architecture options

---

**Summary**: PaddleOCR's YAML configuration system allows flexible experimentation without code changes. The 8-section structure (Global, Architecture, Loss, Optimizer, PostProcess, Metric, Train, Eval) covers all aspects of model training and evaluation.
