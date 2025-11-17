# Architecture Deep Dive

This document explains **how PaddleOCR models are structured** and why they use the modular design pattern.

## 🏗️ The Core Architecture Pattern

All models in PaddleOCR follow this pattern:

```
Input Image → Transform → Backbone → Neck → Head → Output
     📷          🔄          🧠        🔗      🎯      📊
```

Let's understand each component with simple examples.

---

## 1. Transform (Optional) - Image Transformation

**Purpose**: Apply geometric transformations to normalize or augment input images.

### Example: Thin Plate Spline (TPS)

**Problem**: Text might be curved or warped
```
Original:  ╭─ Hello ─╮
            curved text

After TPS: ─ Hello ─
           straight text
```

**When used?**
- Recognition models for curved text (street signs, product labels)
- Scene text OCR

**Common transforms**:
- TPS (Thin Plate Spline)
- STN (Spatial Transformer Network)

**Code reference**: `ppocr/modeling/transforms/tps.py`

---

## 2. Backbone - Feature Extraction

**Purpose**: Extract visual features from images.

Think of it as the "eyes" of the model - it sees patterns, edges, textures.

### What are Features?

**Low-level features** (early layers):
- Edges
- Corners
- Basic shapes

**High-level features** (later layers):
- Text patterns
- Character shapes
- Word structures

### Visual Example

```
Input Image:        Feature Maps:
┌──────────┐       ┌──────────┐  ┌──────────┐  ┌──────────┐
│          │       │██  ██  ██│  │  ████    │  │    ██    │
│  Hello   │  →    │█ ██ ███ │  │████  ████│  │  ██  ██  │
│          │       │██  ██  ██│  │  ████    │  │    ██    │
└──────────┘       └──────────┘  └──────────┘  └──────────┘
                   Edges         Corners       Text regions
```

### Common Backbones

#### 1. **MobileNetV3** (Lightweight)
- **Size**: ~5 MB
- **Speed**: Fast (mobile-friendly)
- **Accuracy**: Good
- **Use case**: Mobile apps, edge devices

**Code**: `ppocr/modeling/backbones/det_mobilenet_v3.py`

#### 2. **ResNet** (Balanced)
- **Size**: ~50 MB
- **Speed**: Medium
- **Accuracy**: Very good
- **Use case**: Server deployment, high accuracy needed

**Code**: `ppocr/modeling/backbones/det_resnet_vd.py`

#### 3. **SVTR** (High accuracy for recognition)
- **Size**: ~30 MB
- **Speed**: Medium
- **Accuracy**: Excellent
- **Use case**: Recognition models in PP-OCRv3+

**Code**: `ppocr/modeling/backbones/rec_svtrnet.py`

#### 4. **PPHGNetV2** (Latest, best accuracy)
- **Size**: ~100 MB
- **Speed**: Slower but very accurate
- **Accuracy**: State-of-the-art
- **Use case**: PP-OCRv4/v5 server models

**Code**: `ppocr/modeling/backbones/rec_pphgnetv2.py`

### Backbone Selection Trade-offs

```
Mobile Models:     Server Models:
Fast ⚡            Accurate 🎯
Small 📦           Large 📚
Good enough ✓      Best quality ✓✓✓
```

---

## 3. Neck - Feature Refinement

**Purpose**: Combine and refine features from the backbone.

Think of it as "connecting the dots" - it makes sense of the features.

### Common Necks

#### 1. **FPN (Feature Pyramid Network)** - For Detection

**What it does**: Combines multi-scale features

```
Backbone Output:           FPN Output:
┌─────┐                   ┌─────┐
│  C5 │ (small, deep)     │  P5 │ (detect large text)
└──┬──┘                   └──┬──┘
   │  ┌─────┐                │  ┌─────┐
   └──│  C4 │ (medium)       └──│  P4 │ (detect medium text)
      └──┬──┘                   └──┬──┘
         │  ┌─────┐                │  ┌─────┐
         └──│  C3 │ (large)        └──│  P3 │ (detect small text)
            └─────┘                   └─────┘

Multi-scale features → Detect text at different sizes
```

**Why needed?**
- Text comes in different sizes (small captions vs large titles)
- FPN allows detecting all sizes efficiently

**Code**: `ppocr/modeling/necks/db_fpn.py`

#### 2. **RNN (Recurrent Neural Network)** - For Recognition

**What it does**: Models sequence information (left-to-right text)

```
Character features: [H] [e] [l] [l] [o]
                     ↓   ↓   ↓   ↓   ↓
RNN processes:      → → → → → → → → →
                     ↑   ↑   ↑   ↑   ↑
Context-aware:      [H] [He] [Hel] [Hell] [Hello]

Each character "knows" what came before it
```

**Why needed?**
- Text is sequential (order matters)
- Context helps (in "cat", 'a' comes between 'c' and 't')

**Code**: `ppocr/modeling/necks/rnn.py`

#### 3. **PAN (Path Aggregation Network)** - For Detection

Similar to FPN but with better feature flow.

**Code**: `ppocr/modeling/necks/csp_pan.py`

### Neck Selection

| Task | Neck | Why? |
|------|------|------|
| Detection | FPN, PAN | Multi-scale features for different text sizes |
| Recognition | RNN, Transformer | Sequence modeling for reading order |
| Table | FPN + Attention | Structure + sequence understanding |

---

## 4. Head - Task-Specific Predictions

**Purpose**: Generate the final output for specific tasks.

Think of it as the "decision maker" - it produces the actual predictions.

### Detection Heads

#### **DB Head** (Differentiable Binarization)

**Input**: Feature maps from neck
**Output**: Probability map + threshold map

```
Feature Map →  DB Head  → Probability Map  → Post-process → Bounding Boxes
                           (0.0 to 1.0)                      [(x,y), (x,y), ...]

Visual:
┌──────────┐            ┌──────────┐            ┌──────────┐
│          │            │██████    │            │┌────────┐│
│  Image   │      →     │██████    │      →     ││ Hello  ││
│          │            │     ████ │            │└────────┘│
└──────────┘            └──────────┘            └──────────┘
```

**Code**: `ppocr/modeling/heads/det_db_head.py`

---

### Recognition Heads

#### 1. **CTC Head** (Connectionist Temporal Classification)

**Input**: Sequence of features
**Output**: Character probabilities at each time step

```
Input features:  [Feature1, Feature2, Feature3, ...]
                     ↓         ↓         ↓
CTC Head:        [Prob_H,  Prob_e,  Prob_l,  ...]
                     ↓         ↓         ↓
Post-process:        H    →    e    →    l    →  "Hello"
```

**Advantages**:
- Simple and fast
- No need for character-level alignment

**Code**: `ppocr/modeling/heads/rec_ctc_head.py`

#### 2. **Attention Head**

**Input**: Sequence of features
**Output**: Character probabilities with attention mechanism

```
Features:  [F1, F2, F3, F4, F5]
            ↓   ↓   ↓   ↓   ↓
Attention: "Look at F1" → predict 'H'
           "Look at F2" → predict 'e'
           "Look at F3" → predict 'l'
           ...
```

**Advantages**:
- Better for irregular text
- Can handle varying-length text better

**Code**: `ppocr/modeling/heads/rec_att_head.py`

---

## 🔄 Complete Flow: Detection Example

Let's trace an image through a **DB (Differentiable Binarization) detection model**.

### Configuration

```yaml
Architecture:
  model_type: det
  algorithm: DB
  Transform: null
  Backbone:
    name: MobileNetV3
    scale: 0.5
  Neck:
    name: DBFPN
    out_channels: 256
  Head:
    name: DBHead
    k: 50
```

### Step-by-Step Flow

```
1. Input Image (640x640x3)
   ↓
2. Backbone: MobileNetV3
   - Extract features at multiple scales
   - Output: [C2, C3, C4, C5] feature maps
   ↓
3. Neck: DBFPN
   - Combine multi-scale features
   - Output: [P2, P3, P4, P5] refined features
   ↓
4. Head: DBHead
   - Predict probability map (text vs non-text)
   - Predict threshold map (for binarization)
   - Output: (prob_map, threshold_map)
   ↓
5. Post-Processing
   - Binarize probability map using threshold
   - Find contours (connected regions)
   - Filter small regions
   - Output: List of bounding boxes
```

### Visual Flow

```
┌─────────────┐
│   Image     │ (640x640x3)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ MobileNetV3         │
│ (Backbone)          │
│ • Conv layers       │
│ • Extract features  │
└──────┬──────────────┘
       │ Feature maps: [10x10x96, 20x20x48, ...]
       ▼
┌─────────────────────┐
│ DBFPN               │
│ (Neck)              │
│ • Fuse features     │
│ • Multi-scale       │
└──────┬──────────────┘
       │ Refined features: [160x160x256]
       ▼
┌─────────────────────┐
│ DBHead              │
│ (Head)              │
│ • Predict prob_map  │
│ • Predict threshold │
└──────┬──────────────┘
       │ prob_map: (640x640), threshold_map: (640x640)
       ▼
┌─────────────────────┐
│ Post-Processing     │
│ • Binarize          │
│ • Find contours     │
│ • Output boxes      │
└──────┬──────────────┘
       │
       ▼
   [Box1, Box2, ...]
   [(x1,y1,x2,y2), ...]
```

---

## 🔄 Complete Flow: Recognition Example

Let's trace a cropped text image through a **SVTR recognition model**.

### Configuration

```yaml
Architecture:
  model_type: rec
  algorithm: SVTR
  Transform: null
  Backbone:
    name: SVTRNet
    dims: [64, 128, 256]
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead
```

### Step-by-Step Flow

```
1. Input Image (32x100x3) - cropped text region
   ↓
2. Backbone: SVTRNet
   - Extract visual features
   - Output: (H/8 x W/4 x C) feature map
   ↓
3. Neck: SequenceEncoder
   - Reshape to sequence: (W/4) x C
   - Each position = one "time step"
   ↓
4. Head: CTCHead
   - Predict character probabilities at each time step
   - Output: (W/4 x num_classes)
   ↓
5. Post-Processing (CTC Decode)
   - Remove duplicates and blank tokens
   - Output: "Hello" (character sequence)
```

### Visual Flow

```
┌─────────────┐
│ Text Crop   │ (32x100x3) - "Hello"
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ SVTRNet             │
│ (Backbone)          │
│ • Vision Trans.     │
│ • Feature extract   │
└──────┬──────────────┘
       │ Features: (4x25x256)
       ▼
┌─────────────────────┐
│ SequenceEncoder     │
│ (Neck)              │
│ • Reshape to seq    │
└──────┬──────────────┘
       │ Sequence: 25 time steps x 256 dim
       ▼
┌─────────────────────┐
│ CTCHead             │
│ (Head)              │
│ • Predict chars     │
└──────┬──────────────┘
       │ Probabilities: 25 x 6625 (char classes)
       ▼
┌─────────────────────┐
│ CTC Decode          │
│ • Remove blanks     │
│ • Collapse dups     │
└──────┬──────────────┘
       │
       ▼
    "Hello"
```

---

## 🔧 Why This Modular Design?

### 1. **Flexibility**

You can mix and match components:

```yaml
# Configuration A: Fast mobile model
Backbone: MobileNetV3
Neck: DBFPN
Head: DBHead

# Configuration B: Accurate server model (just change backbone!)
Backbone: ResNet50
Neck: DBFPN      # Same neck
Head: DBHead     # Same head
```

### 2. **Reusability**

One backbone can be used for multiple tasks:

```
MobileNetV3
    ├── Detection (+ DBFPN + DBHead)
    ├── Recognition (+ RNN + CTCHead)
    └── Classification (+ Pooling + ClsHead)
```

### 3. **Experimentation**

Easy to try new ideas:
- Test different backbones
- Compare FPN vs PAN necks
- Try CTC vs Attention heads

Just change the YAML config - no code changes!

### 4. **Clear Separation of Concerns**

Each component has one job:
- **Backbone**: Extract features
- **Neck**: Refine features
- **Head**: Make predictions

Easier to understand, debug, and improve.

---

## 🎓 Common Architecture Combinations

### Detection

| Model | Backbone | Neck | Head | Characteristics |
|-------|----------|------|------|-----------------|
| PP-OCRv3 Det | MobileNetV3 | DBFPN | DBHead | Fast, mobile-friendly |
| PP-OCRv4 Det | PPLCNetV3 | DBFPN | DBHead | Balanced speed/accuracy |
| DB ResNet50 | ResNet50 | DBFPN | DBHead | High accuracy, slower |

### Recognition

| Model | Backbone | Neck | Head | Characteristics |
|-------|----------|------|------|-----------------|
| PP-OCRv3 Rec | SVTRNet | SequenceEncoder | CTCHead | High accuracy |
| PP-OCRv4 Rec | PPHGNetV2 | RNN | CTCHead | Best accuracy |
| CRNN | ResNet | RNN | CTCHead | Classic, widely used |

---

## 📊 Model Building Process

Here's how PaddleOCR builds a model from config:

```python
# Simplified from ppocr/modeling/architectures/base_model.py

class BaseModel(nn.Layer):
    def __init__(self, config):
        super().__init__()

        # Build each component from config
        self.backbone = build_backbone(config['Backbone'])
        self.neck = build_neck(config['Neck']) if config['Neck'] else None
        self.head = build_head(config['Head'])

    def forward(self, x):
        # Forward pass through pipeline
        x = self.backbone(x)        # Extract features
        if self.neck:
            x = self.neck(x)         # Refine features
        x = self.head(x)             # Make predictions
        return x
```

**Factory pattern** (from `__init__.py` files):

```python
def build_backbone(config):
    name = config['name']
    if name == 'MobileNetV3':
        return MobileNetV3(**config)
    elif name == 'ResNet':
        return ResNet(**config)
    # ... etc.
```

---

## 🔍 Code References

### Key Files

1. **Base Model Architecture**
   - `ppocr/modeling/architectures/base_model.py` - Orchestrates components

2. **Backbones**
   - `ppocr/modeling/backbones/det_mobilenet_v3.py` - MobileNetV3 for detection
   - `ppocr/modeling/backbones/rec_svtrnet.py` - SVTR for recognition

3. **Necks**
   - `ppocr/modeling/necks/db_fpn.py` - FPN for detection
   - `ppocr/modeling/necks/rnn.py` - RNN for recognition

4. **Heads**
   - `ppocr/modeling/heads/det_db_head.py` - DB detection head
   - `ppocr/modeling/heads/rec_ctc_head.py` - CTC recognition head

---

## 💡 Key Takeaways

1. **Modular Design**: Transform → Backbone → Neck → Head
2. **Flexible**: Easy to experiment with different combinations
3. **Reusable**: Same components across different tasks
4. **Config-Driven**: Change models without code changes
5. **Clear Roles**: Each component has a specific purpose

---

Next: [Implementation from Scratch](./04_Implementation_From_Scratch.md)
