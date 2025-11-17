# Implementation from Scratch: Building Your Own OCR System

This guide explains **how to build an OCR system like PaddleOCR from the ground up**, including what to learn, what order to implement things, and why.

## 🎯 Overview: The Journey

Building an OCR system is a multi-stage journey:

```
Stage 1: Prerequisites (2-4 weeks)
   ↓
Stage 2: Simple Detection (2-3 weeks)
   ↓
Stage 3: Simple Recognition (2-3 weeks)
   ↓
Stage 4: Complete Pipeline (1-2 weeks)
   ↓
Stage 5: Optimization (ongoing)
```

**Total time estimate**: 2-3 months for a basic working system (for someone with DL experience)

---

## 📚 Stage 1: Prerequisites & Foundation

### 1.1 Knowledge You Need

#### A. Python Basics ✓
- Object-oriented programming
- File I/O
- Basic data structures (lists, dicts)

#### B. Deep Learning Fundamentals ⚠️ (Critical!)

**Concepts to master**:
1. **Neural Networks**
   - Layers (Dense, Conv, RNN, Attention)
   - Activation functions (ReLU, Sigmoid, Softmax)
   - Loss functions (MSE, Cross-entropy)
   - Backpropagation

2. **CNNs (Convolutional Neural Networks)**
   - Convolution operation
   - Pooling
   - Feature maps
   - Common architectures (ResNet, MobileNet)

3. **RNNs (Recurrent Neural Networks)**
   - LSTM/GRU
   - Sequence modeling
   - Bidirectional RNNs

4. **Transformers** (for advanced models)
   - Self-attention
   - Multi-head attention
   - Vision Transformers

**Learning resources**:
- Course: Fast.ai Practical Deep Learning
- Book: "Deep Learning" by Ian Goodfellow
- Tutorial: PyTorch/PaddlePaddle official tutorials

#### C. Computer Vision Basics

1. **Image Processing**
   - Color spaces (RGB, grayscale)
   - Resizing, cropping, padding
   - Normalization

2. **Object Detection** (for text detection)
   - Bounding boxes
   - IoU (Intersection over Union)
   - Non-maximum suppression (NMS)

3. **Sequence Recognition** (for text recognition)
   - CTC (Connectionist Temporal Classification)
   - Attention mechanisms

#### D. Framework Knowledge

Choose one deep learning framework:
- **PaddlePaddle** (what PaddleOCR uses)
- **PyTorch** (most popular)
- **TensorFlow** (widely used)

Learn:
- Building models (nn.Layer/nn.Module)
- Training loops
- Data loading (DataLoader)
- GPU usage

### 1.2 Tools Setup

```bash
# Core tools
- Python 3.7+
- PaddlePaddle / PyTorch
- OpenCV (image processing)
- NumPy (numerical operations)
- Matplotlib (visualization)

# Optional but helpful
- Jupyter Notebook (experimentation)
- TensorBoard / VisualDL (training visualization)
- Git (version control)
```

---

## 🔍 Stage 2: Text Detection from Scratch

### 2.1 Understanding the Problem

**Goal**: Given an image, find bounding boxes around text regions.

**Input**: Image (H x W x 3)
**Output**: List of boxes [(x1, y1, x2, y2, x3, y3, x4, y4), ...]

### 2.2 Choose a Detection Algorithm

Start with **DB (Differentiable Binarization)** - it's what PP-OCR uses and is relatively simple.

**DB Algorithm Overview**:
1. Convert detection to a segmentation problem
2. Predict a probability map (each pixel = text likelihood)
3. Predict a threshold map (for adaptive binarization)
4. Post-process to get bounding boxes

### 2.3 Implementation Steps

#### Step 1: Data Preparation (Week 1)

**Task**: Collect and format training data

```python
# Data format for detection:
# dataset/
#   ├── images/
#   │   ├── img1.jpg
#   │   ├── img2.jpg
#   └── labels/
#       ├── img1.txt
#       ├── img2.txt

# Label format (img1.txt):
# Each line: x1,y1,x2,y2,x3,y3,x4,y4,transcription
100,50,200,50,200,80,100,80,Hello
300,100,400,100,400,130,300,130,World
```

**Where to get data**:
- Public datasets: ICDAR 2015, COCO-Text, MLT
- Synthetic data generators
- Your own labeled data

**Code to write**:
```python
class TextDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        # Load image paths and labels
        pass

    def __getitem__(self, idx):
        # 1. Load image
        # 2. Load labels (bounding boxes)
        # 3. Apply augmentations
        # 4. Generate ground truth maps for training
        #    - probability_map (text=1, background=0)
        #    - threshold_map (for DB algorithm)
        # 5. Return image, gt_maps
        pass
```

#### Step 2: Build the Model (Week 2)

**Architecture**: Backbone → Neck → Head

```python
# Simplified DB model structure

class DBNet(nn.Layer):
    def __init__(self):
        super().__init__()
        # 1. Backbone: Extract features
        self.backbone = MobileNetV3()  # or ResNet

        # 2. Neck: Fuse multi-scale features
        self.neck = FPN(in_channels=[...], out_channels=256)

        # 3. Head: Predict probability + threshold maps
        self.head = DBHead(in_channels=256)

    def forward(self, x):
        # Forward pass
        features = self.backbone(x)  # Multi-scale features
        features = self.neck(features)  # Fused features
        pred_maps = self.head(features)  # (prob_map, threshold_map)
        return pred_maps
```

**What to implement first**:
1. **Backbone**: Use pre-trained MobileNetV3 or ResNet (don't train from scratch!)
2. **Neck**: Implement simple FPN
3. **Head**: Two conv layers for probability and threshold maps

#### Step 3: Define Loss Function (Week 2)

**DB Loss** has three components:

```python
class DBLoss(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # pred: (prob_map, threshold_map)
        # gt: (gt_prob_map, gt_threshold_map, mask)

        # 1. Probability map loss (Binary Cross-Entropy)
        prob_loss = binary_cross_entropy(pred['prob'], gt['prob'], mask)

        # 2. Threshold map loss (L1 loss)
        thresh_loss = l1_loss(pred['thresh'], gt['thresh'], mask)

        # 3. Binary map loss (for hard binarization)
        binary_map = differentiable_binarization(pred['prob'], pred['thresh'])
        binary_loss = dice_loss(binary_map, gt['prob'], mask)

        # Total loss
        total_loss = prob_loss + thresh_loss + binary_loss
        return total_loss
```

#### Step 4: Training Loop (Week 2-3)

```python
# Simplified training loop

def train():
    model = DBNet()
    optimizer = Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=16)

    for epoch in range(num_epochs):
        for batch in dataloader:
            images, gt_maps = batch

            # Forward
            pred_maps = model(images)

            # Compute loss
            loss = db_loss(pred_maps, gt_maps)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate and save checkpoint
        if epoch % 10 == 0:
            evaluate(model, val_dataloader)
            save_checkpoint(model, f'epoch_{epoch}.pth')
```

#### Step 5: Post-Processing (Week 3)

**Convert probability map → bounding boxes**:

```python
def postprocess_db(prob_map, threshold_map):
    # 1. Binarize using threshold
    binary_map = (prob_map > threshold_map).astype('uint8')

    # 2. Find contours (connected regions)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Convert contours to boxes
    boxes = []
    for contour in contours:
        # Filter small regions
        if cv2.contourArea(contour) < min_area:
            continue

        # Get bounding box (or polygon)
        box = cv2.minAreaRect(contour)  # Rotated rectangle
        box = cv2.boxPoints(box)  # Convert to 4 points
        boxes.append(box)

    return boxes
```

#### Step 6: Evaluation (Week 3)

```python
def evaluate_detection(pred_boxes, gt_boxes):
    # Compute metrics:
    # - Precision: % of predicted boxes that are correct
    # - Recall: % of ground truth boxes that are detected
    # - F1-score: Harmonic mean of precision and recall

    # Use IoU (Intersection over Union) to match pred and gt boxes
    # IoU > 0.5 = match

    precision, recall, f1 = compute_metrics(pred_boxes, gt_boxes)
    return precision, recall, f1
```

### 2.4 Simplified Alternative (if starting from zero)

If building DB from scratch is too complex, start simpler:

**Option 1: Use EAST (Efficient and Accurate Scene Text) algorithm**
- Simpler than DB
- Still effective

**Option 2: Use YOLO for text detection**
- Treat text as objects
- Simpler training pipeline
- Lower accuracy but easier to start

---

## 📖 Stage 3: Text Recognition from Scratch

### 3.1 Understanding the Problem

**Goal**: Given a cropped text image, output the text string.

**Input**: Image (H x W x 3) - typically 32 x 100 or similar
**Output**: String (e.g., "Hello")

### 3.2 Choose a Recognition Algorithm

Start with **CRNN + CTC** (Classic and widely used):
- **CRNN**: Convolutional Recurrent Neural Network
- **CTC**: Connectionist Temporal Classification (loss function)

### 3.3 Implementation Steps

#### Step 1: Data Preparation (Week 1)

```python
# Data format for recognition:
# dataset/
#   ├── train/
#   │   ├── word_1.jpg    (contains "Hello")
#   │   ├── word_2.jpg    (contains "World")
#   └── train_labels.txt

# Label file format:
# train/word_1.jpg	Hello
# train/word_2.jpg	World
```

**Where to get data**:
- Synthetic data generators (highly recommended!)
- MJSynth, SynthText datasets
- Real-world datasets: IIIT5K, SVT, ICDAR

**Code**:
```python
class TextRecognitionDataset(Dataset):
    def __init__(self, image_dir, label_file, char_dict):
        # Load image paths and text labels
        self.char_dict = char_dict  # Character to index mapping

    def __getitem__(self, idx):
        # 1. Load image
        # 2. Resize to fixed height (e.g., 32 pixels), variable width
        # 3. Normalize
        # 4. Encode label to indices: "Hello" → [8, 5, 12, 12, 15]
        # 5. Return image, label_indices, label_length
        pass
```

#### Step 2: Build the Model (Week 2)

**CRNN Architecture**:

```python
class CRNN(nn.Layer):
    def __init__(self, num_classes):
        super().__init__()

        # 1. CNN: Extract visual features
        self.cnn = nn.Sequential(
            # Conv layers to extract features
            # Input: (B, 3, 32, W)
            # Output: (B, 512, 1, W/4)  # Height collapsed to 1
        )

        # 2. RNN: Model sequence
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )

        # 3. Output layer
        self.fc = nn.Linear(512, num_classes)  # 512 = 256 * 2 (bidirectional)

    def forward(self, x):
        # x: (B, 3, 32, W)

        # CNN
        conv_features = self.cnn(x)  # (B, 512, 1, W/4)

        # Reshape for RNN: (W/4, B, 512)
        conv_features = conv_features.squeeze(2).permute(2, 0, 1)

        # RNN
        rnn_output, _ = self.rnn(conv_features)  # (W/4, B, 512)

        # Output
        output = self.fc(rnn_output)  # (W/4, B, num_classes)

        return output
```

#### Step 3: CTC Loss (Week 2)

**What is CTC?**
- Allows training without character-level alignment
- Handles variable-length sequences

```python
# Using PaddlePaddle or PyTorch built-in CTC loss

loss_fn = nn.CTCLoss()

# During training:
preds = model(images)  # (T, B, num_classes) - T=time steps
log_probs = F.log_softmax(preds, dim=2)

# Calculate loss
loss = loss_fn(log_probs, labels, input_lengths, label_lengths)
```

#### Step 4: Training Loop (Week 2-3)

```python
def train():
    model = CRNN(num_classes=len(char_dict))
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for images, labels, label_lengths in dataloader:
            # Forward
            preds = model(images)
            log_probs = F.log_softmax(preds, dim=2)

            # Input lengths (number of time steps)
            input_lengths = [preds.shape[0]] * batch_size

            # CTC loss
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

#### Step 5: CTC Decoding (Week 3)

**Convert model output → text**:

```python
def ctc_decode(preds, char_dict):
    # preds: (T, B, num_classes)

    # 1. Get best character at each time step
    _, preds_idx = preds.max(2)  # (T, B)

    # 2. Remove duplicates and blank tokens
    # Example: ['-', 'H', 'H', 'e', '-', 'l', 'l', 'o', '-']
    #       →  ['H', 'e', 'l', 'l', 'o']
    #       →  "Hello"

    decoded = []
    for i in range(batch_size):
        char_list = []
        prev_char = None
        for t in range(T):
            char_idx = preds_idx[t, i]
            if char_idx != blank_idx and char_idx != prev_char:
                char_list.append(char_dict[char_idx])
            prev_char = char_idx
        decoded.append(''.join(char_list))

    return decoded
```

#### Step 6: Evaluation (Week 3)

```python
def evaluate_recognition(model, dataloader, char_dict):
    correct = 0
    total = 0

    for images, labels in dataloader:
        preds = model(images)
        decoded_preds = ctc_decode(preds, char_dict)
        decoded_labels = decode_labels(labels, char_dict)

        # Compare predictions with ground truth
        for pred, label in zip(decoded_preds, decoded_labels):
            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy
```

---

## 🔗 Stage 4: Complete OCR Pipeline

### 4.1 Combine Detection + Recognition

```python
class OCRSystem:
    def __init__(self):
        self.detector = DBNet()
        self.recognizer = CRNN()

        # Load trained weights
        self.detector.load_weights('detection_model.pth')
        self.recognizer.load_weights('recognition_model.pth')

    def ocr(self, image):
        # 1. Detection: Find text regions
        boxes = self.detector.predict(image)

        # 2. For each detected box:
        results = []
        for box in boxes:
            # Crop text region
            text_crop = crop_image(image, box)

            # Resize to recognition input size
            text_crop = resize(text_crop, (32, 100))

            # Recognition: Read text
            text = self.recognizer.predict(text_crop)

            results.append({
                'box': box,
                'text': text,
                'confidence': confidence
            })

        return results
```

### 4.2 Optional: Add Text Direction Classification

Some text might be upside-down. Add a classifier:

```python
class TextDirectionClassifier(nn.Layer):
    def __init__(self):
        # Simple CNN classifier
        # Input: Text crop
        # Output: 0° or 180°
        pass

# In OCR pipeline:
if use_angle_classifier:
    angle = self.classifier.predict(text_crop)
    if angle == 180:
        text_crop = rotate_180(text_crop)
```

---

## ⚡ Stage 5: Optimization & Improvements

### 5.1 Performance Optimization

1. **Model Compression**
   - Quantization (FP32 → INT8)
   - Pruning (remove less important weights)
   - Knowledge distillation (train small model from large model)

2. **Inference Speed**
   - Use TensorRT, ONNX Runtime
   - Batch processing
   - GPU utilization

3. **Accuracy Improvements**
   - More training data
   - Better augmentations
   - Ensemble models
   - Advanced architectures (SVTR, Transformers)

### 5.2 Multi-Language Support

```python
# Train separate recognition models for different languages
# Or use a unified character set

char_dict = load_char_dict('multilang_dict.txt')  # Contains en + zh + ja + ...
recognizer = CRNN(num_classes=len(char_dict))
```

### 5.3 Production Deployment

1. **API Service**
   ```python
   from flask import Flask, request

   app = Flask(__name__)
   ocr = OCRSystem()

   @app.route('/ocr', methods=['POST'])
   def ocr_api():
       image = request.files['image']
       result = ocr.ocr(image)
       return jsonify(result)
   ```

2. **Docker Deployment**
   ```dockerfile
   FROM python:3.8
   COPY . /app
   RUN pip install -r requirements.txt
   CMD ["python", "server.py"]
   ```

---

## 📊 Implementation Roadmap Summary

### Phase 1: Learning (2-4 weeks)
- [ ] Deep learning fundamentals
- [ ] CNN, RNN basics
- [ ] Framework tutorial (PaddlePaddle/PyTorch)

### Phase 2: Detection (2-3 weeks)
- [ ] Prepare detection dataset
- [ ] Implement DB model
- [ ] Training loop
- [ ] Post-processing
- [ ] Evaluation

### Phase 3: Recognition (2-3 weeks)
- [ ] Prepare recognition dataset
- [ ] Implement CRNN model
- [ ] CTC loss and training
- [ ] CTC decoding
- [ ] Evaluation

### Phase 4: Pipeline (1-2 weeks)
- [ ] Combine detection + recognition
- [ ] End-to-end testing
- [ ] Optimize inference speed

### Phase 5: Deployment (1-2 weeks)
- [ ] Model export
- [ ] API service
- [ ] Docker containerization
- [ ] Performance monitoring

---

## 🎓 Learning Resources

### Books
- "Deep Learning" by Ian Goodfellow
- "Programming Computer Vision with Python" by Jan Erik Solem

### Courses
- Fast.ai Practical Deep Learning
- Stanford CS231n (Computer Vision)
- Coursera Deep Learning Specialization

### Papers to Read
1. **Detection**:
   - "Real-time Scene Text Detection with Differentiable Binarization" (DB)
   - "EAST: An Efficient and Accurate Scene Text Detector"

2. **Recognition**:
   - "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (CRNN)
   - "SVTR: Scene Text Recognition with a Single Visual Model" (SVTR)

### Code References
- PaddleOCR GitHub (study the code!)
- EasyOCR (PyTorch-based, simpler)
- TrOCR (Transformer-based OCR from Microsoft)

---

## 🚀 Quick Start Alternative

**Not ready to build from scratch?** Start simpler:

### Option 1: Fine-tune Pre-trained Models
```python
# Use PP-OCR pre-trained models
from paddleocr import PaddleOCR

ocr = PaddleOCR()
# Fine-tune on your custom dataset
```

### Option 2: Use Existing Open-Source
- PaddleOCR (this project!)
- EasyOCR
- Tesseract OCR (traditional, non-DL)

### Option 3: Commercial APIs
- Google Vision API
- AWS Textract
- Azure Computer Vision

---

## 💡 Key Takeaways

1. **Start Simple**: Don't try to build PP-OCRv5 from day 1
2. **Use Pre-trained Models**: Transfer learning is your friend
3. **Iterate**: Build → Test → Improve → Repeat
4. **Data is King**: More data = better models
5. **Study Existing Code**: Learn from PaddleOCR's implementation

---

Next: [PaddleOCR-Only Implementation Guide](./05_PaddleOCR_Only_Guide.md) - If you want to use PaddleOCR but only for basic OCR (skip tables, structure, etc.)
