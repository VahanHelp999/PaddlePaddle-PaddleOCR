# Deployment Options: Taking Your Model to Production

This guide covers different ways to deploy PaddleOCR models in production environments.

## 🎯 Deployment Overview

```
Development → Export → Deploy → Monitor
    🔧          📦        🚀        📊
```

PaddleOCR supports multiple deployment options:

| Deployment Type | Language | Platform | Use Case |
|----------------|----------|----------|----------|
| Python API | Python | Any | Prototyping, server |
| C++ Inference | C++ | Linux, Windows | High performance |
| Mobile (Lite) | Java/Swift | iOS, Android | Mobile apps |
| ONNX Runtime | Any | Cross-platform | Framework flexibility |
| HTTP Serving | Python | Cloud/Server | API service |
| Docker | Any | Cloud | Containerized deployment |

---

## 1. 🐍 Python API Deployment

### Option A: Using PaddleOCR Package (Easiest)

**Installation**:
```bash
pip install paddleocr
```

**Usage**:
```python
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Initialize (downloads models automatically)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Run OCR
result = ocr.ocr('image.jpg', cls=True)

# Print results
for line in result[0]:
    box = line[0]
    text, confidence = line[1]
    print(f"Text: {text}, Confidence: {confidence:.2f}")

# Visualize
image = Image.open('image.jpg')
boxes = [line[0] for line in result[0]]
texts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]
im_show = draw_ocr(image, boxes, texts, scores)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

**Pros**:
- ✅ Easiest to use
- ✅ Pre-trained models included
- ✅ Multi-language support

**Cons**:
- ❌ Slower than C++
- ❌ Python dependency

---

### Option B: Using Custom Trained Models

```python
from tools.infer.predict_system import TextSystem

# Initialize with custom models
args = {
    'det_model_dir': './inference/my_det_model/',
    'rec_model_dir': './inference/my_rec_model/',
    'rec_char_dict_path': './my_dict.txt',
    'use_angle_cls': False,
    'use_gpu': True,
}

text_sys = TextSystem(args)

# Run OCR
import cv2
img = cv2.imread('test.jpg')
result = text_sys(img)

for box, (text, score) in result:
    print(f"Text: {text}, Score: {score:.2f}")
```

---

## 2. ⚡ C++ Inference (High Performance)

### Why C++?

**Performance comparison** (on same hardware):
- Python: ~100ms per image
- C++: ~30ms per image (3x faster)

**Use cases**:
- High-throughput servers
- Real-time applications
- Embedded systems

### Setup

#### Prerequisites
```bash
# Install PaddlePaddle C++ library
# Download from: https://paddleinference.paddlepaddle.org.cn/

# Install OpenCV
apt-get install libopencv-dev

# Install other dependencies
apt-get install cmake build-essential
```

#### Build
```bash
cd deploy/cpp_infer
mkdir build
cd build

cmake .. \
    -DPADDLE_LIB=/path/to/paddle_inference \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DUSE_TENSORRT=OFF \
    -DOPENCV_DIR=/path/to/opencv \
    -DCUDNN_LIB=/path/to/cudnn \
    -DCUDA_LIB=/path/to/cuda

make -j
```

#### Usage
```bash
# Run detection + recognition
./build/ppocr \
    --det_model_dir=./inference/det_model/ \
    --rec_model_dir=./inference/rec_model/ \
    --image_dir=./test.jpg \
    --use_angle_cls=true \
    --cls_model_dir=./inference/cls_model/
```

### C++ API Example

```cpp
#include "ocr_det.h"
#include "ocr_rec.h"
#include "ocr_cls.h"

int main() {
    // Initialize models
    OCRDet det("./inference/det_model/");
    OCRRec rec("./inference/rec_model/", "./dict.txt");
    OCRCls cls("./inference/cls_model/");

    // Load image
    cv::Mat img = cv::imread("test.jpg");

    // Detection
    std::vector<std::vector<float>> boxes = det.Run(img);

    // For each detected box
    for (auto& box : boxes) {
        // Crop
        cv::Mat crop = GetCropImage(img, box);

        // Classification (optional)
        int angle = cls.Run(crop);
        if (angle == 180) {
            cv::rotate(crop, crop, cv::ROTATE_180);
        }

        // Recognition
        std::string text;
        float score;
        rec.Run(crop, text, score);

        std::cout << "Text: " << text << ", Score: " << score << std::endl;
    }

    return 0;
}
```

**Pros**:
- ✅ Very fast (3x faster than Python)
- ✅ Low memory overhead
- ✅ Suitable for production

**Cons**:
- ❌ More complex setup
- ❌ Requires C++ knowledge

---

## 3. 📱 Mobile Deployment (Paddle Lite)

### iOS and Android Apps

**Use cases**:
- Mobile OCR apps
- On-device processing (privacy)
- Offline OCR

### Setup for Android

#### 1. Export Lite Model

```bash
pip install paddlelite

# Convert model to Lite format
paddle_lite_opt \
    --model_file=./inference/det_model/inference.pdmodel \
    --param_file=./inference/det_model/inference.pdiparams \
    --optimize_out=./inference/det_model_lite \
    --optimize_out_type=naive_buffer \
    --valid_targets=arm
```

#### 2. Integrate into Android

```java
// Load models
PaddlePredictor detPredictor = PaddlePredictor.createPaddlePredictor(detConfig);
PaddlePredictor recPredictor = PaddlePredictor.createPaddlePredictor(recConfig);

// Prepare input
Bitmap bitmap = BitmapFactory.decodeFile("image.jpg");
float[] inputData = preprocessImage(bitmap);

// Run detection
detPredictor.setInput(inputData);
detPredictor.run();
float[] detOutput = detPredictor.getOutput();

// Process detection results
List<Box> boxes = postprocessDet(detOutput);

// Run recognition on each box
for (Box box : boxes) {
    Bitmap crop = cropImage(bitmap, box);
    float[] recInput = preprocessImage(crop);
    recPredictor.setInput(recInput);
    recPredictor.run();
    float[] recOutput = recPredictor.getOutput();
    String text = decodeText(recOutput);
    System.out.println("Text: " + text);
}
```

#### 3. Example Projects

```bash
# Android demo
cd deploy/android_demo

# iOS demo
cd deploy/ios_demo
```

**Model sizes** (Lite optimized):
- Detection: ~2-3 MB
- Recognition: ~4-5 MB
- Total: ~6-8 MB

**Performance**:
- Detection: ~200ms (mobile CPU)
- Recognition per word: ~50ms
- Total for typical document: ~1-2 seconds

**Pros**:
- ✅ Runs on-device (privacy)
- ✅ Offline capability
- ✅ Small model size

**Cons**:
- ❌ Slower than server
- ❌ Limited accuracy (mobile models)

---

## 4. 🌐 HTTP Serving

### Option A: PaddleHub Serving

**Setup**:
```bash
# Install PaddleHub
pip install paddlehub

# Install OCR service modules
hub install deploy/hubserving/ocr_system/
```

**Start Service**:
```bash
hub serving start -m ocr_system -p 8866
```

**Client Usage**:
```python
import requests
import base64

# Encode image
with open('test.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf8')

# Send request
response = requests.post(
    'http://localhost:8866/predict/ocr_system',
    json={'images': [image_data]}
)

# Parse results
result = response.json()
for line in result['results'][0]:
    text = line['text']
    confidence = line['confidence']
    print(f"Text: {text}, Confidence: {confidence}")
```

---

### Option B: Custom Flask/FastAPI Server

**Flask Example**:
```python
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import base64
import cv2
import numpy as np

app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    # Get image from request
    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run OCR
    result = ocr.ocr(img, cls=True)

    # Format response
    response = []
    for line in result[0]:
        response.append({
            'box': line[0],
            'text': line[1][0],
            'confidence': line[1][1]
        })

    return jsonify({'results': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**FastAPI Example** (faster, async):
```python
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
import cv2
import numpy as np

app = FastAPI()
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.post('/ocr')
async def ocr_endpoint(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run OCR
    result = ocr.ocr(img, cls=True)

    # Format response
    response = []
    for line in result[0]:
        response.append({
            'box': line[0],
            'text': line[1][0],
            'confidence': line[1][1]
        })

    return {'results': response}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

**Load Balancing** (multiple workers):
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using uvicorn (FastAPI)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 5. 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM paddlepaddle/paddle:2.5.1

# Install dependencies
RUN pip install paddleocr flask

# Copy application
COPY app.py /app/app.py
COPY models/ /app/models/

WORKDIR /app

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "app.py"]
```

### Build and Run

```bash
# Build image
docker build -t paddleocr-server:latest .

# Run container
docker run -d -p 8000:8000 --name ocr-server paddleocr-server:latest

# Test
curl -X POST http://localhost:8000/ocr -F "file=@test.jpg"
```

### Docker Compose (with GPU)

```yaml
version: '3'
services:
  ocr-server:
    image: paddleocr-server:latest
    ports:
      - "8000:8000"
    runtime: nvidia  # GPU support
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
    restart: always
```

---

## 6. 🔄 ONNX Export (Cross-Platform)

### Why ONNX?

- Use models in TensorFlow, PyTorch, etc.
- Deploy with ONNXRuntime (fast, cross-platform)
- Support for more hardware backends

### Export to ONNX

```bash
# Install paddle2onnx
pip install paddle2onnx

# Export detection model
paddle2onnx \
    --model_dir ./inference/det_model/ \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./inference/det_model.onnx \
    --opset_version 11

# Export recognition model
paddle2onnx \
    --model_dir ./inference/rec_model/ \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./inference/rec_model.onnx \
    --opset_version 11
```

### Inference with ONNXRuntime

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
sess = ort.InferenceSession('./inference/det_model.onnx')

# Prepare input
img = cv2.imread('test.jpg')
img = cv2.resize(img, (640, 640))
img = img.transpose((2, 0, 1))  # HWC -> CHW
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

# Run inference
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: img})

# Process output
result = postprocess(output)
```

---

## 7. ⚡ Model Optimization

### Quantization (FP32 → INT8)

**Why?**
- 4x smaller model size
- 2-4x faster inference
- Minimal accuracy loss (<1%)

**Using PaddleSlim**:
```python
from paddleslim.quant import quant_post_static

# Quantize model
quant_post_static(
    model_dir='./inference/det_model/',
    quantize_model_path='./inference/det_model_quant/',
    sample_generator=sample_generator,
    batch_size=32,
    batch_nums=10
)
```

### Pruning (Remove Redundant Weights)

```python
from paddleslim.prune import Pruner

pruner = Pruner()
pruned_model = pruner.prune(
    model,
    ratios=[0.3] * 100  # Prune 30% of each layer
)
```

### Knowledge Distillation

Train small model from large model (see training docs).

**Comparison**:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| Original | 100 MB | 100ms | 90% |
| Quantized | 25 MB | 40ms | 89.5% |
| Pruned | 50 MB | 60ms | 88% |
| Distilled | 30 MB | 50ms | 88.5% |

---

## 8. 📊 Production Best Practices

### 1. **Model Versioning**

```
models/
├── v1.0/
│   ├── det_model/
│   └── rec_model/
├── v1.1/
│   ├── det_model/
│   └── rec_model/
└── latest -> v1.1/
```

### 2. **Monitoring**

Track metrics:
- **Inference time** (P50, P95, P99)
- **Throughput** (requests per second)
- **Error rate**
- **Resource usage** (CPU, GPU, memory)

```python
import time
import logging

def ocr_with_logging(img):
    start = time.time()
    try:
        result = ocr.ocr(img)
        latency = time.time() - start
        logging.info(f"OCR success, latency: {latency:.2f}s")
        return result
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        raise
```

### 3. **Caching**

Cache results for repeated requests:
```python
import hashlib
import pickle

cache = {}

def ocr_with_cache(img):
    # Compute image hash
    img_hash = hashlib.md5(img.tobytes()).hexdigest()

    # Check cache
    if img_hash in cache:
        return cache[img_hash]

    # Run OCR
    result = ocr.ocr(img)

    # Cache result
    cache[img_hash] = result
    return result
```

### 4. **Batch Processing**

Process multiple images together:
```python
def batch_ocr(images, batch_size=8):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = ocr.ocr(batch)
        results.extend(batch_results)
    return results
```

### 5. **Error Handling**

```python
def robust_ocr(img_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            img = cv2.imread(img_path)
            result = ocr.ocr(img)
            return result
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
```

### 6. **Resource Management**

```python
import threading

# Thread-safe OCR
lock = threading.Lock()

def thread_safe_ocr(img):
    with lock:
        return ocr.ocr(img)

# Or use process pool
from multiprocessing import Pool

def init_worker():
    global ocr
    ocr = PaddleOCR()

def ocr_worker(img_path):
    img = cv2.imread(img_path)
    return ocr.ocr(img)

with Pool(processes=4, initializer=init_worker) as pool:
    results = pool.map(ocr_worker, image_paths)
```

---

## 9. 🚀 Deployment Checklist

### Before Deployment
- [ ] Models exported and tested
- [ ] Inference speed acceptable
- [ ] Accuracy validated on test set
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Resource limits set

### During Deployment
- [ ] Monitor inference latency
- [ ] Monitor error rates
- [ ] Monitor resource usage
- [ ] Have rollback plan
- [ ] Test with real traffic

### After Deployment
- [ ] Analyze performance metrics
- [ ] Collect edge cases
- [ ] Plan model improvements
- [ ] Document issues and solutions

---

## 🎯 Deployment Selection Guide

```
Need fastest inference? → C++ Inference
Need mobile app? → Paddle Lite
Need API service? → HTTP Serving + Docker
Need cross-platform? → ONNX Runtime
Need quick prototype? → Python API
Need scalability? → Docker + Kubernetes
```

---

**Summary**: PaddleOCR supports diverse deployment options from Python APIs to C++ inference, mobile apps, and cloud services. Choose based on your performance requirements, platform, and technical constraints.
