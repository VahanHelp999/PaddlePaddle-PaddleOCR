# FAQ: Frequently Asked Questions

Common questions and answers about PaddleOCR, organized by topic.

---

## 📚 General Questions

### Q1: What is PaddleOCR?

**A**: PaddleOCR is an open-source OCR (Optical Character Recognition) toolkit developed by Baidu. It includes:
- Text detection models (find text in images)
- Text recognition models (read what the text says)
- Support for 80+ languages
- Pre-trained models ready to use
- Tools for training custom models

See [01_Project_Overview.md](./01_Project_Overview.md) for details.

---

### Q2: PaddleOCR vs Tesseract vs EasyOCR - Which is better?

**A**: Comparison:

| Feature | PaddleOCR | Tesseract | EasyOCR |
|---------|-----------|-----------|---------|
| Accuracy | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Very Good |
| Speed | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡ Slow |
| Languages | 80+ | 100+ | 80+ |
| Mobile support | ✅ Yes | ❌ No | ❌ No |
| Training tools | ✅ Yes | ⚠️ Complex | ⚠️ Limited |
| Deep Learning | ✅ Yes | ❌ No (traditional) | ✅ Yes |

**Recommendation**:
- **PaddleOCR**: Best overall (accuracy + speed + features)
- **Tesseract**: Simple documents, legacy systems
- **EasyOCR**: If you prefer PyTorch

---

### Q3: What version should I use? (v3, v4, v5)

**A**:

| Version | Release | Recommendation |
|---------|---------|----------------|
| PP-OCRv5 | 2024 | ✅ **Use this** (latest, best) |
| PP-OCRv4 | 2023 | ✅ Stable, excellent |
| PP-OCRv3 | 2022 | ⚠️ Older but still good |
| PP-OCRv2 | 2021 | ❌ Outdated |
| PP-OCRv1 | 2020 | ❌ Outdated |

**Guideline**: Always use the latest stable version (v4 or v5) unless you have specific constraints.

---

## 🚀 Getting Started

### Q4: How do I install PaddleOCR?

**A**:

**Quick install** (inference only):
```bash
pip install paddleocr
```

**Full install** (training + inference):
```bash
# Clone repository
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# Install dependencies
pip install -r requirements.txt

# Install PaddlePaddle (CPU)
pip install paddlepaddle

# Install PaddlePaddle (GPU)
pip install paddlepaddle-gpu
```

---

### Q5: How do I use PaddleOCR for the first time?

**A**:

```python
from paddleocr import PaddleOCR

# Initialize (first time downloads models ~40MB)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Run OCR
result = ocr.ocr('image.jpg', cls=True)

# Print results
for line in result[0]:
    text = line[1][0]
    confidence = line[1][1]
    print(f"{text} (confidence: {confidence:.2f})")
```

**Note**: First run downloads pre-trained models automatically.

---

### Q6: Which languages are supported?

**A**: 80+ languages including:

**Popular**:
- English (`lang='en'`)
- Chinese (`lang='ch'`)
- Japanese (`lang='japan'`)
- Korean (`lang='korean'`)
- French (`lang='french'`)
- German (`lang='german'`)
- Spanish (`lang='spanish'`)

**Others**:
- Arabic (`lang='arabic'`)
- Hindi (`lang='hi'`)
- Russian (`lang='ru'`)
- Thai (`lang='th'`)
- Vietnamese (`lang='vi'`)
- And 65+ more...

**Full list**: See `ppocr/utils/dict/` directory for all supported languages.

---

## 🔧 Training & Customization

### Q7: Can I train my own models?

**A**: Yes! PaddleOCR is designed for custom training.

**Steps**:
1. Prepare your dataset (detection or recognition)
2. Create/modify a config file
3. Download pre-trained weights
4. Train using `python tools/train.py -c config.yml`

See [08_Training_Pipeline.md](./08_Training_Pipeline.md) for complete guide.

---

### Q8: How much data do I need to train a model?

**A**:

| Scenario | Detection | Recognition |
|----------|-----------|-------------|
| Fine-tuning pre-trained model | 500-1000 images | 5,000-10,000 words |
| Training from scratch | 10,000+ images | 100,000+ words |
| Small improvements | 100-500 images | 1,000-5,000 words |

**Tips**:
- **Always fine-tune** rather than training from scratch
- Use data augmentation to artificially increase data
- Synthetic data (e.g., StyleText) can help

---

### Q9: Training is too slow. How can I speed it up?

**A**:

**Solutions**:
1. **Use GPU**:
   ```yaml
   Global:
     use_gpu: true
   ```

2. **Increase batch size** (if GPU memory allows):
   ```yaml
   Train:
     loader:
       batch_size_per_card: 32  # Increase from 16
   ```

3. **Use mixed precision training**:
   ```yaml
   Global:
     use_amp: true
   ```

4. **Use multiple GPUs**:
   ```bash
   python -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c config.yml
   ```

5. **Reduce image size** (during prototyping):
   ```yaml
   Train:
     dataset:
       transforms:
         - RandomCrop:
             size: [480, 480]  # Reduce from [640, 640]
   ```

---

### Q10: My model overfits. What should I do?

**A**:

**Symptoms**:
- Training loss decreases
- Validation loss increases or plateaus
- Training accuracy high, validation accuracy low

**Solutions**:
1. **More training data** (best solution)
2. **Stronger data augmentation**:
   ```yaml
   Train:
     dataset:
       transforms:
         - IaaAugment:  # Add more augmentations
         - ColorJitter:
         - RandomBlur:
   ```

3. **Increase regularization**:
   ```yaml
   Optimizer:
     regularizer:
       name: L2
       factor: 0.0001  # Increase from 0.00001
   ```

4. **Use smaller model**:
   ```yaml
   Architecture:
     Backbone:
       scale: 0.5  # Reduce from 1.0
   ```

5. **Early stopping**: Stop training when validation metric stops improving

---

## 🐛 Troubleshooting

### Q11: Error: "Out of Memory (OOM)"

**A**:

**Causes**: Batch size too large for GPU memory

**Solutions**:
1. **Reduce batch size**:
   ```yaml
   Train:
     loader:
       batch_size_per_card: 4  # Reduce from 16
   ```

2. **Reduce image size**:
   ```yaml
   - RandomCrop:
       size: [480, 480]  # Reduce from [640, 640]
   ```

3. **Use smaller model**:
   ```yaml
   Architecture:
     Backbone:
       scale: 0.5
   ```

4. **Enable gradient accumulation**:
   ```yaml
   Global:
     accumulation_steps: 4
   ```

---

### Q12: Loss is NaN during training

**A**:

**Causes**:
- Learning rate too high
- Gradient explosion
- Incorrect data preprocessing

**Solutions**:
1. **Lower learning rate**:
   ```yaml
   Optimizer:
     lr:
       learning_rate: 0.0001  # Reduce from 0.001
   ```

2. **Add gradient clipping**:
   ```yaml
   Optimizer:
     clip_norm: 10.0
   ```

3. **Check data**:
   ```python
   # Visualize a batch
   for batch in dataloader:
       images = batch['image']
       print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
       # Should be normalized (e.g., [0, 1] or [-1, 1])
   ```

4. **Use mixed precision carefully**:
   ```yaml
   Global:
     use_amp: false  # Disable if causing issues
   ```

---

### Q13: Model accuracy is very low after training

**A**:

**Possible causes**:

1. **Wrong labels**: Double-check your label file format
   ```python
   # Verify labels
   with open('train_list.txt', 'r') as f:
       for line in f:
           print(line)  # Check format
   ```

2. **Model not loading pre-trained weights**:
   ```yaml
   Global:
     pretrained_model: ./pretrain_models/best_accuracy  # Verify path exists
   ```

3. **Wrong config parameters**:
   - Verify `Architecture` matches your task
   - Check `Loss` function is correct

4. **Insufficient training**:
   - Train for more epochs
   - Check if loss is still decreasing

5. **Bad data quality**:
   - Verify images load correctly
   - Check augmentations aren't too strong

---

### Q14: Inference is too slow

**A**:

**Detection + Recognition system** can be slow. Optimize:

1. **Use GPU**:
   ```python
   ocr = PaddleOCR(use_gpu=True)
   ```

2. **Use mobile models**:
   ```python
   ocr = PaddleOCR(
       det_model_dir='path/to/mobile_det_model',
       rec_model_dir='path/to/mobile_rec_model'
   )
   ```

3. **Disable angle classifier** (if not needed):
   ```python
   ocr = PaddleOCR(use_angle_cls=False)
   ```

4. **Use C++ inference** (3x faster than Python):
   See [09_Deployment_Options.md](./09_Deployment_Options.md)

5. **Quantize models** (INT8):
   ```bash
   paddle_lite_opt --optimize_out_type=naive_buffer
   ```

6. **Batch processing**:
   ```python
   results = ocr.ocr([img1, img2, img3])  # Process multiple images
   ```

---

### Q15: Detection works but recognition is wrong

**A**:

**Possible causes**:

1. **Wrong character dictionary**:
   ```python
   ocr = PaddleOCR(
       rec_char_dict_path='path/to/correct_dict.txt',
       lang='en'  # Or your language
   )
   ```

2. **Detection boxes too tight/loose**:
   ```yaml
   PostProcess:
     unclip_ratio: 2.0  # Increase from 1.5 to expand boxes
   ```

3. **Text is rotated**:
   ```python
   ocr = PaddleOCR(use_angle_cls=True)  # Enable angle classifier
   ```

4. **Recognition model not suitable**:
   - Try different recognition model
   - Fine-tune on your specific font/style

---

## 🌍 Multi-Language & Special Cases

### Q16: How do I OCR Chinese + English mixed text?

**A**:

```python
# Use Chinese model (includes English)
ocr = PaddleOCR(lang='ch')

result = ocr.ocr('mixed_text.jpg')
```

**Note**: Chinese model (`lang='ch'`) handles both Chinese and English characters.

---

### Q17: How do I OCR vertical text?

**A**:

**Option 1**: Use angle classifier
```python
ocr = PaddleOCR(use_angle_cls=True)
result = ocr.ocr('vertical_text.jpg', cls=True)
```

**Option 2**: Pre-rotate image
```python
import cv2

img = cv2.imread('vertical_text.jpg')
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
result = ocr.ocr(img)
```

**Option 3**: Train vertical text detection model (advanced)

---

### Q18: How do I OCR curved or perspective-distorted text?

**A**:

1. **Use TPS (Thin Plate Spline) transformation**:
   ```yaml
   Architecture:
     Transform:
       name: TPS
       num_fiducial: 20
   ```

2. **Use advanced detection models**:
   - FCE (Fourier Contour Embedding) for curved text
   - Config: `configs/det/det_r50_vd_fce.yml`

3. **Pre-process with perspective correction**:
   ```python
   # Use OpenCV to correct perspective before OCR
   import cv2
   corrected = cv2.warpPerspective(img, matrix, (width, height))
   result = ocr.ocr(corrected)
   ```

---

### Q19: How do I OCR handwritten text?

**A**:

**Important**: PaddleOCR's default models are trained on printed text. For handwriting:

1. **Collect handwritten dataset** (e.g., IAM, RIMES)
2. **Fine-tune recognition model**:
   ```bash
   python tools/train.py -c configs/rec/rec_handwriting.yml
   ```
3. **Consider specialized handwriting recognition models** (outside PaddleOCR)

**Note**: Handwriting is much harder than printed text. Expect lower accuracy.

---

## 📱 Deployment Questions

### Q20: Can I use PaddleOCR in a mobile app?

**A**: Yes! Use **Paddle Lite**.

**Steps**:
1. Export model to Lite format:
   ```bash
   paddle_lite_opt --model_file=inference.pdmodel --optimize_out=model_lite
   ```

2. Integrate into Android/iOS:
   - Android: See `deploy/android_demo/`
   - iOS: See `deploy/ios_demo/`

**Model sizes**: ~6-8 MB total (detection + recognition)

See [09_Deployment_Options.md](./09_Deployment_Options.md) for details.

---

### Q21: How do I deploy PaddleOCR as a web service?

**A**:

**Option 1**: PaddleHub Serving (easiest)
```bash
hub serving start -m ocr_system -p 8866
```

**Option 2**: Flask/FastAPI
```python
from flask import Flask, request
from paddleocr import PaddleOCR

app = Flask(__name__)
ocr = PaddleOCR()

@app.route('/ocr', methods=['POST'])
def ocr_api():
    file = request.files['image']
    img = file.read()
    result = ocr.ocr(img)
    return {'result': result}

app.run(host='0.0.0.0', port=8000)
```

**Option 3**: Docker
```bash
docker build -t paddleocr-server .
docker run -p 8000:8000 paddleocr-server
```

See [09_Deployment_Options.md](./09_Deployment_Options.md) for complete guide.

---

### Q22: Can I convert PaddleOCR models to ONNX?

**A**: Yes!

```bash
# Install paddle2onnx
pip install paddle2onnx

# Convert model
paddle2onnx \
    --model_dir ./inference/det_model/ \
    --save_file ./inference/det_model.onnx \
    --opset_version 11
```

**Use with ONNXRuntime**:
```python
import onnxruntime as ort

sess = ort.InferenceSession('det_model.onnx')
output = sess.run(None, {input_name: input_data})
```

---

## 💡 Best Practices

### Q23: What's the recommended workflow for a new OCR project?

**A**:

```
1. Use pre-trained models first
   ↓
2. Evaluate on your data
   ↓
3. If accuracy < 80%:
   - Collect 500-1000 labeled images
   - Fine-tune models
   ↓
4. If accuracy < 90%:
   - Collect more data (2000-5000 images)
   - Train with better augmentation
   ↓
5. Optimize for deployment
   - Quantization
   - C++ inference
   ↓
6. Deploy and monitor
```

---

### Q24: Should I use detection+recognition or end-to-end models?

**A**:

**Detection + Recognition (Recommended)** ✅:
- **Pros**: Higher accuracy, more flexible, easier to debug
- **Cons**: Two-stage (slightly slower)

**End-to-End** ⚠️:
- **Pros**: Single model, potentially faster
- **Cons**: Lower accuracy, less flexible, harder to improve

**Recommendation**: Use detection + recognition for production systems.

---

### Q25: How do I handle low-quality images?

**A**:

**Pre-processing**:
1. **Resize** to higher resolution
2. **Denoise**:
   ```python
   import cv2
   denoised = cv2.fastNlMeansDenoisingColored(img)
   ```
3. **Sharpen**:
   ```python
   kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
   sharpened = cv2.filter2D(img, -1, kernel)
   ```
4. **Binarization** (for documents):
   ```python
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
   ```

**Or**: Train models on low-quality augmented data.

---

## 🔍 Advanced Topics

### Q26: What is knowledge distillation and should I use it?

**A**:

**Knowledge Distillation**: Training a small model (student) to mimic a large model (teacher).

**Benefits**:
- Small model with large model accuracy
- Faster inference
- Smaller file size

**When to use**:
- You have a large accurate model
- You need faster inference (mobile/edge)
- You're willing to accept 1-2% accuracy drop

**How**:
```yaml
Architecture:
  model_type: distillation
  algorithm: Distillation
  Models:
    Teacher:
      pretrained: ./large_model
      freeze_params: true
    Student:
      # Student config
```

---

### Q27: Can PaddleOCR detect and recognize tables?

**A**: Yes, but that's **PaddleStructure** (not core PaddleOCR).

**For tables**:
```python
from ppstructure.table import TableSystem

table_sys = TableSystem()
result = table_sys('table_image.jpg')
```

**Output**: Excel file or structured HTML.

See `ppstructure/table/` for details.

**Note**: This guide focuses on basic OCR (text only). For tables, see PaddleStructure docs.

---

### Q28: How do I contribute to PaddleOCR?

**A**:

1. **Report bugs**: GitHub Issues
2. **Suggest features**: GitHub Issues
3. **Fix bugs**: Submit Pull Request
4. **Add models**: Train and share pre-trained weights
5. **Improve docs**: Submit PR to `docs/`

**GitHub**: https://github.com/PaddlePaddle/PaddleOCR

---

## 🆘 Getting Help

### Q29: Where can I get help?

**A**:

1. **Read docs**:
   - This documentation folder
   - Official docs: https://github.com/PaddlePaddle/PaddleOCR/tree/main/doc

2. **Search existing issues**:
   - GitHub Issues: https://github.com/PaddlePaddle/PaddleOCR/issues

3. **Ask questions**:
   - Create new GitHub Issue
   - WeChat group (see repo README)

4. **Commercial support**:
   - Contact Baidu PaddlePaddle team

---

### Q30: I found a bug. What should I do?

**A**:

1. **Verify it's a bug** (not user error)
2. **Create minimal reproducible example**:
   ```python
   from paddleocr import PaddleOCR
   ocr = PaddleOCR()
   result = ocr.ocr('test.jpg')  # Bug happens here
   ```
3. **Report on GitHub Issues** with:
   - PaddleOCR version
   - Python version
   - OS (Linux/Windows/Mac)
   - Error message
   - Minimal code to reproduce
   - Expected vs actual behavior

---

## 📊 Quick Reference

### Common Commands

```bash
# Training
python tools/train.py -c config.yml

# Evaluation
python tools/eval.py -c config.yml -o Global.checkpoints=./output/best

# Export
python tools/export_model.py -c config.yml -o Global.save_inference_dir=./inference/

# Inference (detection)
python tools/infer_det.py --image_dir=test.jpg

# Inference (recognition)
python tools/infer_rec.py --image_dir=word.jpg
```

### Config Quick Reference

```yaml
Global:                  # Training settings
  epoch_num: 500         # Number of epochs
  save_model_dir: ./output/

Architecture:            # Model structure
  Backbone: ...          # Feature extraction
  Neck: ...              # Feature refinement
  Head: ...              # Prediction

Optimizer:               # Training optimization
  lr: 0.001              # Learning rate

Train:                   # Training data
  dataset:
    data_dir: ./data/
  loader:
    batch_size_per_card: 16
```

---

**Need more help?** Check the other documentation files or create a GitHub Issue!
