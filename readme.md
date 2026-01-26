# 🧠 CNN Image Classification Model

A Convolutional Neural Network (CNN) built and trained from scratch for image classification, with a strong focus on understanding training dynamics, controlling overfitting, and achieving reproducible results.

This project emphasizes the **model development process itself** rather than deployment or UI, highlighting how to design, train, regularize, and evaluate deep learning models effectively.

---

## 🎯 Project Objectives

- Design a CNN architecture suitable for small-scale image datasets  
- Understand and control overfitting  
- Apply modern regularization techniques  
- Create reproducible experiments  
- Analyze model behavior rather than maximize raw accuracy  

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  

---

## 📂 Project Structure
cnn-image-classifier/

├── model/
│ ├── train.py # Training pipeline
│ ├── model.py # CNN architecture definition
│ ├── model.keras # Saved trained model
│
├── requirements.txt
├── README.md



---

## 🧪 Model Architecture

Input (32×32×3)
→ Conv(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Conv(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Conv(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
→ Flatten
→ Dense(64) → BatchNorm → ReLU → Dropout(0.5)
→ Dense(10) → Softmax



Design principles:
- Progressive increase in feature depth: `32 → 64 → 128`  
- Stronger regularization in dense layers  
- Batch Normalization for training stability  
- Dropout to prevent memorization  

---

## 📊 Dataset

- CIFAR-10  
- Subset size: **9,000 images**  
- Image resolution: **32×32 RGB**  
- Number of classes: **10**  

---

## 🔄 Data Augmentation

To improve generalization and reduce memorization:

- Rotation  
- Width & height shifts  
- Horizontal flip  
- Zoom  
- Shear  
- Nearest fill mode  

Purpose:
> Encourage invariant feature learning rather than pixel-level memorization.

---

## ⚙️ Training Configuration

| Parameter | Value |
|--------|------|
| Optimizer | Adam |
| Learning Rate | 0.005 |
| Batch Size | 32 |
| Epochs | Up to 40 |
| Early Stopping | `patience=5`, `restore_best_weights=True` |
| Seed | 42 |

Random seeds are fixed for:
- Weight initialization  
- Data augmentation  
- Batch ordering  

Ensuring **reproducibility across runs**.

---

## 📈 Performance

Balanced training point:

| Metric | Value |
|------|------|
| Training Accuracy | ~70% |
| Validation Accuracy | ~52% |
| Validation Loss | ~1.39 |

This reflects:
- Reduced overfitting  
- Stable convergence  
- Controlled model capacity  
- Meaningful generalization  

---

## 🧠 Key Learning Outcomes

- Designed and trained a CNN from scratch  
- Diagnosed and corrected overfitting  
- Implemented layered regularization strategies  
- Controlled experimental randomness  
- Balanced dataset size vs model capacity  
- Developed intuition for hyperparameter sensitivity  
- Understood dataset–domain limitations  

---

## 🧑‍💻 Installation

1. Clone the repository:

git clone https://github.com/yourusername/cnn-image-classifier.git
cd cnn-image-classifier

2. Create and activate a virtual environment:

Windows
```Windows
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:
```Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🚀 Training the Model

To train the CNN from scratch:

cd model
python main.py


This will:
- Load the CIFAR-10 dataset  
- Apply preprocessing and data augmentation  
- Train the CNN with early stopping  
- Save the best-performing model as `model.keras`

---

## 🔍 Using the Trained Model

To load and run inference on an image:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/model.keras")

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

img = image.load_img("example.jpg", target_size=(32, 32))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
idx = pred.argmax()
confidence = pred.max()

print("Prediction:", class_names[idx])
print("Confidence:", confidence)
```
## ⚠️ Note:
The model is trained on CIFAR-10 (32×32 images). Predictions on real-world images may be unreliable due to domain differences.

## 👤 Author

Built by `<Your_Name>`  
Machine Learning Engineer | Deep Learning Enthusiast
