<div align="center">

# 🎭 Facial Emotion Recognition System

### *Decoding Human Emotions Through Computer Vision & Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![DeepFace](https://img.shields.io/badge/DeepFace-Enabled-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/serengil/deepface)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>

<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/palette/macchiato.png" width="600"/>

<br/>

**Presented by Advanced Signal and Image Processing Lab (ASIP Lab)**  
*Dept. of Data Science and Engineering, IISER Bhopal*

---

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Models](#-model-comparison) • [Results](#-results)

</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🔍 Face Detection
- **Haar Cascade** - Robust frontal face detection
- **LBP Cascade** - Fast & lightweight alternative
- Multi-face detection support
- Real-time processing capability

</td>
<td width="50%">

### 🧠 Emotion Analysis
- **7 Core Emotions**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- **2 Extended States**: Depressed*, Confused* (proxy detection)
- Multiple ML algorithms compared
- Deep learning powered analysis

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy matplotlib scikit-learn deepface seaborn
```

### Run Face Detection

```python
import cv2

# Load cascade classifiers
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
lbp_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# Detect faces
img = cv2.imread('your_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f"Detected {len(faces)} face(s)")
```

### Run Emotion Detection (DeepFace)

```python
from deepface import DeepFace

result = DeepFace.analyze(
    img_path="your_image.jpg",
    actions=["emotion"],
    detector_backend="opencv"
)

print(f"Dominant emotion: {result[0]['dominant_emotion']}")
```

---

## 🔬 How It Works

### The Pipeline

```
┌─────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────────┐
│   Image     │───▶│  Preprocessing  │───▶│   Feature    │───▶│    ML      │
│   Input     │    │  & Face Detect  │    │  Extraction  │    │ Classifier │
└─────────────┘    └─────────────────┘    └──────────────┘    └────────────┘
                          │                       │                  │
                          ▼                       ▼                  ▼
                    • Grayscale              • HoG (edges)      • SVM
                    • Scaling                • LBP (texture)    • Random Forest
                    • Denoising                                 • KNN
                                                                • Decision Tree
```

### Feature Extraction Techniques

<details>
<summary><b>📐 Histogram of Oriented Gradients (HoG)</b></summary>

HoG captures structural information by analyzing edge orientations:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Orientations | 9 | Number of gradient direction bins |
| Pixels per Cell | 8×8 | Local region size |
| Cells per Block | 2×2 | Normalization window |
| **Output** | **1764 features** | Compressed representation |

```python
from skimage.feature import hog

hog_features = hog(image, orientations=9, pixels_per_cell=(8,8), 
                   cells_per_block=(2,2), visualize=False)
```

> 💡 **Why HoG?** Captures macro-structure like eyebrow curves and mouth shapes

</details>

<details>
<summary><b>🔲 Local Binary Patterns (LBP)</b></summary>

LBP describes micro-textures and is robust to illumination changes:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| P (Points) | 8 | Neighbor sampling points |
| R (Radius) | 1 | Circular radius |
| Method | Uniform | Reduces feature dimensions |
| **Output** | **59 features** | Histogram of patterns |

```python
from skimage.feature import local_binary_pattern

lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
lbp_hist = np.histogram(lbp, bins=59, range=(0, 59))[0]
```

> 💡 **Why LBP?** Captures skin texture, wrinkles, and fine facial details

</details>

---

## 📊 Model Comparison

We trained and evaluated **4 machine learning algorithms** on **1,575 images** across **7 emotion classes**:

| Model | Accuracy | Training Time | Strengths |
|-------|----------|---------------|-----------|
| 🥇 **SVM (RBF)** | **44.13%** | Medium | Best at finding optimal decision boundaries |
| 🥈 Random Forest | 36.51% | Fast | Ensemble voting, handles noise well |
| 🥉 KNN (k=5) | 32.70% | Very Fast | Simple, instance-based learning |
| Decision Tree | 22.86% | Very Fast | Interpretable, but prone to overfitting |

```
                    Model Performance Comparison
    ┌────────────────────────────────────────────────────┐
    │ SVM          ████████████████████████░░░░  44.1%   │
    │ Random Forest████████████████░░░░░░░░░░░░  36.5%   │
    │ KNN          ██████████████░░░░░░░░░░░░░░  32.7%   │
    │ Decision Tree████████░░░░░░░░░░░░░░░░░░░░  22.9%   │
    └────────────────────────────────────────────────────┘
```

### Why These Accuracies?

> Traditional ML on raw features faces challenges:
> - Subtle differences between emotions (e.g., sad vs. neutral)
> - Limited training data (1,575 images)
> - High intra-class variation
> 
> **For production use**, consider deep learning approaches like DeepFace which achieve **90%+ accuracy**

---

## 🎯 Extended Emotion Detection

Beyond the 7 base emotions, we compute **proxy scores** for complex states:

```python
# Depressed: high sad + neutral, low happy
depressed = clamp01(0.60 * sad + 0.40 * neutral - 0.30 * happy)

# Confused: uncertainty mixture with surprise/fear
confused = clamp01(0.45 * surprise + 0.35 * fear + 0.20 * neutral - 0.20 * happy)
```

### Sample Output

```
Face 1 expanded scores (top 6):
├── happy     : 96.3% ████████████████████
├── neutral   :  3.7% █
├── surprise  :  0.0% 
├── sad       :  0.0% 
├── fear      :  0.0% 
└── angry     :  0.0% 
```

---

## 📈 Results

### Confusion Matrix Analysis

```
                    Predicted Emotion
              ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
              │Angry │Disg. │Fear  │Happy │Neut. │ Sad  │Surp. │
        ┌─────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
 Actual │Angry│  2   │  0   │  3   │  5   │  12  │  11  │  6   │
        │Disg.│  1   │  3   │  2   │  3   │  4   │  5   │  4   │
        │Fear │  2   │  0   │  1   │  6   │  12  │  11  │  6   │
        │Happy│  3   │  1   │  1   │  43  │  4   │  3   │  5   │
        │Neut.│  3   │  0   │  3   │  8   │  26  │  10  │  5   │
        │ Sad │  2   │  1   │  2   │  10  │  12  │  19  │  7   │
        │Surp.│  1   │  0   │  0   │  15  │  7   │  6   │  29  │
        └─────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

### Key Metrics Explained

| Metric | Formula | What It Tells Us |
|--------|---------|------------------|
| **Precision** | TP / (TP + FP) | "When model says Happy, is it right?" |
| **Recall** | TP / (TP + FN) | "Does model find ALL Happy faces?" |
| **F1-Score** | 2 × (P × R) / (P + R) | Balanced measure of both |

---

## 🛠️ Project Structure

```
📁 facial-emotion-recognition/
├── 📓 notebooks/
│   ├── face_detection.ipynb      # Haar & LBP face detection
│   ├── emotion_ml.ipynb          # Traditional ML approach
│   └── emotion_deepface.ipynb    # Deep learning approach
├── 📁 data/
│   ├── haarcascade_frontalface_alt.xml
│   ├── lbpcascade_frontalface.xml
│   └── test_images/
├── 📁 outputs/
│   ├── result_haar.jpg
│   └── result_lbp.jpg
└── 📄 README.md
```

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📓 Emotion Detection Colab | [sorts.pro/emotion](https://sorts.pro/emotion) |
| 📓 Face Detection Colab | [sorts.pro/face](https://sorts.pro/face) |
| 📓 Face + Emotion Colab | [sorts.pro/emotionface](https://sorts.pro/emotionface) |

---

## 📚 References & Theory

<details>
<summary><b>Click to expand technical references</b></summary>

### Cascade Classifiers
- **Haar Features**: Viola-Jones algorithm using integral images for rapid feature computation
- **LBP Features**: Computationally simpler alternative with similar detection accuracy

### Machine Learning Models
- **SVM**: Finds optimal hyperplane using kernel trick (RBF kernel)
- **Random Forest**: Ensemble of 100 decision trees with majority voting
- **KNN**: Classification based on k=5 nearest neighbors in feature space
- **Decision Tree**: Recursive partitioning with max_depth=15

### Deep Learning
- **DeepFace**: Pre-trained CNN achieving state-of-the-art accuracy on FER benchmarks

</details>

---

## 👥 Team

<div align="center">

**Advanced Signal and Image Processing Lab (ASIP Lab)**

*Department of Data Science and Engineering*  
*Indian Institute of Science Education and Research (IISER) Bhopal*

| Role | Name |
|------|------|
| PI | Dr. Samiran Das |
| Presenter | Mr. Sajjan Singh |
| Presenter | Mr. Ramen Ghosh |


</div>

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

<br/>

Made with ❤️ by Sajjan Singh from ASIP Lab, IISER Bhopal

<br/>

[![forthebadge](https://forthebadge.com/images/badges/built-with-science.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-coffee.svg)](https://forthebadge.com)

</div>
