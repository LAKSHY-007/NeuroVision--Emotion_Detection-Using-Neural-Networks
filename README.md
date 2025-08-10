# 🔍 Overview
A high-performance deep learning system for real-time facial emotion recognition, capable of detecting 7 distinct emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised) from live video streams with optimized frame rates.

---

# 🚀 Key Features
- **CNN Architecture:** Custom 6-layer convolutional neural network with BatchNorm and Dropout  
- **Real-Time Processing:** Achieves 15-20 FPS on standard webcams (optimized with OpenCV)  
- **Data Augmentation:** Advanced image transformations during training  
- **Visual Analytics:** Live emotion probability distribution and confidence scores  
- **Training Tools:** Includes learning rate scheduling and early stopping  

---

# 🛠️ Technical Specifications

| Component           | Details                                  |
|---------------------|------------------------------------------|
| **Model**           | Custom CNN (3 Conv blocks → 2 Dense layers) |
| **Input Resolution** | 48×48 grayscale                          |
| **Training Dataset** | 28,709 training images / 7,178 validation samples |
| **Accuracy**        | ~95% validation accuracy (7-class)      |
| **Dependencies**    | TensorFlow, OpenCV, NumPy, Matplotlib   |

---

# 📦 Installation

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
```

#  Screenshots
![Emotion1](https://github.com/user-attachments/assets/b71e840b-2717-4ea5-ab61-40880e351cc3)
  
![Emotion2](https://github.com/user-attachments/assets/3bf52a2b-c4eb-4d2b-859c-1cd95b5fb4a7)








