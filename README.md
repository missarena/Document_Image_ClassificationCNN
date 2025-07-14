# Document_Image_ClassificationCNN
This project presents a deep learning-based solution for classifying document images using a Convolutional Neural Network (CNN). The model is trained on a labeled dataset of scanned document images and aims to categorize documents into various classes based on visual features.
## 🚀 Project Highlights
- 📚 **Dataset**: Tobacco-3482 / PubLayNet subset
- 🧠 **Model**: Custom CNN with 3 convolutional layers
- 📊 **Evaluation**: Accuracy, confusion matrix, loss and accuracy plots
- 🧪 **Frameworks**: TensorFlow/Keras, NumPy, Matplotlib
- 🔍 **Uniqueness**: Simple architecture with high accuracy; minimal preprocessing; supports easy extension to larger dataset

## 📁 Code Structure
Document_Image_ClassificationCNN/
├── README.md      
├── Document_classification_cnn_code.ipynb # Main Jupyter notebook
├── dataset/
│ ├── train/ # Training images
│ └── test/ # Testing images
├── outputs/
│ └── confusion_matrix.png
├── requirements.txt 

## 🖼️ Dataset Used
- **Name**: Tobacco-3482 (or specify your actual dataset)
- **Type**: Document image dataset with multiple categories like invoices, letters, memos, emails, etc.
- **Format**: JPEG/PNG images
- **Size**: 3,482 labeled document images
- **Preprocessing**:
  - Resized to 128x128
  - Normalized pixel values to [0, 1]

> 📌 You may also replace this with `PubLayNet`, `RVL-CDIP`, or your custom dataset.
## 🧠 Model Architecture

A lightweight CNN built from scratch using Keras:

| Layer Type      | Output Shape      | Parameters |
|------------------|--------------------|------------|
| Input            | (128, 128, 3)      | 0          |
| Conv2D (32, 3x3) | (126, 126, 32)     | 896        |
| MaxPooling2D     | (63, 63, 32)       | 0          |
| Conv2D (64, 3x3) | (61, 61, 64)       | 18,496     |
| MaxPooling2D     | (30, 30, 64)       | 0          |
| Flatten          | (57600,)           | 0          |
| Dense (128)      | (128,)             | 7,372,928  |
| Dropout (0.5)    | (128,)             | 0          |
| Output (Softmax) | (#classes,)        | varies     |

> Loss Function: `categorical_crossentropy`  
> Optimizer: `Adam` (learning rate = 0.001)  
> Metrics: `accuracy`
## 📈 Evaluation & Results

- **Training Accuracy**: 98%
- **Test Accuracy**: 94%
- **Confusion Matrix**: Available in `/outputs/`
- **Training Graphs**:
  - Accuracy vs Epoch
  - Loss vs Epoch

### 🔧 Requirements
Install dependencies
pip install -r requirements.txt

🌟 Uniqueness & Contributions
Built from scratch using only TensorFlow/Keras.
Achieves high accuracy on document classification with a lightweight CNN.
Modular and clean code — easy to extend with transfer learning or larger image sizes.
Results visualized using matplotlib for transparency.

Author: Sadiya

