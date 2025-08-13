# 🌾 Rice Leaf Disease Detection


## 📖 Overview
This project implements a deep learning-based image classification system for detecting three major rice leaf diseases:
- **Leaf Blast**
- **Bacterial Blight**
- **Brown Spot**

The system uses convolutional neural networks (CNN) and transfer learning to classify leaf images into the above categories with high accuracy.

---

## 📂 Dataset
The dataset contains images of rice leaves collected from multiple sources.  
Each image is labeled according to the disease type.  
Classes include:
- **Leaf Blast**
- **Bacterial Blight**
- **Brown Spot**

Data augmentation techniques are applied to improve generalization.

---

## 🛠 Tech Stack
- **Language**: Python 3.9+
- **Libraries**: TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn
- **Tools**: Jupyter Notebook, Google Colab

---

## ⚙️ Installation
```bash
git clone https://github.com/username/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
pip install -r requirements.txt
```

---

## 🚀 Usage
```bash
jupyter notebook PRCP_1001_Rice_Leaf_Disease_Detection.ipynb
```
Ensure the dataset is placed in the correct directory before running.

---

## 🔍 Methodology
The approach followed in the notebook includes:

# PRCP-1001: Rice Leaf Disease Detection

# **Project Introduction:**
Rice is one of the major cultivated crops in India. Rice crops are susceptible to various diseases at various stages of cultivation. Early detection and remedy is required to maintain the best quality and quantity in rice. Farmers with their limited knowledge, have a difficult time identifying these diseases manually. Therefore automated image recognition systems using Convolutional Neural Network (CNN) can be very beneficial in such problems.

# **Classification Problem:**
Our goal is to build a model that would automatically classify rice leaf diseases. For this, we have taken three major attacking diseases in rice plants like leaf blast, bacterial blight and brown spot. We created a model that would determine if the future data inputs will fall in either of these 3 diseases- leaf,blast, bacterial blight and brown spot.

We have deviced the project into multiple steps

• Loading Data

• Preparing Dataset

• Data Preproocessing

• Model Building

• Training

• Check

# **Basic Import**

# **Drive mount**

# **Import Keras packages**

# **Getting data**

# **Image in array format**

# **Defining Features & Class**

# **Converting Features img & Class into array**

# **Train test split**

# **Creating CNN architecture**

# **Model Summary**

# **Model architecture figure**

# **Optimizers**

# **Callback**

# **Preprocessing Scale** **images**

# **Getting unique class**

**Defining class labels**

# **Plotting Images**

# **Image Augmentation**

# **Model trainning**

# **Loss Plot**

# **Accuracy Plot**

# **Model Evaluation**

# **Classification report**

# **Confusion Matrix**

# **Conclusion**
Based on the given objectives the dataset has been analysed,the model has been built and the results have been predicted with the test data.The CNN Machine learning model has been fitted and predicted with high accuracy.Also, we observed that by adjusting the training parameters like learning rate, number of epochs, and optimizer methods, we can get significant accuracy with a handmade model having less number of layers than the other traditional models. The better we can detect infections, the simpler it will be for farmers to protect their crops. In the future, we will broaden the scope to include more diseases and algorithms, making disease detection vast, easier and faster.

---

## 📊 Results
The trained CNN and transfer learning models achieved the following performance metrics on the test dataset:

| Model             | Accuracy |
|-------------------|----------|
| Custom CNN        | 0.89     |
| MobileNetV2       | 0.94     |

**Sample Predictions:**  
![Sample Predictions](assets/sample_predictions.png)

---

## 🚧 Future Improvements
- Increase dataset size for better model generalization
- Deploy as a mobile app for real-time predictions
- Integrate with farmer advisory systems

---

## 🙌 Author
**Prathmesh Mane Deshmukh**
