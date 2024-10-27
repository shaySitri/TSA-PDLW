# Time-Series Activity Classification with Wearable Sensor Data ⌚📊

[![Jupyter Notebook](https://img.shields.io/badge/Notebook-TSA_ass2.ipynb-orange?logo=jupyter)](https://github.com/shaySitri/TSA-PDLW/blob/main/TSA_ass2.ipynb)

This repository contains a Jupyter Notebook for classifying activities using multivariate time-series data collected from wearable sensors. The project explores several machine learning and deep learning models, focusing on classifying various physical activities based on x, y, and z acceleration data.

---

## 📄 Project Overview
The project uses time-series data from wearable sensors placed on different parts of the body to classify 18 unique activities (e.g., walking, sitting, washing dishes). The classification task uses a combination of classical machine learning and deep learning techniques to identify activity patterns in acceleration data.

---

## 📊 Data Insights
- **Dataset**: Multivariate time-series data with each row representing a sequence of sensor readings.
- **Sensors**: Data sourced from smartwatches and Vicon sensors, with variations in placement (hands, feet, left, and right sides).
- **Heterogeneity**: Inconsistent sequence lengths, unequal data quantities for each activity, and varied sensor placement.

---

## 🧩 Model Development

### Baseline Models
1. **Last Known Value**: Accuracy = 0.5%
2. **Majority Rule**: Accuracy = 0.9%
3. **K-Nearest Neighbors (KNN)**: Accuracy = 43.5%

### Machine Learning Models
- **Random Forest**: Utilized a flattened structure with separate x, y, and z channels. Achieved validation accuracy of 83.2% using only accelerometer data.

### Deep Learning Models
1. **1D-CNN**:
   - Architecture: 3 convolutional layers with varying kernel sizes (1x1, 3x3, 5x5), ReLU activations, max-pooling, and a fully connected layer.
   - **Results**: Validation accuracy ~66.4%; average accuracy across channels = 83.3%.

2. **GRU**:
   - Architecture: 2 LSTM layers and 1 GRU layer with max-pooling, dropout, and dense layers.
   - **Results**: Validation accuracy ~50.9%.

3. **Improved GRU**:
   - Simplified with 1 GRU layer and increased dropout rate to prevent overfitting.
   - **Results**: Validation accuracy improved to 55.8%.

4. **Autoencoder**:
   - Architecture: Pre-trained on autoregressive tasks, then fine-tuned for classification.
   - **Results**: Achieved validation accuracy of 61.6%.

---

## 🔍 Key Findings
- **Data Diversity**: The heterogeneity in sequence length and sensor type posed challenges.
- **Improved Model Complexity**: Reducing model complexity and adding dropout improved generalization.
- **Transfer Learning**: Pre-training models with autoregressive tasks enhanced classification accuracy.

