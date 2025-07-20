# Face-detector

This Python project uses OpenCV and pre-trained deep learning models to detect faces and recognize their age group, gender, and emotion in real-time using your webcam — no TensorFlow needed, fully compatible with Python 3.13.

# 🎯 Real-Time Face Detection with Age, Gender & Emotion Recognition

This project demonstrates a real-time computer vision system that detects human faces from a webcam feed and predicts the **age group**, **gender**, and **emotion** of each detected face. Built with **OpenCV**, it leverages **pre-trained models** in **ONNX** and **Caffe** formats and is fully compatible with **Python 3.13** — no TensorFlow or Keras required.

---

## 🚀 Features

- 🔍 **Face Detection** using Haar Cascade Classifier  
- 🙂 **Emotion Recognition** using `emotion-ferplus-8.onnx` (FER+ dataset)  
- 🚻 **Gender Classification** using Caffe-based CNN  
- 🧒 **Age Estimation** into 8 categories (e.g., 0-2, 4-6, ..., 60-100)  
- 🖥️ **Real-time processing** via webcam
- ✅ Fully runs on Python 3.13 (no TensorFlow)

---

## 📦 Requirements

>Install the dependencies with:
pip install opencv-python numpy

>📁 Pre-trained Models (Place in same folder as the script)
Task	Files Needed
Emotion	emotion-ferplus-8.onnx
Age	age_deploy.prototxt, age_net.caffemodel
Gender	gender_deploy.prototxt, gender_net.caffemodel

> You can download the models from:

1)Emotion (ONNX): FER+ ONNX
2)Age/Gender (Caffe): LearnOpenCV AgeGender

>After placing all model files in the same directory:
python detect_all.py
Press q to quit the webcam feed.


>🧠 Age Groups Used

(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

>🙂 Emotions Recognized

neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
