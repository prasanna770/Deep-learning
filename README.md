
# 🩺 Diabetic Retinopathy Detection using CNN

## 📌 Project Overview

This project focuses on the **early detection of Diabetic Retinopathy
(DR)** from retinal fundus images using a **Convolutional Neural
Network (CNN)**.
The model classifies images into five severity levels:

* 0 → No DR
* 1 → Mild
* 2 → Moderate
* 3 → Severe
* 4 → Proliferative DR

Early detection helps prevent vision loss in diabetic patients.

---

## 🧠 Dataset

* *Dataset:* APTOS 2019 Blindness Detection
* Due to computational limitations in Google Colab, a **resized subset
(1000 images)** was used.
* Images were resized to *224 × 224* and normalized.

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy & Pandas
* Matplotlib & Seaborn
* Streamlit (for web app)
* Ngrok (for public deployment)
* Google Colab

---

## 🏗️ Model Architecture

The CNN model consists of:

* 3 Convolution layers (32, 64, 128 filters)
* MaxPooling layers
* Flatten layer
* Dense layer (128 neurons)
* Dropout (0.5) for overfitting reduction
* Output layer with *Softmax (5 classes)*

*Loss Function:* Categorical Crossentropy
*Optimizer:* Adam

---

## 📊 Model Evaluation

The model performance was evaluated using:

* Training vs Validation Accuracy Graph
* Training vs Validation Loss Graph
* Confusion Matrix
* Test Accuracy Score

---

##  Deployment

A *Streamlit web application* was developed for real-time prediction.

Users can:

1. Upload a fundus image
2. Get the predicted DR stage instantly

Ngrok was used to generate a *public URL* for demonstration.

---

##  Project Structure


diabetic-retinopathy-cnn/
│── app.py                # Streamlit web app
│── dr_model.h5           # Trained CNN model
│── images                # Saved graphs for README & report
│── requirements.txt      # Required Python libraries
│── README.md             # Project documentation


---

##  How to Run the Project

### 1️ Install Dependencies

bash
pip install -r requirements.txt


### 2️ Run Streamlit App

bash
streamlit run app.py

This project helps in gaining hands on experience with deep learning models mainly used Convolutional neural network model is used this project for automated diabetic retinopathy detection using fundus images. It focuses on analyzing retinal images to identify early signs of the disease and improve diagnosis accuracy.


---

## 🎯 Results

* Successfully classified fundus images into 5 DR stages
* Achieved good accuracy on test data using a CNN model
* Demonstrated real-time prediction through a web interface

![Accuracy](images/accuracy.png)
![Loss](images/loss.png)
![Confusion Matrix](images/confusion_matrix.png)


---

## 📚 Future Improvements

* Use *ResNet50 / Transfer Learning* for higher accuracy
* Train on the full APTOS dataset
* Deploy using *Streamlit Cloud or Docker*

---

##  Author

*Rongali Prasanna*
B.Tech – Computer Science Engineering (CSE)
Jawaharlal Nehru Technological University, Kakinada
Graduation Year: 2026

---

## 🏁 Conclusion

This project demonstrates how deep learning can assist in the
*automated screening of Diabetic Retinopathy*, enabling faster and
more accessible diagnosis in healthcare.
