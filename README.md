# heart-disease-detection
• Developed KNN-based prediction model using 13 clinical features (91% accuracy).
• Built frontend interface for real-time user input and dynamic prediction output.
• Integrated ML model with backend logic for live inference
The Heart Disease Prediction System is a desktop-based machine learning application developed using Python. It predicts the likelihood of heart disease based on clinical and medical parameters entered by the user. The project combines data preprocessing, supervised machine learning, and a graphical user interface (GUI) to deliver real-time predictions in an easy-to-use format.

This application demonstrates how machine learning models can be integrated into a practical healthcare tool to support early risk detection.

📌 Project Overview

Cardiovascular diseases are one of the leading causes of death worldwide. Early prediction and diagnosis play a critical role in reducing risk and improving patient outcomes.

This system uses a trained K-Nearest Neighbors (KNN) classification model to analyze patient health indicators and determine whether the individual is at risk of heart disease.

Users input medical parameters through a graphical interface, and the system processes the data, applies the trained model, and provides an instant prediction result.

🧠 How the System Works

The application follows a complete machine learning workflow:

1. Data Collection

The system uses a labeled heart disease dataset containing medical attributes and a target variable indicating the presence or absence of heart disease.

2. Data Preprocessing

The dataset undergoes preprocessing to prepare it for machine learning:

Categorical features are identified and transformed using one-hot encoding.

Continuous features are standardized using feature scaling.

The target column is separated from the feature set.

This ensures the model receives properly formatted and normalized data for accurate predictions.

3. Model Training

A K-Nearest Neighbors classifier is trained on the processed dataset. The model learns patterns between medical attributes and heart disease outcomes.

KNN was chosen because it is simple, effective for smaller datasets, and performs well for distance-based classification problems.

4. Graphical User Interface

The application includes a desktop GUI built using Tkinter. The interface allows users to enter medical values such as:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol level

Fasting blood sugar

Resting ECG results

Maximum heart rate achieved

Exercise-induced angina

ST depression (Oldpeak)

Slope

Number of major vessels

Thalassemia

After clicking the Predict button, the system processes the input data and displays one of the following results:

Heart Disease Detected

Normal Heart Condition

The output is visually distinguished using color indicators for clarity.

⚙️ Technologies Used

Python

NumPy

Pandas

Scikit-Learn

Tkinter
