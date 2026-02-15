
import numpy as np
import pandas as pd
from pathlib import Path

#Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, accuracy_score

#GUI
import tkinter as tk
from tkinter import messagebox

#Loading Dataset
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"
df = pd.read_csv(DATA_PATH)

#Data Preprocessing
categorical_val = []
continous_val = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
        
categorical_val.remove('target')
dfs = pd.get_dummies(df, columns = categorical_val)

sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale])

#Model Building
X=dfs.drop("target",axis=1)
Y=dfs["target"]
X_values = X.to_numpy()
Y_values = Y.to_numpy()

np.random.seed(42)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_values, Y_values)

#GUI Model
root = tk.Tk()
root.title("Heart Disease Predictor")
root.geometry("600x600")

# Input variables
age_var = tk.IntVar()
sex_var = tk.IntVar()
cp_var = tk.IntVar()
trestbps_var = tk.IntVar()
chol_var = tk.IntVar()
fbs_var = tk.IntVar()
restecg_var = tk.IntVar()
thalach_var = tk.IntVar()
exang_var = tk.IntVar()
oldpeak_var = tk.DoubleVar()
slope_var = tk.IntVar()
ca_var = tk.IntVar()
thal_var = tk.IntVar()

# Function to predict heart disease
def predict():
    try:
        # Get the user inputs
        age = age_var.get()
        sex = sex_var.get()
        cp = cp_var.get()
        trestbps = trestbps_var.get()
        chol = chol_var.get()
        fbs = fbs_var.get()
        restecg = restecg_var.get()
        thalach = thalach_var.get()
        exang = exang_var.get()
        oldpeak = oldpeak_var.get()
        slope = slope_var.get()
        ca = ca_var.get()
        thal = thal_var.get()

        # Preprocess the input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        input_data = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)

        # Scale the input data
        input_data[col_to_scale] = sc.transform(input_data[col_to_scale])

        # Predict heart disease
        prediction = knn.predict(input_data.to_numpy())[0]

        # Display the result in UI and terminal
        if prediction == 1:
            result_text = "Prediction: Heart Disease"
            result_color = "red"
        else:
            result_text = "Prediction: Normal Heart"
            result_color = "green"

        result_label.config(text=result_text, fg=result_color)
        print(result_text)
        messagebox.showinfo("Prediction Result", result_text)
    except Exception as exc:
        result_label.config(text=f"Error: {exc}", fg="orange red")
        messagebox.showerror("Prediction Error", str(exc))
        

# GUI elements
age_label = tk.Label(root, text="Age:")
age_label.pack()
age_entry = tk.Entry(root, textvariable=age_var)
age_entry.pack()

sex_label = tk.Label(root, text="Sex:")
sex_label.pack()
sex_entry = tk.Entry(root, textvariable=sex_var)
sex_entry.pack()

cp_label = tk.Label(root, text="Chest pain:")
cp_label.pack()
cp_entry = tk.Entry(root, textvariable=cp_var)
cp_entry.pack()

trestbps_label = tk.Label(root, text="Resting blood pressure:")
trestbps_label.pack()
trestbps_entry = tk.Entry(root, textvariable=trestbps_var)
trestbps_entry.pack()

chol_label = tk.Label(root, text="Serum cholesterol:")
chol_label.pack()
chol_entry = tk.Entry(root, textvariable=chol_var)
chol_entry.pack()

fbs_label = tk.Label(root, text="Fasting blood sugar:")
fbs_label.pack()
fbs_entry = tk.Entry(root, textvariable=fbs_var)
fbs_entry.pack()

restecg_label = tk.Label(root, text="Resting electrocardiographic results:")
restecg_label.pack()
restecg_entry = tk.Entry(root, textvariable=restecg_var)
restecg_entry.pack()

thalach_label = tk.Label(root, text="Maximum heart rate achieved:")
thalach_label.pack()
thalach_entry = tk.Entry(root, textvariable=thalach_var)
thalach_entry.pack()

exang_label = tk.Label(root, text="Exercise-induced angina:")
exang_label.pack()
exang_entry = tk.Entry(root, textvariable=exang_var)
exang_entry.pack()

oldpeak_label = tk.Label(root, text="Oldpeak:")
oldpeak_label.pack()
oldpeak_entry = tk.Entry(root, textvariable=oldpeak_var)
oldpeak_entry.pack()

slope_label = tk.Label(root, text="Slope:")
slope_label.pack()
slope_entry = tk.Entry(root, textvariable=slope_var)
slope_entry.pack()

ca_label = tk.Label(root, text="Ca:")
ca_label.pack()
ca_entry = tk.Entry(root, textvariable=ca_var)
ca_entry.pack()

thal_label = tk.Label(root, text="Thalassemia:")
thal_label.pack()
thal_entry = tk.Entry(root, textvariable=thal_var)
thal_entry.pack()
# Add other input fields (e.g., sex, cp, trestbps, etc.) similarly...

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial Black", 15))
result_label.pack(pady=10)


# Run the GUI event loop
root.mainloop()

