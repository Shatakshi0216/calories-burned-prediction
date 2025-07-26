# 🔥 Predicting Calories Burned During a Workout

This is a simple machine learning web application built using **Streamlit** and **pure Python (no sklearn)**.  
It predicts the number of calories burned during a workout based on:

- Gender  
- Age  
- Duration of the workout  
- Heart rate

The model is trained using **Linear Regression with Gradient Descent** implemented from scratch.

---

## 🚀 Features

- Custom linear regression (no external ML libraries)
- Clean web UI using Streamlit
- Realtime calorie prediction based on user inputs
- Easy to train and update weights manually or from file

---

## 🧠 How It Works

### Model Inputs:
- Gender: `0` (Female), `1` (Male)
- Age: Integer
- Duration (in minutes)
- Heart Rate (BPM)

### Output:
- Calories Burned (float)

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/calorie-predictor-app.git
cd calorie-predictor-app
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
.
├── app.py               # Streamlit frontend
├── model.json           # Trained weights and bias (optional, if using JSON)
├── your_file.csv        # Dataset file for training
├── train.py             # Your custom gradient descent trainer
├── requirements.txt
└── README.md
```

---

## ✍️ Author

**Shatakshi Tiwari**  
📫 Feel free to connect!

---

## 💡 Note

This project avoids using sklearn intentionally to demonstrate how machine learning can be implemented from scratch for educational purposes.
