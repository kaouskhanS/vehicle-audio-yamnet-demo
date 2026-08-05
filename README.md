# 🚗 Vehicle Audio Fault Detection using YAMNet

An AI-powered web application that detects vehicle faults by analyzing engine and vehicle sounds using Google's **YAMNet** audio classification model. Users can upload audio recordings of their vehicles, and the system predicts possible mechanical issues such as engine knock, brake squeal, exhaust leaks, gear noise, flat tires, or normal operating conditions.

---

# 📖 Project Overview

Vehicle maintenance often depends on manually identifying unusual sounds, which can be difficult for non-experts. This project leverages **Deep Learning** and **Audio Classification** to automatically recognize vehicle-related sounds and identify potential faults.

The application uses **TensorFlow Hub's YAMNet model** to extract audio features and classify uploaded vehicle sounds through a user-friendly web interface.

---

# ✨ Features

- 🎵 Upload vehicle audio recordings (.wav)
- 🤖 AI-powered audio classification using YAMNet
- 🔍 Detect common vehicle faults
- 📊 Confidence score for predictions
- 🌐 Interactive web interface
- ⚡ FastAPI backend
- 💻 HTML, CSS & JavaScript frontend
- 📁 Sample audio dataset included

---

# 🚗 Detectable Vehicle Conditions

- ✅ Normal Vehicle Sound
- 🔧 Engine Knock
- 🛞 Flat Tire
- 🚦 Brake Squeal
- 🔩 Gear Noise
- 💨 Exhaust Leak

---

# 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- FastAPI
- Uvicorn

### AI / Machine Learning
- TensorFlow
- TensorFlow Hub
- YAMNet
- NumPy
- Librosa

---

# 📂 Project Structure

```text
vehicle-audio-yamnet-demo/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── datasets/
│   ├── brake_squeal/
│   ├── engine_knock/
│   ├── exhaust_leak/
│   ├── flat_tire/
│   ├── gear_noise/
│   └── no_issue/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/your-username/vehicle-audio-yamnet-demo.git
```

```bash
cd vehicle-audio-yamnet-demo
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Backend

```bash
uvicorn backend.app:app --reload
```

The backend will start at:

```
http://127.0.0.1:8000
```

---

# 🌐 Run the Frontend

Open the `frontend/index.html` file in your browser or serve it using a local web server.

---

# 🔄 Workflow

1. Upload a vehicle sound recording.
2. Preprocess the audio.
3. Extract embeddings using YAMNet.
4. Analyze the sound pattern.
5. Predict the vehicle condition.
6. Display the predicted fault and confidence score.

---

# 📊 Dataset Categories

| Folder | Description |
|---------|-------------|
| brake_squeal | Brake squealing sounds |
| engine_knock | Engine knocking sounds |
| exhaust_leak | Exhaust leakage sounds |
| flat_tire | Flat tire sounds |
| gear_noise | Gearbox noise |
| no_issue | Normal vehicle sounds |

---

# 📸 Screenshots

You can add screenshots of:

- Home Page
- Audio Upload Interface
- Prediction Result
- Confidence Score Display
- Fault Detection Output

Create a `screenshots/` folder and include the images.

---

# 🎯 Objectives

- Detect vehicle faults using sound analysis.
- Reduce manual diagnosis time.
- Improve preventive vehicle maintenance.
- Demonstrate AI applications in automotive diagnostics.
- Build a lightweight and user-friendly diagnostic tool.

---

# 🚀 Future Enhancements

- 🎤 Live microphone recording
- 📱 Mobile-friendly interface
- ☁️ Cloud deployment
- 📈 Fault history dashboard
- 🔊 Support for additional vehicle fault classes
- 🧠 Custom-trained deep learning model
- 🚗 OBD-II integration for combined diagnostics

---

# 📦 Requirements

- Python 3.10+
- FastAPI
- TensorFlow
- TensorFlow Hub
- NumPy
- Librosa
- Uvicorn

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# 💡 How It Works

The system processes uploaded vehicle audio files and extracts audio embeddings using **Google's YAMNet model**. These embeddings are analyzed to classify the sound into one of the predefined vehicle fault categories. The predicted class and confidence score are then presented through the web interface.

---

# 👨‍💻 Author

**Kaous Khan S and Sujitha A**

- Aspiring Data Analyst
- Machine Learning Enthusiast
- Python Developer
- AI & Deep Learning Learner

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📄 License

This project is developed for educational and research purposes. Feel free to use, modify, and extend it for learning and academic projects.

---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub. Your support helps showcase the project and encourages future improvements.

What's included:
- backend/: FastAPI backend that uses a Keras model saved at models/vehicle_classifier.h5
- frontend/: Static frontend (upload + record)
- datasets/: synthetic sample audio for classes: no_issue, engine_knock, brake_squeal, flat_tire, exhaust_leak, gear_noise
- training/: Keras training script (train_keras.py) that creates mel-spectrograms and trains a CNN
- models/: contains vehicle_classifier.h5 (either trained or placeholder)
- preprocessing/: processing utilities
- requirements.txt: Python dependencies for backend and training

Important:
- I attempted to train a Keras model here. See training_log.txt for details.
- YAMNet-based pipeline requires TensorFlow Hub and internet access to download the YAMNet model. I included a Keras mel-spectrogram classifier as a demo that works offline once dependencies are installed.
- To reproduce full YAMNet+classifier training, ensure internet access and modify training script to extract YAMNet embeddings from TensorFlow Hub.

How to run locally:
1. python -m venv venv
2. source venv/bin/activate (Windows: venv\Scripts\activate)
3. pip install -r requirements.txt
4. uvicorn backend.app:app --reload --port 8000
5. Open frontend/index.html and set BACKEND to http://localhost:8000 if needed.

