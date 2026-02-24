# ==========================================================
# Voice-Based Stress Detection using MFCC + Time-Series
# Advanced Time Series Analysis (ATSA Version)
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib

from feature_extraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------
# Create output folders
# ----------------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

dataset_path = "dataset"

# ==========================================================
# VISUALIZATION FUNCTIONS
# ==========================================================

def save_waveform(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    plt.figure()
    librosa.display.waveshow(signal, sr=sr)
    plt.title("Waveform")
    plt.savefig("results/waveform.png")
    plt.close()


def save_spectrogram(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    S = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(abs(S))
    plt.figure()
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.title("Spectrogram")
    plt.savefig("results/spectrogram.png")
    plt.close()


def save_mfcc_trajectory(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    plt.figure()
    plt.plot(mfcc[0])
    plt.title("MFCC Coefficient 1 Over Time")
    plt.xlabel("Frame")
    plt.ylabel("MFCC Value")
    plt.savefig("results/mfcc_trajectory.png")
    plt.close()


# ==========================================================
# TIME-SERIES ANALYSIS FUNCTIONS (ATSA)
# ==========================================================

def mfcc_variance_analysis(dataset_path):
    normal_var = []
    stressed_var = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                signal, sr = librosa.load(path, sr=22050)

                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                variance = np.var(mfcc)

                emotion_code = file.split("-")[2]

                if emotion_code in ["05", "06", "07"]:
                    stressed_var.append(variance)
                else:
                    normal_var.append(variance)

    return normal_var, stressed_var


def energy_variance_analysis(dataset_path):
    normal_energy = []
    stressed_energy = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                signal, sr = librosa.load(path, sr=22050)

                rms = librosa.feature.rms(y=signal)
                variance = np.var(rms)

                emotion_code = file.split("-")[2]

                if emotion_code in ["05", "06", "07"]:
                    stressed_energy.append(variance)
                else:
                    normal_energy.append(variance)

    return normal_energy, stressed_energy


def pitch_contour_plot(normal_file, stressed_file):
    signal_n, sr_n = librosa.load(normal_file, sr=22050)
    signal_s, sr_s = librosa.load(stressed_file, sr=22050)

    pitch_n, mag_n = librosa.piptrack(y=signal_n, sr=sr_n)
    pitch_s, mag_s = librosa.piptrack(y=signal_s, sr=sr_s)

    pitch_n = pitch_n[mag_n > np.median(mag_n)]
    pitch_s = pitch_s[mag_s > np.median(mag_s)]

    plt.figure()
    plt.plot(pitch_n[:200], label="Normal")
    plt.plot(pitch_s[:200], label="Stressed")
    plt.legend()
    plt.title("Pitch Contour Comparison")
    plt.savefig("results/pitch_comparison.png")
    plt.close()


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

print("Extracting features...")

X = []
y = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            features = extract_features(path)
            X.append(features)

            emotion_code = file.split("-")[2]

            if emotion_code in ["05", "06", "07"]:
                y.append(1)  # Stressed
            else:
                y.append(0)  # Normal

X = np.array(X)
y = np.array(y)

print("Feature extraction complete.")

# ==========================================================
# SAMPLE VISUALIZATIONS
# ==========================================================

sample_file = None
normal_file = None
stressed_file = None

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)
            emotion_code = file.split("-")[2]

            if sample_file is None:
                sample_file = full_path

            if emotion_code not in ["05", "06", "07"] and normal_file is None:
                normal_file = full_path

            if emotion_code in ["05", "06", "07"] and stressed_file is None:
                stressed_file = full_path

        if sample_file and normal_file and stressed_file:
            break
    if sample_file and normal_file and stressed_file:
        break

save_waveform(sample_file)
save_spectrogram(sample_file)
save_mfcc_trajectory(sample_file)

print("Basic visualizations saved.")

# ==========================================================
# FEATURE COMPARISON (Mean)
# ==========================================================

normal_features = X[y == 0]
stressed_features = X[y == 1]

normal_mean = np.mean(normal_features, axis=0)
stressed_mean = np.mean(stressed_features, axis=0)

plt.figure()
plt.plot(normal_mean, label="Normal")
plt.plot(stressed_mean, label="Stressed")
plt.legend()
plt.title("Feature Comparison: Normal vs Stressed")
plt.savefig("results/feature_comparison.png")
plt.close()

print("Feature comparison saved.")

# ==========================================================
# TIME-SERIES VARIANCE ANALYSIS
# ==========================================================

normal_var, stressed_var = mfcc_variance_analysis(dataset_path)

plt.figure()
plt.boxplot([normal_var, stressed_var], labels=["Normal", "Stressed"])
plt.title("MFCC Variance Comparison")
plt.savefig("results/mfcc_variance_comparison.png")
plt.close()

normal_e, stressed_e = energy_variance_analysis(dataset_path)

plt.figure()
plt.boxplot([normal_e, stressed_e], labels=["Normal", "Stressed"])
plt.title("Energy Variance Comparison")
plt.savefig("results/energy_variance_comparison.png")
plt.close()

pitch_contour_plot(normal_file, stressed_file)

print("Advanced time-series analysis saved.")

# ==========================================================
# MACHINE LEARNING MODEL
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/stress_model.pkl")

print("Model trained and saved.")

# ==========================================================
# EVALUATION
# ==========================================================

pred = model.predict(X_test)

report = classification_report(y_test, pred)
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")
print("ALL outputs stored inside 'results/' folder.")