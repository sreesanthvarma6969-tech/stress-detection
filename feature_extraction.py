import librosa
import numpy as np

def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    return signal, sr

def extract_mfcc(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def extract_zcr(signal):
    zcr = librosa.feature.zero_crossing_rate(signal)
    return np.mean(zcr)

def extract_centroid(signal, sr):
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    return np.mean(centroid)

def extract_energy(signal):
    rms = librosa.feature.rms(y=signal)
    return np.mean(rms)

def extract_pitch(signal, sr):
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    return np.mean(pitch) if len(pitch) > 0 else 0

def extract_features(file_path):
    signal, sr = load_audio(file_path)
    
    mfcc = extract_mfcc(signal, sr)
    zcr = extract_zcr(signal)
    centroid = extract_centroid(signal, sr)
    energy = extract_energy(signal)
    pitch = extract_pitch(signal, sr)
    
    features = np.hstack([mfcc, zcr, centroid, energy, pitch])
    return features