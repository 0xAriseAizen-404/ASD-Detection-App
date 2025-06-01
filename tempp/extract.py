import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Define dataset paths
base_dir = 'C:/Users/mahes/OneDrive/Desktop/FinalYearProjects/Projects/ASD-Detection-App/app'
RECORDINGS_FOLDER = os.path.join(base_dir, "recordings")
FEATURES_FOLDER = os.path.join(base_dir, "tempp", "features")

# Ensure features folder exists
os.makedirs(FEATURES_FOLDER, exist_ok=True)

def extract_mfcc_features(audio_file, n_mfcc=40, max_len=100):
    """Extract MFCC features from an audio file and pad/truncate to a fixed length."""
    y, sr = librosa.load(audio_file, sr=22050)  # Load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to ensure (n_mfcc, max_len)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # Transpose to (max_len, n_mfcc) = (100, 40)
    mfcc = mfcc.T
    print(f"Extracted MFCC shape for {os.path.basename(audio_file)}: {mfcc.shape}")  # Debug
    return mfcc

# Save extracted features
for file in os.listdir(RECORDINGS_FOLDER):
    if file.endswith(".m4a"):
        audio_path = os.path.join(RECORDINGS_FOLDER, file)
        mfcc_features = extract_mfcc_features(audio_path)
        feature_path = os.path.join(FEATURES_FOLDER, file.replace(".m4a", ".npy"))
        np.save(feature_path, mfcc_features)  # Save as (100, 40)
        print(f"Features saved: {feature_path}, Shape: {mfcc_features.shape}")
