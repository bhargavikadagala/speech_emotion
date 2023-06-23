import streamlit as st
import numpy as np
import librosa
import librosa.display
import keras

# Load the pre-trained emotion recognition model
model = keras.models.load_model("mlp_model_weights.h5")

# Define the emotion labels
emotion_labels = ['Angry', 'Fearful', 'Happy', 'Sad', 'Neutral']


# Function to extract features from audio
def extract_features(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, duration=3)

    # Extract features using librosa
    features = librosa.feature.mfcc(y=y, sr=sr)

    # Pad or truncate the features to a fixed length
    max_len = 128
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]

    return features


# Streamlit app
def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file and the model will predict the emotion.")

    # File uploader
    audio_file = st.file_uploader("Upload an audio file", type=['wav'])

    if audio_file is not None:
        # Convert the audio file to a format readable by librosa
        audio_path = 'temp.wav'
        with open(audio_path, 'wb') as f:
            f.write(audio_file.read())

        # Extract features from the audio file
        features = extract_features(audio_path)

        # Reshape the features for the model input
        features = np.reshape(features, (1, features.shape[0], features.shape[1], 1))

        # Predict the emotion
        emotion_probabilities = model.predict(features)[0]
        predicted_emotion = emotion_labels[np.argmax(emotion_probabilities)]

        # Display the emotion prediction
        st.write("Predicted Emotion:", predicted_emotion)

        # Display the emotion probabilities as a bar chart
        st.bar_chart(emotion_probabilities)

        # Display the audio waveform
        st.write("Audio Waveform:")
        y, sr = librosa.load(audio_path)
        st.line_chart(y)

        # Display the audio spectrogram
        st.write("Audio Spectrogram:")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y))), sr=sr, x_axis='time', y_axis='log')
        st.pyplot()

        # Remove the temporary audio file
        os.remove(audio_path)


if _name_ == "_main_":
    main()