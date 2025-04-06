import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model and weights
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion Detection Logic
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                max_index = int(np.argmax(prediction))
                emotion = emotion_labels[max_index]
                cv2.putText(img, emotion, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 102, 255), 2)
        return img

# --- Custom CSS ---
st.markdown("""
<style>
:root {
    --primary-bg: #1a1a2e;    /* Dark blue-gray background */
    --secondary-bg: #2d2d4a;  /* Lighter blue-gray for cards */
    --accent: #e91e63;        /* Vibrant pink for highlights */
    --text-color: #f0f2f5;    /* Light gray for text */
}

/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body, .stApp {
    background-color: var(--primary-bg);
    color: var(--text-color);
    font-family: 'Roboto', 'Segoe UI', sans-serif;
    line-height: 1.6;
}

/* Container */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Title */
h1 {
    color: var(--accent);
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 1rem;
    animation: fadeInDown 1s ease-in-out;
}

/* Subheading */
h4 {
    color: var(--text-color);
    text-align: center;
    font-weight: 400;
    opacity: 0.9;
    margin-bottom: 2rem;
}

/* Webcam Wrapper */
#cam-wrapper {
    background-color: var(--secondary-bg);
    padding: 2rem;
    border-radius: 15px;
    
    margin: 2rem auto;
    max-width: 600px;
    animation: fadeInUp 1s ease;
}

.streamlit-webrtc video {
    width: 100% !important;
    max-width: 480px;
    height: auto !important;
    border-radius: 10px;
    display: block;
    margin: 0 auto;
    
}

/* Info Box */
.info-box {
    background-color: var(--secondary-bg);
    padding: 2rem;
    border-radius: 15px;
    margin: 2rem auto;
    width: 100%;
    animation: fadeIn 1.5s ease;
}

.info-box h2 {
    color: var(--accent);
    text-align: center;
    margin-bottom: 1rem;
    font-size: 1.8rem;
}

.info-box p {
    font-size: 1rem;
    text-align: center;
    opacity: 0.9;
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        padding: 15px;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    h4 {
        font-size: 1rem;
    }
    
    #cam-wrapper {
        padding: 1.5rem;
        margin: 1.5rem auto;
    }
    
    .info-box {
        padding: 1.5rem;
        margin: 1.5rem auto;
    }
    
    .info-box h2 {
        font-size: 1.5rem;
    }
    
    .info-box p {
        font-size: 0.9rem;
    }
}

@media (max-width: 480px) {
    h1 {
        font-size: 1.8rem;
    }
    
    #cam-wrapper {
        padding: 1rem;
    }
    
    .streamlit-webrtc video {
        max-width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

# --- App UI ---
st.title("🎭 Real-Time Face Emotion Detection")
st.markdown("<h4>Express yourself — we'll detect your emotion live!</h4>", unsafe_allow_html=True)
st.write("")

# --- Info Box ---
st.markdown("""
<div class='info-box'>
    <h2>🧠 How It Works</h2>
    <p>
        This app uses your webcam to capture your face in real time. Using a deep learning model trained on facial expressions,
        it processes each frame, detects your face using OpenCV, and classifies your emotion into one of 7 categories:
        <strong>Angry</strong>, <strong>Disgust</strong>, <strong>Fear</strong>, <strong>Happy</strong>,
        <strong>Neutral</strong>, <strong>Sad</strong>, or <strong>Surprise</strong>.
    </p>
    <p>
        All processing is done locally — nothing is sent to any server. It's fast, secure, and private.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Webcam Component ---

webrtc_streamer(key="emotion-detect", video_processor_factory=VideoTransformer)


# --- About Section ---
st.markdown("""
<div class='info-box'>
    <h2>📌 About This Project</h2>
    <p>
        This application combines <strong>real-time computer vision</strong> with a <strong>Convolutional Neural Network (CNN)</strong>
        to classify human emotions through webcam input.
    </p>
    <p>
        Built using <strong>Streamlit</strong> for rapid prototyping, <strong>OpenCV</strong> for face detection,
        and <strong>Keras + TensorFlow</strong> for emotion classification — this project demonstrates the power
        of AI in understanding human behavior.
    </p>
    <p>
        Designed with a modern, animated dark UI for a smooth and visually pleasant experience.
    </p>
</div>
""", unsafe_allow_html=True)