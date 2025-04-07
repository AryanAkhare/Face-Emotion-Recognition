import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import base64

# ---------- Load Model ----------
classifier = load_model('Face_Emotion_Recognition.h5')
classifier.load_weights("Face_Emotion_Recognition_Weights.weights.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------- Helper: Convert Image to Base64 ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------- CSS Styling with Background ----------
bg_image = get_base64_image("background.jpg")

st.markdown(f"""
<style>
:root {{
  --primary-bg: #0f0f0f;
  --secondary-bg: #1c1c1c;
  --accent: #fafafa;
  --text-color: #eaeaea;
  --muted-text: #cccccc;
  --shadow-color: rgba(0,0,0,0.4);
  --border-color: #444;
}}

.stApp {{
  background: url("data:image/jpg;base64,{bg_image}");
  background-size: cover;
  background-attachment: fixed;
  color: var(--text-color);
}}

.stApp::before {{
  content: "";
  position: fixed;
  top: 0;
  left: 0;
  height: 100%;
  width: 100%;
  background: var(--primary-bg);
  opacity: 0.9;
  z-index: -1;
}}

h1 {{
  color: var(--accent);
  text-align: center;
  font-size: 2.5rem;
}}

h4 {{
  color: var(--text-color);
  text-align: center;
  font-weight: 400;
  opacity: 0.9;
}}

.info-box {{
  background-color: transparent;
  padding: 2rem;
  
  margin: 2rem auto;
  max-width: 800px;
  ;
  border: none;
}}

.info-box h2 {{
  color: var(--accent);
  text-align: center;
  margin-bottom: 1rem;
}}

.info-box p {{
  color: var(--muted-text);
  line-height: 1.8;
  font-size: 1rem;
}}

.streamlit-webrtc video {{
  width: 100% !important;
  max-width: 500px;
  border-radius: 12px;
  margin: 1rem auto;
  display: block;
  border: 2px solid var(--border-color);
}}
</style>
""", unsafe_allow_html=True)

# ---------- Emotion Detection Logic ----------
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
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

# ---------- Streamlit App ----------
st.title("🎭 Real-Time Face Emotion Detection")
st.markdown("<h4>Express yourself — we'll detect your emotion live!</h4>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
  <h2>🧠 How It Works</h2>
  <p>This app captures live video from your webcam and uses deep learning to recognize your facial expression in real time.</p>
  <p>Here's how it works step-by-step:</p>
  <ul>
    <li><strong>1.</strong> A video frame is captured using your webcam.</li>
    <li><strong>2.</strong> <strong>OpenCV</strong> detects faces in the frame.</li>
    <li><strong>3.</strong> Each detected face is resized and passed to a trained <strong>CNN</strong> model.</li>
    <li><strong>4.</strong> The model predicts the emotion: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.</li>
    <li><strong>5.</strong> The emotion label is displayed live above your face in the video.</li>
  </ul>
  <p><em>Note: Everything runs locally in your browser — no internet processing or external servers involved.</em></p>
</div>
""", unsafe_allow_html=True)

webrtc_streamer(key="emotion-detect", video_processor_factory=VideoTransformer)

st.markdown("""
<div class='info-box'>
  <h2>📌 About This Project</h2>
  <p>This project demonstrates the power of real-time <strong>computer vision</strong> and <strong>deep learning</strong> applied to human emotion recognition.</p>
  <p>It uses a <strong>Convolutional Neural Network (CNN)</strong> trained to classify facial expressions into seven basic emotions: 
  <strong>Angry</strong>, <strong>Disgust</strong>, <strong>Fear</strong>, <strong>Happy</strong>, <strong>Neutral</strong>, <strong>Sad</strong>, and <strong>Surprise</strong>.</p>
  <p>The video feed is processed frame by frame using <strong>OpenCV</strong> to detect faces, which are then classified using a model built with <strong>Keras</strong> and powered by <strong>TensorFlow</strong>.</p>
  <p>Streamlit powers the interface — all processing is local, private, and interactive.</p>
  <p>This project is ideal for education, demonstration, and exploring how AI understands facial cues.</p>
</div>
""", unsafe_allow_html=True)
