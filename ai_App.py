import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Stress Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI-Assisted Stress Detection")
st.caption("Using Facial Expression + Lifestyle Factors")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5")

model = load_emotion_model()
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📷 Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------
st.subheader("📊 Lifestyle Inputs")

sleep_val = st.slider("😴 Sleep Quality / Hours", 0, 10, 5)
workload_val = st.slider("📚 Workload Level", 0, 10, 5)
screen_val = st.slider("📱 Screen Time", 0, 10, 5)
physical_val = st.slider("🏃 Physical Activity", 0, 10, 5)
social_val = st.slider("👥 Social Interaction", 0, 10, 5)

# ---------------------------------------------------
# ANALYZE
# ---------------------------------------------------
if uploaded_file and st.button("🔍 Analyze Stress", use_container_width=True):

    try:
        # READ IMAGE
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # FACE DETECTION
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            st.error("❌ No face detected. Please upload a clear front-face image.")
            st.stop()

        x, y, w, h = faces[0]

        # Draw face box
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(img_np, caption="Detected Face", use_container_width=True)

        # Crop face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        # Preprocess
        img_array = face / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)

        # ---------------------------------------------------
        # EMOTION PREDICTION
        # ---------------------------------------------------
        pred = model.predict(img_array, verbose=0)[0]
        predicted_class = labels[np.argmax(pred)]
        emotion_probs = dict(zip(labels, pred))

        st.success(f"😊 Detected Emotion: {predicted_class.title()}")

        st.write("### 📊 Emotion Confidence")
        st.bar_chart(emotion_probs)

        # Emotion weighted score
        emotion_val = float(
            emotion_probs['happy'] * 2 +
            emotion_probs['neutral'] * 5 +
            emotion_probs['surprise'] * 4 +
            emotion_probs['sad'] * 8 +
            emotion_probs['angry'] * 9 +
            emotion_probs['fear'] * 9 +
            emotion_probs['disgust'] * 7
        )

        # ---------------------------------------------------
        # FUZZY SYSTEM
        # ---------------------------------------------------
        emotion = ctrl.Antecedent(np.arange(0, 11, 1), 'emotion')
        sleep = ctrl.Antecedent(np.arange(0, 11, 1), 'sleep')
        workload = ctrl.Antecedent(np.arange(0, 11, 1), 'workload')
        screen = ctrl.Antecedent(np.arange(0, 11, 1), 'screen')
        physical = ctrl.Antecedent(np.arange(0, 11, 1), 'physical')
        social = ctrl.Antecedent(np.arange(0, 11, 1), 'social')
        stress = ctrl.Consequent(np.arange(0, 11, 1), 'stress')

        for var in [emotion, sleep, workload, screen, physical, social, stress]:
            var['low'] = fuzz.trimf(var.universe, [0, 0, 5])
            var['medium'] = fuzz.trimf(var.universe, [3, 5, 7])
            var['high'] = fuzz.trimf(var.universe, [5, 10, 10])

        rules = [
            ctrl.Rule(emotion['high'] & workload['high'], stress['high']),
            ctrl.Rule(sleep['low'] & workload['high'], stress['high']),
            ctrl.Rule(screen['high'] & physical['low'], stress['high']),
            ctrl.Rule(social['low'] & emotion['high'], stress['high']),

            ctrl.Rule(emotion['low'] & sleep['high'] & workload['low'], stress['low']),
            ctrl.Rule(physical['high'] & social['high'], stress['low']),
            ctrl.Rule(screen['low'] & sleep['high'], stress['low']),

            ctrl.Rule(workload['medium'] & sleep['medium'], stress['medium']),
            ctrl.Rule(emotion['medium'] & social['medium'], stress['medium']),
            ctrl.Rule(screen['medium'] & workload['medium'], stress['medium']),
        ]

        system = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(system)

        sim.input['emotion'] = min(emotion_val * 1.5, 10)
        sim.input['sleep'] = sleep_val
        sim.input['workload'] = workload_val
        sim.input['screen'] = screen_val
        sim.input['physical'] = physical_val
        sim.input['social'] = social_val

        sim.compute()
        score = sim.output['stress']

        # ---------------------------------------------------
        # RESULT
        # ---------------------------------------------------
        if score < 4:
            level = "Low"
            st.success(f"🟢 Stress Level: {level}")
        elif score < 7:
            level = "Moderate"
            st.warning(f"🟡 Stress Level: {level}")
        else:
            level = "High"
            st.error(f"🔴 Stress Level: {level}")

        st.metric("📈 Stress Score", f"{score:.2f} / 10")

        # ---------------------------------------------------
        # GRAPH
        # ---------------------------------------------------
        st.subheader("📉 Stress Membership Graph")

        x_axis = np.arange(0, 11, 0.1)

        low = fuzz.trimf(x_axis, [0, 0, 5])
        medium = fuzz.trimf(x_axis, [3, 5, 7])
        high = fuzz.trimf(x_axis, [5, 10, 10])

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(x_axis, low, label='low')
        ax.plot(x_axis, medium, label='moderate')
        ax.plot(x_axis, high, label='high')

        ax.fill_between(x_axis, 0, medium, alpha=0.3)
        ax.axvline(score, linewidth=4)

        ax.set_xlabel("stress")
        ax.set_ylabel("Membership")
        ax.legend()

        st.pyplot(fig)

        # ---------------------------------------------------
        # RECOMMENDATIONS
        # ---------------------------------------------------
        st.subheader("💡 Personalized Recommendations")

        shown = False

        if workload_val > 7:
            st.info("📚 Reduce workload and take breaks.")
            shown = True

        if sleep_val < 5:
            st.info("😴 Improve sleep schedule.")
            shown = True

        if screen_val > 6:
            st.info("📱 Reduce screen time.")
            shown = True

        if physical_val < 5:
            st.info("🏃 Increase physical activity.")
            shown = True

        if social_val < 5:
            st.info("👥 Improve social interaction.")
            shown = True

        if predicted_class in ["sad", "angry", "fear"]:
            st.info("🧘 Practice meditation or breathing exercises.")
            shown = True

        if level == "High":
            st.error("⚠️ Consider talking to a counselor or taking immediate rest.")

        if not shown:
            st.success("✅ Great balance detected. Maintain your healthy lifestyle.")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.caption("Developed using Deep Learning + Fuzzy Logic + Streamlit")