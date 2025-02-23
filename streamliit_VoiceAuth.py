import os
import sys
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
import streamlit as st
from PIL import Image

from VoiceAuthBackend import (get_file_metadata,
                              get_score_label,
                              predict_vggish, predict_yamnet,
                              save_metadata, typewriter_effect, predict_hf2,
                              predict_hf, predict_rf, visualize_mfcc, create_mel_spectrogram,
                              )

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
TF_ENABLE_ONEDNN_OPTS=0
TF_CPP_MIN_LOG_LEVEL=2
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
    os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
else:
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")


def update_progress(progress_bar, progress, text="Processing...", eta=None):
    progress_bar.progress(progress)
    st.text(text)
    if eta is not None:
        st.text(f"Estimated Time: {eta:.2f} seconds")


# Streamlit application layout
st.set_page_config(
    page_title="VoiceAuth - Deepfake Audio and Voice Detector",
    page_icon="images/voiceauth.webp",  # Adjust with your own image path
    layout="wide",  # 'wide' to utilize full screen width
    initial_sidebar_state="auto",  # Sidebar visibility
)

# Add custom CSS for additional styling
st.markdown("""
    <style>
        /* Global styles */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f0f0f0;
        }
        .css-1d391kg { 
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Sidebar styles */
        .css-ffhzg2 {
            background-color: #ffffff;
            color: #333333;
            font-family: 'Arial', sans-serif;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        /* Header styling */
        h1 {
            font-size: 3rem;
            color: #4CAF50;
        }
        h2 {
            font-size: 1.5rem;
            color: #333333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049; /* Darker green for hover effect */
        }
        /* Image styling */
        .stImage {
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .stRadio {
            font-size: 1.2rem;
        }
        .stText {
            font-size: 1.0rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("VoiceAuth: Deepfake Audio and Voice Detector")
logo_image = Image.open("images/bot2.png")  # Your logo here
st.image(logo_image, width=150)
st.markdown("""
🚀 **VoiceAuth** is here to redefine how we validate the authenticity of audio files. Whether you're a journalist, a business leader, or simply someone who values truth, VoiceAuth equips you with cutting-edge tools to detect and fight deepfake audio effortlessly.
### Who is it for?
🔊 **Media Professionals**: Ensure your audio content is credible and tamper-proof.  
🛡️ **Law Enforcement**: Authenticate voice recordings in investigations.  
📞 **Businesses**: Protect call centers and secure internal communications.  
🎓 **Educators & Researchers**: Dive into real-world machine learning and voice analytics.  
🔒 **Security Experts**: Enhance voice biometrics and authentication systems.
### Why VoiceAuth?
✅ **Detect Deepfakes with Precision**: Leverage advanced AI models, including **Random Forest** and **Hugging Face** technologies for superior accuracy.  
✅ **User-Friendly**: Intuitive interface tailored for both tech-savvy users and beginners.  
✅ **Fast & Reliable**: Real-time analysis with confidence scores, metadata extraction, and visual insights.  
✅ **Multi-Model Capability**: Use models like **Random Forest**, **Melody**, or **960h** individually or combine them for superior results.  
✅ **Portable & Secure**: Runs seamlessly on your system with no internet dependency for predictions.
""")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "ogg", "flac", "aac", "m4a", "mp4", "mov", "avi", "mkv", "webm"],
)

model_option = st.radio("Select Model(s)", ("Random Forest", "Melody", "960h", "All"))

if uploaded_file:
    if st.button("Run Prediction"):
        progress_bar = st.progress(0)
        file_uuid = str(uuid.uuid4())
        temp_dir = "temp_dir"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, os.path.basename(uploaded_file.name))

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        start_time = time.time()
        progress_bar.progress(10)

        rf_is_fake = hf_is_fake = hf2_is_fake = False
        rf_confidence = hf_confidence = hf2_confidence = 0.0
        combined_confidence = 0.0

        def run_rf_model():
            return predict_rf(temp_file_path)

        def run_hf_model():
            return predict_hf(temp_file_path)

        def run_hf2_model():
            return predict_hf2(temp_file_path)

        if model_option == "All":
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(run_rf_model): "Random Forest",
                    executor.submit(run_hf_model): "Melody",
                    executor.submit(run_hf2_model): "960h",
                }
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        if model_name == "Random Forest":
                            rf_is_fake, rf_confidence = future.result()
                        elif model_name == "Melody":
                            hf_is_fake, hf_confidence = future.result()
                        elif model_name == "960h":
                            hf2_is_fake, hf2_confidence = future.result()
                    except Exception as e:
                        st.error(f"Error in {model_name} model: {e}")

            confidences = [rf_confidence, hf_confidence, hf2_confidence]
            valid_confidences = [conf for conf in confidences if conf > 0]
            combined_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
            combined_result = rf_is_fake or hf_is_fake or hf2_is_fake
        
        else:
            if model_option == "Random Forest":
                rf_is_fake, rf_confidence = run_rf_model()
                combined_confidence = rf_confidence
                combined_result = rf_is_fake
            elif model_option == "Melody":
                hf_is_fake, hf_confidence = run_hf_model()
                combined_confidence = hf_confidence
                combined_result = hf_is_fake
            elif model_option == "960h":
                hf2_is_fake, hf2_confidence = run_hf2_model()
                combined_confidence = hf2_confidence
                combined_result = hf2_is_fake

        progress_bar.progress(80)
        result_text = get_score_label(combined_confidence)
        st.text(f"Confidence: {result_text} ({combined_confidence:.2f})")
        
        file_metadata = get_file_metadata(temp_file_path)
        st.text(f"File Metadata: {file_metadata}")
        
        typewriter_effect(st, f"Processing complete. Result: {result_text}")
        save_metadata(file_uuid, temp_file_path, model_option, result_text, combined_confidence)
        
        mfcc_path = visualize_mfcc(temp_file_path)
        st.image(mfcc_path, caption="MFCC Visualization", use_container_width=True)
        
        mel_spectrogram_path = create_mel_spectrogram(temp_file_path)
        st.image(mel_spectrogram_path, caption="Mel Spectrogram", use_container_width=True)

        progress_bar.progress(100)


def open_donate():
    donate_url = "https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD"
    webbrowser.open(donate_url)

st.markdown("---")
st.markdown("""### Transforming Industries!
🎙️ **Journalism**: Verify audio sources before publishing.  
⚖️ **Legal**: Strengthen audio evidence for court cases.  
📈 **Business**: Detect fake voice inputs in customer interactions.  
🔬 **Research**: Analyze voice patterns and expand your knowledge of machine learning.
""")
st.markdown("---")
# Contact section with modern footer
contact_expander = st.expander("Contact & Support")
contact_expander.markdown(
    "For assistance: [Email](mailto:sadiqkassamali@gmail.com)"
)
donate_expander = st.expander("Donate")
donate_expander.markdown(
    "[Buy Me Coffee](https://buymeacoffee.com/sadiqkassamali)"
)
donate_expander.markdown(
    "[Donate to Support](https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD)"
)
