import os
import sys
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from PIL import Image
from VoiceAuthBackend import (
    get_file_metadata, get_score_label, predict_vggish, predict_yamnet,
    save_metadata, typewriter_effect, predict_hf2, predict_hf, predict_rf,
    visualize_mfcc, create_mel_spectrogram
)

def setup_environment():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
        os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
    else:
        os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")

def update_progress(progress_bar, progress, text="Processing...", eta=None):
    progress_bar.progress(progress)
    st.text(text)
    if eta:
        st.text(f"Estimated Time: {eta:.2f} seconds")

def configure_ui():
    st.set_page_config(
        page_title="VoiceAuth - Deepfake Audio and Voice Detector",
        page_icon="images/voiceauth.webp",
        layout="wide"
    )
    st.markdown("""
        <style>
            h1 { font-size: 2.5rem; color: #4CAF50; }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            .stButton>button:hover { background-color: #45a049; }
        </style>
    """, unsafe_allow_html=True)

def run_models(temp_file_path, model_option):
    results = {"Random Forest": (False, 0.0), "Melody": (False, 0.0), "960h": (False, 0.0)}
    def model_wrapper(model_func):
        return model_func(temp_file_path)
    
    if model_option == "All":
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(model_wrapper, predict_rf): "Random Forest",
                executor.submit(model_wrapper, predict_hf): "Melody",
                executor.submit(model_wrapper, predict_hf2): "960h"
            }
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    results[model_name] = future.result()
                except Exception as e:
                    st.error(f"Error in {model_name} model: {e}")
    else:
        model_mapping = {"Random Forest": predict_rf, "Melody": predict_hf, "960h": predict_hf2}
        results[model_option] = model_wrapper(model_mapping[model_option])
    
    confidences = [conf for _, conf in results.values() if conf > 0]
    combined_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    combined_result = any(is_fake for is_fake, _ in results.values())
    return combined_result, combined_confidence

def main():
    setup_environment()
    configure_ui()
    
    st.title("VoiceAuth: Deepfake Audio and Voice Detector")
    st.image(Image.open("images/bot2.png"), width=150)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac", "aac", "m4a"])
    model_option = st.radio("Select Model(s)", ("Random Forest", "Melody", "960h", "All"))
    
    if uploaded_file and st.button("Run Prediction"):
        progress_bar = st.progress(0)
        temp_file_path = os.path.join("temp_dir", str(uuid.uuid4()) + os.path.splitext(uploaded_file.name)[1])
        os.makedirs("temp_dir", exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        start_time = time.time()
        update_progress(progress_bar, 10, "Running Predictions...")
        combined_result, combined_confidence = run_models(temp_file_path, model_option)
        
        update_progress(progress_bar, 80, "Processing Results...")
        result_text = get_score_label(combined_confidence)
        st.text(f"Confidence: {result_text} ({combined_confidence:.2f})")
        st.text(f"File Metadata: {get_file_metadata(temp_file_path)}")
        typewriter_effect(st, f"Processing complete. Result: {result_text}")
        
        save_metadata(str(uuid.uuid4()), temp_file_path, model_option, result_text, combined_confidence)
        st.image(visualize_mfcc(temp_file_path), caption="MFCC Visualization", use_container_width=True)
        st.image(create_mel_spectrogram(temp_file_path), caption="Mel Spectrogram", use_container_width=True)
        update_progress(progress_bar, 100, "Complete!")
        
    st.markdown("---")
    st.expander("Contact & Support").markdown("[Email](mailto:sadiqkassamali@gmail.com)")
    donate_expander = st.expander("Donate")
    donate_expander.markdown("[Buy Me Coffee](https://buymeacoffee.com/sadiqkassamali)")
    donate_expander.markdown("[Donate](https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com)")

if __name__ == "__main__":
    main()
