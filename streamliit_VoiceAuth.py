import os
import sys
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
import streamlit as st
from PIL import Image
from tempfile import NamedTemporaryFile
from audioread.exceptions import NoBackendError
from moviepy.editor import AudioFileClip  # for video-to-audio conversion

from VoiceAuthBackend import (
    get_file_metadata,
    get_score_label,
    predict_vggish,
    predict_yamnet,
    save_metadata,
    typewriter_effect,
    predict_hf2,
    predict_hf,
    predict_rf,
    visualize_mfcc,
    create_mel_spectrogram,
)

# Setup environment variables
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
    if eta is not None:
        st.text(f"Estimated Time: {eta:.2f} seconds")


def extract_audio_from_video(video_file_path):
    """Extracts audio from a video file and returns the path to the audio file."""
    video = AudioFileClip(video_file_path)
    audio_path = video_file_path.replace(".mp4", ".wav")
    video.write_audiofile(audio_path)
    return audio_path


# Streamlit application layout
st.set_page_config(
    page_title="VoiceAuth - Deepfake Audio and Voice Detector",
    page_icon="images/voiceauth.webp",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
    <style>
        /* Global Styles */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7f9fc;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }
        
        /* Centering container */
        .css-1d391kg {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header styling */
        h1, h2, h3 {
            color: #333;
            text-align: center;
            font-weight: 600;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 25px;
            padding: 14px 28px;
            font-size: 16px;
            border: none;
            transition: transform 0.3s, background-color 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .stButton>button:active {
            background-color: #388e3c;
        }

        /* Card-style container for each section */
        .stMarkdown, .stText, .stTextArea {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Input Fields */
        .stTextInput, .stSelectbox, .stRadio, .stFileUploader {
            border-radius: 8px;
            border: 2px solid #ccc;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #fafafa;
        }
        .stTextInput:focus, .stSelectbox:focus, .stRadio:focus, .stFileUploader:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.4);
        }

        /* Progress bar custom styles */
        .progress-bar {
            background-color: #e0e0e0;
            height: 8px;
            border-radius: 5px;
        }
        .progress-bar .progress {
            background-color: #4CAF50;
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease-out;
        }

        /* Hover effects for links */
        a {
            color: #4CAF50;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease-in-out;
        }
        a:hover {
            color: #388e3c;
        }

        /* Custom Footer Style */
        footer {
            text-align: center;
            background-color: #f4f4f4;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 -4px 6px rgba(0,0,0,0.1);
        }

        /* Particle Background Animation */
        .particles-js {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            pointer-events: none;
        }

        /* Add some animated text effects */
        .model-title {
            font-size: 1.5em;
            color: #4CAF50;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

    </style>
""", unsafe_allow_html=True)

# Add particles.js to create particle effect on the background (using a CDN)
st.markdown("""
    <script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
    <script>
        particlesJS("particles-js", {
            "particles": {
                "number": {
                    "value": 80,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": "#4CAF50"
                },
                "shape": {
                    "type": "circle",
                    "stroke": {
                        "width": 0,
                        "color": "#000"
                    },
                    "polygon": {
                        "nb_sides": 5
                    }
                },
                "opacity": {
                    "value": 0.5,
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 1,
                        "opacity_min": 0.1
                    }
                },
                "size": {
                    "value": 3,
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 10,
                        "size_min": 0.1
                    }
                },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#4CAF50",
                    "opacity": 0.4,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 4,
                    "direction": "none",
                    "random": false,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {
                        "enable": false,
                        "rotateX": 600,
                        "rotateY": 600
                    }
                }
            },
            "interactivity": {
                "detect_on": "window",
                "events": {
                    "onhover": {
                        "enable": true,
                        "mode": "repulse"
                    },
                    "onclick": {
                        "enable": true,
                        "mode": "push"
                    }
                }
            },
            "retina_detect": true
        });
    </script>
    <div id="particles-js"></div>
""", unsafe_allow_html=True)


# Main App UI
st.title("VoiceAuth - Deepfake Audio and Voice Detector")
st.markdown("### Detect fake voices using deep learning models")
logo_image = Image.open("images/bot2.png")
st.image(logo_image, width=150)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "ogg", "flac", "aac", "m4a"],
)

# Model selection
model_option = st.radio(
    "Select Model(s)", ("Random Forest", "Melody", "960h", "All")
)

if uploaded_file:
    if st.button("Run Prediction"):
        # Create a progress bar
        progress_bar = st.progress(0)
        st.text("Processing...")

        # Temporary file handling using tempfile.NamedTemporaryFile
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            # Save uploaded file content to the temp file
            temp_file.write(uploaded_file.getbuffer())
            temp_file.seek(0)  # Reset file pointer to the beginning

            # Extract audio from video if uploaded file is a video format
            if uploaded_file.type in ['video/mp4', 'video/mkv', 'video/avi', 'video/mov', 'video/webm']:
                st.text("Extracting audio from video...")
                temp_file.name = extract_audio_from_video(uploaded_file.name)

            # Extract audio duration
            try:
                audio_length = librosa.get_duration(path=temp_file.name)
                st.text(f"Audio Duration: {audio_length:.2f} seconds")
            except NoBackendError:
                st.text("Error: No backend available to process the audio file.")
                audio_length = None

            # Start processing
            start_time = time.time()
            update_progress(progress_bar, 0.1, "Starting analysis...")

            # Run predictions
            rf_is_fake = hf_is_fake = hf2_is_fake = False
            rf_confidence = hf_confidence = hf2_confidence = 0.0
            combined_confidence = 0.0

            # Model prediction functions
            def run_rf_model():
                return predict_rf(temp_file.name)

            def run_hf_model():
                return predict_hf(temp_file.name)

            def run_hf2_model():
                return predict_hf2(temp_file.name)

            # Parallel processing for all models
            if model_option == "All":
                with ThreadPoolExecutor(max_workers=20) as executor:
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
                            st.text(f"Error in {model_name} model: {e}")

                # Combine results
                confidences = [rf_confidence, hf_confidence, hf2_confidence]
                valid_confidences = [c for c in confidences if isinstance(c, (int, float)) and c > 0]
                combined_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
                combined_result = rf_is_fake or hf_is_fake or hf2_is_fake

            # Single model predictions
            elif model_option == "Random Forest":
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

            # Update progress
            update_progress(progress_bar, 1.0, "Finalizing results...")

            # Display results
            result_text = get_score_label(combined_confidence)
            st.text(f"Confidence: {result_text} ({combined_confidence:.2f})")
            st.text("Prediction: Fake" if combined_result else "Prediction: Real")
            st.markdown("---")
            # File metadata
            file_format, file_size, audio_length, bitrate, _ = get_file_metadata(temp_file.name)
            st.text(
                f"File Format: {file_format}, Size: {file_size:.2f} MB, "
                f"Audio Length: {audio_length:.2f} sec, Bitrate: {bitrate:.2f} Mbps"
            )
            st.markdown("---")

            # Save metadata
            model_used = model_option if model_option != "All" else "Random Forest, Melody, and 960h"
            prediction_result = "Fake" if combined_result else "Real"
            save_metadata(str(uuid.uuid4()), temp_file.name, model_used, prediction_result, combined_confidence)

            # Visualizations
            mfcc_path = visualize_mfcc(temp_file.name)
            st.image(mfcc_path, caption="MFCC Visualization", use_container_width=True)
            st.markdown("---")
            mel_spectrogram_path = create_mel_spectrogram(temp_file.name)
            st.image(mel_spectrogram_path, caption="Mel Spectrogram", use_container_width=True)


def open_donate():
    donate_url = "https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD"
    webbrowser.open(donate_url)


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
