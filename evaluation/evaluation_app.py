"""
File: evaluation/evaluation_app.py
Description: Interactive test script for locomotion agent using Streamlit.
"""

import json
import os
import openai
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
from core.agent import LocomotionAgent
from utils.logging_config import setup_logging
from evaluation.evaluate import Evaluator
import shutil
import base64
from config.config import Config

if 'session_id' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not Config.USE_STM and not Config.USE_LTM:
        memory_suffix = "NoMem"
    elif Config.USE_STM and Config.USE_LTM:
        memory_suffix = "STM_LTM"
    elif Config.USE_STM and not Config.USE_LTM:
        memory_suffix = "STMOnly"
    else:
        memory_suffix = "LTMOnly"
    
    st.session_state.session_id = f"{timestamp}_{memory_suffix}"

logger = setup_logging(session_id=st.session_state.session_id)

def create_test_timestamp(time_str: str) -> datetime:
    """
    Converts a timestamp string formatted as 'HH:MM:SS.ms' (or 'MM:SS.ms')
    to a datetime object set for today.
    """
    base_time = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        total_seconds = int(minutes) * 60 + float(seconds)
    else:
        total_seconds = float(time_str)
    return base_time + timedelta(seconds=total_seconds)

def main():
    st.title("Locomotion Agent Testing")
    
    memory_config = []
    if Config.USE_STM:
        memory_config.append("STM")
    if Config.USE_LTM:
        memory_config.append("LTM")
    
    if memory_config:
        st.info(f"Memory configuration: {' + '.join(memory_config)}")
    else:
        st.warning("Memory disabled (NoMem configuration)")

    try:
        openai_client = openai.OpenAI()
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.info("Please make sure OPENAI_API_KEY environment variable is set correctly")
        return

    st.sidebar.header("Configuration")
    json_path = st.sidebar.text_input("JSON Path", value=Config.DATA_PATH)
    image_dir = st.sidebar.text_input("Image Directory", value=Config.IMAGE_DIR)
    mode = st.sidebar.selectbox(
        "Mode",
        [
            "Mode 1: Manual",
            "Mode 2: Auto (clear only)",
            "Mode 3: Auto (non safety-critical)",
            "Mode 4: Fully Auto",
            "Mode 5: Auto (refined only)"
        ]
    )

    if 'started' not in st.session_state:
        st.session_state.started = False
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = Evaluator()
    if 'evaluator_raw' not in st.session_state:
        st.session_state.evaluator_raw = Evaluator()
    if 'sample_results' not in st.session_state:
        st.session_state.sample_results = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_timestamp' not in st.session_state:
        st.session_state.current_timestamp = None

    if st.session_state.data is None:
        try:
            with open(json_path, "r") as f:
                st.session_state.data = json.load(f)
            st.success(f"Successfully loaded {len(st.session_state.data)} samples from {json_path}")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    if not st.session_state.started:
        if st.button("Start Testing"):
            st.session_state.started = True
            st.session_state.agent = LocomotionAgent("Agent1", openai_client)
            st.session_state.agent.ltm.clear_all_memories()
            st.rerun()
        return

    if st.session_state.current_index < len(st.session_state.data):
        row = st.session_state.data[st.session_state.current_index]
        
        st.header(f"Sample {st.session_state.current_index + 1}/{len(st.session_state.data)}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sample Details:**")
            st.write(f"ID: {row['id']}")
            st.write(f"Command: {row['command']}")
            st.write(f"Command Type: {row.get('command_type', 'clear')}")
            st.write(f"True Label: {row['locomotion_mode']}")
            st.write(f"Timestamp: {row['timestamp_start']}")

        with col2:
            image_path = f"{image_dir}/{row['id']}.jpg"
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"Image ID: {row['id']}")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

        if st.session_state.current_result is None:
            st.session_state.current_timestamp = create_test_timestamp(row['timestamp_start'])
            with st.spinner("Processing..."):
                result_data = st.session_state.agent.run(
                    command=row['command'],
                    image_path=image_path,
                    prompt_path=None,
                    schema_path=None,
                    timestamp=st.session_state.current_timestamp,
                    use_stm=Config.USE_STM,
                    use_ltm=Config.USE_LTM
                )
                st.session_state.current_result = result_data

        if st.session_state.current_result is not None:
            result_data = st.session_state.current_result
            raw_result = result_data["raw_result"]
            final_result = result_data["final_result"]
            
            raw_predicted_label = raw_result.get("response", {}).get("locomotion_mode", "Unknown")
            raw_discrepancy = raw_result.get("response", {}).get("scores", {}).get("discrepancy", None)
            raw_clarity_score = raw_result.get("clarity_score", None)
            predicted_label = final_result.get("response", {}).get("locomotion_mode", "Unknown")
            refined = final_result.get("refined", False)
            clarity_score = final_result.get("clarity_score")
            confidence = final_result.get("response", {}).get("reasoning", {}).get("confidence")

            st.markdown(
                f"""
                <hr>
                <div style="font-size: 13px;">
                  <p>Predicted Label (Perception): {raw_predicted_label}</p>
                  <p>Perception Discrepancy: {raw_discrepancy}</p>
                  <p>Clarity Score (Perception): {raw_clarity_score}</p>
                  <p>Refined: <span style="color: {'green' if refined else 'red'};">{refined}</span></p>
                  <p>Predicted Label (Final): {predicted_label}</p>
                  <p>Confidence (Final): {confidence}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Raw Result (Perception Only)")
            st.json(raw_result)

            st.subheader("Final Result")
            st.json(final_result)

            st.write("**Generated Prompts:**")
            with st.expander("Show Prompts", expanded=False):
                st.markdown("*Perception Prompt:*")
                if final_result.get('refined', False):
                    st.markdown(f"```\n{final_result.get('perception_prompt', '')}\n```")
                else:
                    st.markdown(f"```\n{final_result.get('generated_prompt', '')}\n```")
                
                if final_result.get('refined', False):
                    st.markdown("*Refinement Prompt:*")
                    st.markdown(f"```\n{final_result.get('generated_prompt', '')}\n```")


            raw_refined = False
            raw_confidence = raw_result.get("response", {}).get("reasoning", {}).get("confidence")

            current_command_type = row.get("command_type", "clear").strip().lower()

            auto_process = False
            if mode == "Mode 2: Auto (clear only)" and current_command_type == "clear":
                auto_process = True
            elif mode == "Mode 3: Auto (non safety-critical)" and current_command_type != "safety-critical":
                auto_process = True
            elif mode == "Mode 4: Fully Auto":
                auto_process = True
            elif mode == "Mode 5: Auto (refined only)" and not final_result.get("refined", False):
                auto_process = True

            if auto_process:
                st.info("Automatically advancing based on selected mode.")
                st.session_state.evaluator.add_sample(
                    row['locomotion_mode'],
                    predicted_label,
                    row.get('command_type', 'clear'),
                    refined=refined,
                    clarity_score=clarity_score,
                    confidence=confidence
                )
                st.session_state.evaluator_raw.add_sample(
                    row['locomotion_mode'],
                    raw_predicted_label,
                    row.get('command_type', 'clear'),
                    refined=raw_refined,
                    clarity_score=raw_clarity_score,
                    confidence=raw_confidence
                )
                st.session_state.sample_results.append({
                    "id": row['id'],
                    "command": row['command'],
                    "true_label": row['locomotion_mode'],
                    "predicted_label": predicted_label,
                    "command_type": row.get('command_type', 'clear'),
                    "timestamp": st.session_state.current_timestamp.isoformat(),
                    "raw_result": raw_result,
                    "final_result": final_result
                })
                st.session_state.current_result = None
                st.session_state.current_timestamp = None
                st.session_state.current_index += 1
                st.rerun()
            else:
                try:
                    with open("/Users/ehsan/Downloads/beep-30b.mp3", "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    alert_sound_html = f"""
                    <audio autoplay>
                      <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    </audio>
                    """
                    st.markdown(alert_sound_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error("Error playing sound: " + str(e))
                
                if st.button("Next Sample"):
                    st.session_state.evaluator.add_sample(
                        row['locomotion_mode'],
                        predicted_label,
                        row.get('command_type', 'clear'),
                        refined=refined,
                        clarity_score=clarity_score,
                        confidence=confidence
                    )
                    st.session_state.evaluator_raw.add_sample(
                        row['locomotion_mode'],
                        raw_predicted_label,
                        row.get('command_type', 'clear'),
                        refined=raw_refined,
                        clarity_score=raw_clarity_score,
                        confidence=raw_confidence
                    )
                    st.session_state.sample_results.append({
                        "id": row['id'],
                        "command": row['command'],
                        "true_label": row['locomotion_mode'],
                        "predicted_label": predicted_label,
                        "command_type": row.get('command_type', 'clear'),
                        "timestamp": st.session_state.current_timestamp.isoformat(),
                        "raw_result": raw_result,
                        "final_result": final_result
                    })
                    st.session_state.current_result = None
                    st.session_state.current_timestamp = None
                    st.session_state.current_index += 1
                    st.rerun()

    else:
        st.success("Testing completed!")
        
        evaluation_final = {
            "overall_report": st.session_state.evaluator.get_overall_report(),
            "command_type_reports": st.session_state.evaluator.get_command_type_reports(),
            "clarity_stats": st.session_state.evaluator.get_clarity_score_distribution(),
            "ltm_engagement_rate": st.session_state.evaluator.get_ltm_hit_rate(),
            "brier_score": st.session_state.evaluator.get_brier_score(),
            "ece": st.session_state.evaluator.get_ece(),
            "sample_results": st.session_state.sample_results
        }
        
        evaluation_raw = {
            "overall_report": st.session_state.evaluator_raw.get_overall_report(),
            "command_type_reports": st.session_state.evaluator_raw.get_command_type_reports(),
            "clarity_stats": st.session_state.evaluator_raw.get_clarity_score_distribution(),
            "brier_score": st.session_state.evaluator_raw.get_brier_score(),
            "ece": st.session_state.evaluator_raw.get_ece(),
            "sample_results": [
                {
                    "id": sample["id"],
                    "command": sample["command"],
                    "true_label": sample["true_label"],
                    "predicted_label": sample["raw_result"]["response"]["locomotion_mode"],
                    "command_type": sample["command_type"],
                    "timestamp": sample["timestamp"],
                    "raw_result": sample["raw_result"]
                }
                for sample in st.session_state.sample_results
            ]
        }
        
        final_results_folder = os.path.join("results", st.session_state.session_id)
        os.makedirs(final_results_folder, exist_ok=True)
        
        evaluation_final_file = os.path.join(final_results_folder, "evaluation_final.json")
        with open(evaluation_final_file, "w") as f:
            json.dump(evaluation_final, f, indent=4, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        
        evaluation_raw_file = os.path.join(final_results_folder, "evaluation_raw.json")
        with open(evaluation_raw_file, "w") as f:
            json.dump(evaluation_raw, f, indent=4, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        
        try:
            shutil.copy(json_path, os.path.join(final_results_folder, os.path.basename(json_path)))
        except Exception as e:
            st.error(f"Error copying data JSON file: {str(e)}")
            
        try:
            shutil.copy("config/config.py", os.path.join(final_results_folder, "config.py"))
        except Exception as e:
            st.error(f"Error copying config.py: {str(e)}")
        
        st.write("**Evaluation for Final Results:**")
        st.json(evaluation_final)
        st.info(f"Aggregated final evaluation stored in {evaluation_final_file}")
        st.write("**Evaluation for Raw Results (Perception-only):**")
        st.json(evaluation_raw)
        st.info(f"Aggregated raw evaluation stored in {evaluation_raw_file}")

if __name__ == "__main__":
    main()