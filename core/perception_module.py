"""
File: core/perception_module.py
Description: Processes commands, encodes images, and interacts with GPT.
"""
from gpt.construct_prompt import extract_stm_context
import json
import core.utils.embedding_utils as embedding_utils
from utils.logging_config import setup_logging
from datetime import datetime
from config.config import Config

logger = setup_logging()

def perception_module(command, image_path, gpt_interface, short_term_memory, 
                     prompt_path=None, schema_path=None, timestamp=None):
    """
    Processes commands, encodes images, and interacts with GPT to generate perception data.

    Args:
        command (str): Command to process.
        image_path (str): Path to the image.
        gpt_interface: Interface for GPT interactions.
        short_term_memory: Short-term memory instance.
        prompt_path (str, optional): Path to the prompt file.
        schema_path (str, optional): Path to the schema file.
        timestamp (datetime, optional): Timestamp for the operation.

    Returns:
        dict: Processed perception data.
    """
    if short_term_memory is None:
        prompt_path = prompt_path or f"{Config.PROMPT_DIR}/perception_prompt_stm_ablation.txt"
    else:
        prompt_path = prompt_path or f"{Config.PROMPT_DIR}/perception_prompt.txt"
    
    schema_path = schema_path or f"{Config.SCHEMA_DIR}/perception_schema.json"
    timestamp = timestamp or datetime.now().astimezone()

    raw_prompt = gpt_interface.load_prompt(prompt_path)
    stm_summary = extract_stm_context(short_term_memory) if short_term_memory else ""
    rendered_prompt = gpt_interface.generate_prompt(raw_prompt, command=command, stm_summary=stm_summary)
    logger.info(f"Generated Prompt: {rendered_prompt}")

    encoded_image = gpt_interface.encode_image(image_path)
    json_schema = gpt_interface.load_schema(schema_path)

    response = gpt_interface.send_gpt_request(
        prompt=rendered_prompt,
        encoded_image=encoded_image,
        json_schema=json_schema,
        max_tokens=600,
        temperature=0.7,
    )

    gpt_content = response.choices[0].message.content
    parsed_response = json.loads(gpt_content)

    object_details_text = ""
    if Config.INCLUDE_OBJECT_DETAILS_TEXT:
        objects_info = parsed_response.get("objects_and_obstacles", {})
        primary_obj = objects_info.get("primary_object", "Unknown")
        secondary_objs = objects_info.get("secondary_objects", [])
        spatial_relation = objects_info.get("spatial_relationship", "Not specified")
        secondary_str = ", ".join(secondary_objs) if secondary_objs else "None"
        object_details_text = (
            f"{primary_obj}. "
            f"[{secondary_str}]. "
            # f"{spatial_relation}."
        )

    if Config.USE_COT_FOR_TEXT_EMBEDDING:
        base_text = parsed_response.get("reasoning", {}).get("chain_of_thought", command)
    else:
        base_text = command

    text_for_embedding = f"{base_text} {object_details_text}".strip()

    embeddings = embedding_utils.generate_combined_embedding(text_for_embedding, image_path)
    scores = parsed_response.get("scores", {})

    return {
        "timestamp": timestamp,
        "command": command,
        "image_path": image_path,
        "response": parsed_response,
        "scores": scores,
        "embeddings": embeddings,
        "generated_prompt": rendered_prompt
    }
