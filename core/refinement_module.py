"""
File: core/refinement_module.py
Description: Handles the refinement process using STM and LTM context to refine initial perceptions.
"""
from gpt.construct_prompt import extract_ltm_context, extract_stm_context
from gpt.gpt4_interface import GPTInterface
from config.config import Config
import json

def refinement_module(command, image_path, ltm_context, stm_context, initial_perception, 
                     gpt_interface, prompt_path=None, schema_path=None,
                     timestamp=None):
    """
    Handles the refinement process using both LTM and STM context to refine initial perceptions.

    Args:
        command (str): Command to process.
        image_path (str): Path to the image.
        ltm_context: Context from long-term memory.
        stm_context: Context from short-term memory.
        initial_perception: Initial perception data following our standard schema.
        gpt_interface: GPT interface for model calls.
        prompt_path (str, optional): Prompt file path.
        schema_path (str, optional): JSON schema file path.
        timestamp (datetime, optional): Timestamp for the operation.

    Returns:
        dict: Refined data.
    """
    prompt_path = prompt_path or f"{Config.PROMPT_DIR}/refinement_prompt.txt"
    schema_path = schema_path or f"{Config.SCHEMA_DIR}/perception_schema.json"
    
    raw_prompt = gpt_interface.load_prompt(prompt_path)
    
    rendered_prompt = gpt_interface.generate_prompt(
        raw_prompt,
        command=command,
        stm_summary=extract_stm_context(stm_context),
        ltm_summary=extract_ltm_context(ltm_context),
        initial_perception_json=json.dumps(initial_perception)
    )
    
    encoded_image = gpt_interface.encode_image(image_path)
    
    refinement_response = gpt_interface.send_gpt_request(
        prompt=rendered_prompt,
        encoded_image=encoded_image,
        json_schema=gpt_interface.load_schema(schema_path),
        max_tokens=Config.GPT_MAX_TOKENS,
        temperature=Config.GPT_REFINEMENT_TEMPERATURE
    )
    
    refined_content = json.loads(refinement_response.choices[0].message.content)
    
    
    return {
        "timestamp": initial_perception.get("timestamp", timestamp),
        "command": command,
        "image_path": image_path,
        "response": refined_content,
        "scores": initial_perception.get("scores", {}),
        "clarity_score": initial_perception.get("clarity_score", 0.0),
        "generated_prompt": rendered_prompt,
        "refined": True
    }
