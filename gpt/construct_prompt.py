"""
File: gpt/construct_prompt.py
Description: Helper functions for constructing prompts.
"""

import datetime

def extract_stm_context(stm_memory):
    """
    Constructs a memory summary for the prompt based on available Short-Term Memory (STM).

    Args:
        stm_memory (list or object with 'memory' attribute): A list of STM entries or a ShortTermMemory object.

    Returns:
        str: Formatted memory summary for the prompt.
    """
    if stm_memory is None:
        return "No recent memory available."


    if not isinstance(stm_memory, list) and hasattr(stm_memory, "memory"):
        stm_memory = stm_memory.memory

    if not stm_memory:
        return "No recent memory available."

    stm_memory = sorted(stm_memory, key=lambda x: x.get("timestamp"), reverse=True)
    summary_parts = []
    for idx, entry in enumerate(stm_memory, start=1):
        response = entry.get("response", {})
        if not response:
            summary_parts.append(f"- Missing data for memory entry {idx}.")
            continue

        locomotion_mode = response.get("locomotion_mode", f"Unknown locomotion {idx}")
        environment = response.get("scene_context", {}).get("environment", f"Unknown environment {idx}")
        primary_object = response.get("objects_and_obstacles", {}).get("primary_object", f"Unknown object {idx}")
        terrain_type = response.get("terrain_features", {}).get("type", f"Unknown terrain {idx}")
        terrain_condition = response.get("terrain_features", {}).get("condition", f"Unknown condition {idx}")
        timestamp = entry.get("timestamp", f"Unknown timestamp {idx}")

        summary_parts.append(
            f"- At {timestamp}: {locomotion_mode} in a {environment} environment, interacting with a {primary_object}."
        )

    return "\n".join(summary_parts)


def extract_ltm_context(ltm_memory):
    """
    Constructs a summarized memory context for LTM.

    Args:
        ltm_memory (list): List of LTM entries.

    Returns:
        str: Summarized LTM context for the prompt.
    """
    if not ltm_memory:
        return "No relevant memory available."

    summary_parts = ["Summary of relevant memories (the order does not imply sequence):"]
    
    for idx, entry in enumerate(ltm_memory, 1):
        metadata = entry.get("metadata", {})
        response = metadata.get("response", {})
        scores = metadata.get("scores", {})
        
        locomotion_mode = response.get("locomotion_mode", "Unknown locomotion")
        environment = response.get("scene_context", {}).get("environment", "Unknown environment")
        primary_object = response.get("objects_and_obstacles", {}).get("primary_object", f"Unknown object {idx}")
        confidence = response.get("reasoning", {}).get("confidence", 0.0)
        discrepancy = scores.get("discrepancy", 0.0)
        composite_score = entry.get("composite_score", 0.0)
        timestamp = metadata.get("timestamp", "")
        short_summary = response.get("short_summary", "")


        summary_parts.append(
            f"-  {locomotion_mode}\n"
            f"  •	Summary: {short_summary}\n"
            # f"  - Relevance Score: {composite_score:.2f}\n"
            # f"  - Primary Object: {primary_object}\n"
            # f"  - Confidence: {confidence:.1f}/1.0\n"
            # f"  - Discrepancy (Command-Scene Mismatch): {discrepancy:.1f}/1.0\n"
            # f"  - Observed: {timestamp}"
        )

    return "\n\n".join(summary_parts)