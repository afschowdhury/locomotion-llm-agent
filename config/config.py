"""
File: config/config.py
Description: Configuration settings for the agent.
"""
import os

class Config:
    # GPT Model Settings
    GPT_MODEL_NAME = "gpt-4o"
    GPT_MAX_TOKENS = 600
    GPT_PERCEPTION_TEMPERATURE = 0.7
    GPT_REFINEMENT_TEMPERATURE = 0.5

    # Memory Usage Settings
    USE_STM = True
    USE_LTM = True

    # Embedding Settings
    SINGLE_EMBEDDING_DIM = 768       
    COMBINED_EMBEDDING_DIM = 1536      
    TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"
    IMAGE_EMBEDDING_MODEL = "openai/clip-vit-large-patch14"
    IMAGE_EMBEDDING_WEIGHT = 1
    USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL = True
    USE_COT_FOR_TEXT_EMBEDDING = False
    INCLUDE_OBJECT_DETAILS_TEXT = False

    
    # Short-Term Memory (STM) Settings
    STM_RETENTION_THRESHOLD = 30

    # LTM Composite Score Weights for Retrieval
    LTM_COMPOSITE_SIMILARITY_WEIGHT = 0.65
    LTM_COMPOSITE_IMPORTANCE_WEIGHT = 0.2
    LTM_COMPOSITE_CONFIDENCE_WEIGHT = 0.15
    LTM_DISCREPANCY_PENALTY_WEIGHT = 0.3
    LTM_VAGUENESS_PENALTY_WEIGHT = 0.2
    LTM_SAFETY_CRITICAL_PENALTY_REDUCTION = 0.7
    LTM_FREQUENCY_BOOST_WEIGHT = 1.1
    LTM_SAFETY_CRITICAL_BOOST_WEIGHT = 1.2
    LTM_BOOST_DELAY_THRESHOLD = 15


    # Long-Term Memory (LTM) Maintenance Settings
    LTM_DECAY_RATE_SAFETY_CRITICAL = 0.005
    LTM_MIN_IMPORTANCE_SAFETY_CRITICAL = 0.3
    LTM_DECAY_RATE_ROUTINE = 0.03        
    LTM_MIN_IMPORTANCE_ROUTINE = 0.05
    LTM_PRUNE_IMPORTANCE_THRESHOLD = 0.1
    LTM_SAFETY_MODES = ["Construction Ladder Down Climbing", 
                        "Vertical Ladder Down Climbing", 
                        "Stair Descension", 
                        "Low Space Navigation"]
    

    # Weights for clarity score computation (used in transferring to LTM)
    # clarity_score = w1 * (1 - vagueness) + w2 * (1 - discrepancy) + w3 * confidence
    CLARITY_SCORE_WEIGHT_VAGUENESS = 0.3
    CLARITY_SCORE_WEIGHT_DISCREPANCY = 0.5
    CLARITY_SCORE_WEIGHT_CONFIDENCE = 0.2
    # CLARITY_SCORE_THRESHOLD = 0.75

    # Dynamic clarity threshold parameters:
    CLARITY_SCORE_THRESHOLD_MIN = 0.35
    CLARITY_SCORE_THRESHOLD_MAX = 0.75
    CLARITY_SCORE_THRESHOLD_RAMP_PER_CYCLE = 0.01

    # Paths
    PROMPT_DIR = "gpt/prompts"
    SCHEMA_DIR = "gpt/schemas"
    LOG_DIR = "logs"
    
    DATA_PATH = "data/data.json"
    IMAGE_DIR = "/path/to/your/images"