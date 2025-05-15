"""
File: core/agent.py
Description: Defines the LocomotionAgent class to manage memory, perception,
             and refinement workflows for locomotion agent.
"""

import datetime
from core.perception_module import perception_module
from core.refinement_module import refinement_module
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from gpt.gpt4_interface import GPTInterface
from utils.logging_config import setup_logging
from datetime import timezone
from config.config import Config
import json


logger = setup_logging()

class LocomotionAgent:
    def __init__(self, name, client):
        """
        Initializes the LocomotionAgent.

        Args:
            name (str): Unique name for the agent.
            client: GPT client instance.
        """
        self.name = name
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.gpt = GPTInterface(client=client)
        self.curr_time = None
        self.cycle_count = 0  
        logger.info(f"Initialized Locomotion Agent with name: {self.name}")

    def perceive(self, command, image_path, gpt_interface, short_term_memory, prompt_path, schema_path, timestamp=None):
        """
        Calls the perception module and returns its result.

        Args:
            command (str): Command to process.
            image_path (str): Path to the image.
            gpt_interface: Interface for GPT interactions.
            short_term_memory: Short-term memory instance.
            prompt_path (str): Path to the prompt file.
            schema_path (str): Path to the schema file.
            timestamp (datetime, optional): Timestamp for the operation.

        Returns:
            dict: Result from the perception module.
        """
        return perception_module(
            command=command,
            image_path=image_path,
            gpt_interface=gpt_interface,
            short_term_memory=short_term_memory,
            prompt_path=prompt_path,
            schema_path=schema_path,
            timestamp=timestamp
        )

    def retrieve(self, query_embedding, filter_tags=None, discrepancy_score=0.5):
        """
        Retrieves relevant context from long-term memory using embeddings.

        Args:
            query_embedding: The embedding vector to query.
            filter_tags (list, optional): Metadata filters for retrieval.
            discrepancy_score (float): The discrepancy score for retrieval.

        Returns:
            list: Retrieved results with metadata and similarity scores.
        """
        results = self.ltm.retrieve(
            query_embedding=query_embedding,
            top_k=5,
            current_discrepancy=discrepancy_score
        )
        
        formatted_results = [{
            "metadata": result["metadata"] if isinstance(result["metadata"], dict) else json.loads(result["metadata"]),
            "scores": (result["metadata"] if isinstance(result["metadata"], dict) else json.loads(result["metadata"])).get("scores", {}),
            "response": (result["metadata"] if isinstance(result["metadata"], dict) else json.loads(result["metadata"])).get("response", {}),
            "composite_score": result.get("composite_score")
        } for result in results]
        
        logger.info(f"[{self.name}] Retrieved long-term context: {formatted_results}")
        return formatted_results

    def _transfer_to_ltm(self, data: dict, is_refined: bool = False):
        """
        Transfers perception/refinement data to long-term memory.

        Args:
            data (dict): Data containing perception/refinement information.
            is_refined (bool): Indicates if the data is from a refinement step.
        """
        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            text_embedding = data.get("embeddings", {}).get("text_embedding")
            image_embedding = data.get("embeddings", {}).get("image_embedding")
            if not text_embedding or not image_embedding:
                logger.warning(f"[{self.name}] Missing text/image embeddings. Skipping LTM add.")
                return
        else:
            embedding = data.get("embeddings", {}).get("combined_embedding")
            if not embedding:
                logger.warning(f"[{self.name}] No 'combined_embedding' found. Skipping LTM add.")
                return

        ts = data.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()

        metadata = {
            "timestamp": ts,
            "command": data.get("command"),
            "image_path": data.get("image_path"),
            "response": data.get("response"),          
            "scores": data.get("scores", {}),             
            "clarity_score": data.get("clarity_score"),     
            "refined": data.get("refined", False),
            "embeddings": data.get("embeddings", {})  # Add this line to include embeddings
        }

        importance_score = data.get("scores", {}).get("importance", 0.7)
        
        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            self.ltm.add(
                embedding=None,
                metadata=metadata,
                initial_importance=importance_score
            )
        else:
            self.ltm.add(
                embedding=embedding,
                metadata=metadata,
                initial_importance=importance_score
            )
        logger.info(f"[{self.name}] Transferred event to LTM with importance={importance_score}.")

    def refine(self, command, image_path, ltm_context, stm_context, initial_perception, timestamp=None, short_term_memory=None):
        """
        Updates memory entry based on insights after retrieval.

        Args:
            command (str): Command to process.
            image_path (str): Path to the image.
            ltm_context: Context from long-term memory.
            stm_context: Context from short-term memory.
            initial_perception: Initial perception data.
            timestamp (datetime, optional): Timestamp for the operation.

        Returns:
            dict: Result from the refinement module.
        """
        return refinement_module(
            command=command,
            image_path=image_path,
            ltm_context=ltm_context,
            stm_context=stm_context,
            initial_perception=initial_perception,
            gpt_interface=self.gpt,
            timestamp=timestamp
        )

    def run(self, command, image_path, prompt_path=None, schema_path=None, timestamp=None, use_stm=True, use_ltm=True):
        """
        Executes the complete agent workflow.

        Args:
            command (str): Command to process.
            image_path (str): Path to the image.
            prompt_path (str, optional): Path to the prompt file.
            schema_path (str, optional): Path to the schema file.
            timestamp (datetime, optional): Timestamp for the operation.
            use_stm (bool): Whether to use short-term memory.
            use_ltm (bool): Whether to use long-term memory.

        Returns:
            dict: Contains raw and final results of the agent's processing.
        """
        current_time = timestamp or datetime.now().astimezone()
        
        if not use_stm:
            logger.info(f"[{self.name}] STM ablation enabled: short-term memory disabled for perception.")
        
        # Phase 1: Perception
        stm = self.stm if use_stm else None

        stm_snapshot = stm.memory.copy() if use_stm else None
        
        logger.info(f"[{self.name}] Starting perception phase")
        perception_data = self.perceive(
            command=command,
            image_path=image_path,
            gpt_interface=self.gpt,
            short_term_memory=stm,
            prompt_path=prompt_path,
            schema_path=schema_path,
            timestamp=current_time
        )
        raw_result = perception_data.copy()
        logger.debug(f"Perception data: {perception_data}")
        
        # Phase 2: STM Update
        if use_stm:
            logger.info(f"[{self.name}] Updating STM")
            self.stm.update({
                "timestamp": current_time.isoformat(),
                "command": perception_data["command"],
                "image_path": perception_data["image_path"],
                "response": perception_data["response"],
                "scores": perception_data.get("scores", {})
            })
        
        # Phase 3: STM Maintenance
        if use_stm:
            logger.info(f"[{self.name}] Pruning STM")
            self.stm.prune(current_time=current_time)
        
        # Phase 4: Decide Transfer to LTM (and refinement) using Clarity Score
        scores = perception_data.get("scores", {})
        vagueness = scores.get("vagueness", 1.0)
        discrepancy = scores.get("discrepancy", 1.0)
        confidence = perception_data.get("response", {}).get("reasoning", {}).get("confidence", 0.0)

        clarity_score = (
            Config.CLARITY_SCORE_WEIGHT_VAGUENESS * (1 - vagueness) +
            Config.CLARITY_SCORE_WEIGHT_DISCREPANCY * (1 - discrepancy) +
            Config.CLARITY_SCORE_WEIGHT_CONFIDENCE * confidence
        )
        
        effective_threshold = min(
            Config.CLARITY_SCORE_THRESHOLD_MAX,
            Config.CLARITY_SCORE_THRESHOLD_MIN + self.cycle_count * Config.CLARITY_SCORE_THRESHOLD_RAMP_PER_CYCLE
        )
        logger.info(
            f"[{self.name}] Calculated clarity_score: {clarity_score:.2f}, "
            f"using effective clarity threshold: {effective_threshold:.2f}"
        )
        perception_data["clarity_score"] = clarity_score
        raw_result = perception_data.copy()
        query_embedding = perception_data.get("embeddings", {})
        
        if clarity_score > effective_threshold:
            # Data is clear
            perception_data["refined"] = False
            if use_ltm:
                logger.info(f"[{self.name}] Data is clear; transferring directly to LTM.")
                self._transfer_to_ltm(perception_data)
            else:
                logger.info(f"[{self.name}] Data is clear; but LTM ablation enabled. Skipping LTM transfer/refinement.")
            result_data = perception_data.copy()
        else:
            # Data is ambiguous
            if use_ltm:
                logger.info(f"[{self.name}] Data is ambiguous; performing refinement as LTM is enabled.")
                ltm_context = self.retrieve(
                    query_embedding=query_embedding,
                    discrepancy_score=discrepancy
                ) if query_embedding else None
                refinement_data = self.refine(
                    command=command,
                    image_path=image_path,
                    ltm_context=ltm_context,
                    stm_context=stm_snapshot,
                    initial_perception=perception_data["response"],
                    timestamp=current_time
                )
                
                result_data = {
                    "timestamp": current_time,
                    "command": command,
                    "image_path": image_path,
                    "response": refinement_data["response"],
                    "generated_prompt": refinement_data["generated_prompt"],
                    "embeddings": perception_data.get("embeddings", {}),  # Preserve embeddings from perception
                    "scores": perception_data.get("scores", {}),          # Preserve scores from perception
                    "perception_prompt": perception_data.get("generated_prompt", ""),
                    "refined": True,
                    "clarity_score": clarity_score
                }
                self._transfer_to_ltm(result_data)
            else:
                logger.info(f"[{self.name}] Data is ambiguous; but LTM ablation enabled. Skipping refinement.")
                perception_data["refined"] = False
                result_data = perception_data.copy()

        # Phase 5: LTM Maintenance
        if use_ltm:
            logger.debug(f"[{self.name}] Performing LTM maintenance")
            self.ltm.decay_importance()
            self.ltm.prune()
        else:
            logger.info(f"[{self.name}] LTM maintenance bypassed (LTM ablation enabled).")
        
        self.cycle_count += 1
        logger.info(f"[{self.name}] Cycle complete; total cycles: {self.cycle_count}")
        return {"raw_result": raw_result, "final_result": result_data}

