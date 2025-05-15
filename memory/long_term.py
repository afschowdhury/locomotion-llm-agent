"""
File: memory/long_term.py
Description: Implements the LongTermMemory class using ChromaDB for vector and metadata storage.
"""
from utils.logging_config import setup_logging
ltm_logger = setup_logging("ltm_logger")

import chromadb
from chromadb.config import Settings
from datetime import datetime
import json
from uuid import uuid4
from config.config import Config

class LongTermMemory:
    """
    Manages long-term memory using ChromaDB for vector and metadata storage.
    """
    def __init__(self, class_name="MemoryEvent", embedding_dim=None,
                 safety_modes=None):
        """
        Initializes the LongTermMemory system with adaptive memory management.

        Args:
            class_name (str): Base name for collections.
            embedding_dim (int, optional): Dimension of embeddings.
            safety_modes (list, optional): Safety-critical modes.
        """
        self.client = chromadb.HttpClient(host='localhost', port=8000)
        self.class_name = class_name
        self.embedding_dim = embedding_dim or Config.COMBINED_EMBEDDING_DIM
        self.safety_critical_modes = safety_modes or Config.LTM_SAFETY_MODES
        self._initialize_collections()

    def _initialize_collections(self):
        """Initializes the collections for memory management."""
        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            self.text_collection = self.client.get_or_create_collection(
                name=f"{self.class_name}_text",
                metadata={"hnsw:space": "cosine"}
            )
            self.image_collection = self.client.get_or_create_collection(
                name=f"{self.class_name}_image",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=self.class_name,
                metadata={"hnsw:space": "cosine"}
            )

    def clear_all_memories(self):
        """
        Deletes all memories from the ChromaDB collections and recreates them.
        """
        try:
            if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
                text_ids = self.text_collection.get()["ids"]
                if text_ids:
                    self.text_collection.delete(ids=text_ids)
                    ltm_logger.info(f"Deleted {len(text_ids)} items from {self.class_name}_text collection")
                
                image_ids = self.image_collection.get()["ids"]
                if image_ids:
                    self.image_collection.delete(ids=image_ids)
                    ltm_logger.info(f"Deleted {len(image_ids)} items from {self.class_name}_image collection")
                
                self.client.delete_collection(name=f"{self.class_name}_text")
                self.client.delete_collection(name=f"{self.class_name}_image")
                self._initialize_collections()
                ltm_logger.info("Recreated text and image collections for a fresh start")
            else:
                all_ids = self.collection.get()["ids"]
                if all_ids:
                    self.collection.delete(ids=all_ids)
                    ltm_logger.info(f"Deleted {len(all_ids)} items from {self.class_name} collection")
                
                self.client.delete_collection(name=self.class_name)
                self._initialize_collections()
                ltm_logger.info("Recreated single collection for a fresh start")
        except Exception as e:
            ltm_logger.error(f"Error clearing memories: {str(e)}")

    def _determine_category(self, metadata):
        """
        Categorizes events for adaptive memory management based on locomotion mode.

        Args:
            metadata (str): JSON string of metadata.

        Returns:
            str: Category of the event ('safety_critical' or 'routine').
        """
        locomotion_mode = json.loads(metadata).get("response", {}).get("locomotion_mode", "")
        return "safety_critical" if locomotion_mode in self.safety_critical_modes else "routine"

    def add(self, embedding, metadata, initial_importance=0.7):
        """
        Stores an event with automatic categorization and initial importance.

        Args:
            embedding: Embedding vector for the event.
            metadata (dict): Metadata for the event.
            initial_importance (float): Initial importance score.
        """
        metadata_str = json.dumps(metadata)
        category = self._determine_category(metadata_str)
        uuid = metadata.get("image_path", "").split("/")[-1].split(".")[0] or str(uuid4())

        properties = {
            "original_metadata": metadata_str,
            "importance": initial_importance,
            "timestamp": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "category": category,
            "access_count": 0
        }

        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            text_embedding = metadata.get("embeddings", {}).get("text_embedding")
            image_embedding = metadata.get("embeddings", {}).get("image_embedding")
            self.text_collection.add(
                ids=[uuid],
                embeddings=[text_embedding],
                metadatas=[properties]
            )
            self.image_collection.add(
                ids=[uuid],
                embeddings=[image_embedding],
                metadatas=[properties]
            )
        else:
            combined_embedding = metadata.get("embeddings", {}).get("combined_embedding")
            self.collection.add(
                ids=[uuid],
                embeddings=[combined_embedding],
                metadatas=[properties]
            )

    def retrieve(self, query_embedding, top_k=5, current_discrepancy=0.5):
        """
        Retrieves and re-ranks events with usage-based reinforcement.

        Args:
            query_embedding: Embedding vector for querying.
            top_k (int): Number of top results to return.
            current_discrepancy (float): Current discrepancy score.

        Returns:
            list: List of events sorted by composite score.
        """
        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            w_text = 1 - current_discrepancy
            w_image = current_discrepancy
            ltm_logger.info(f"current_discrepancy: {current_discrepancy}")
            ltm_logger.info(f"w_text: {w_text}, w_image: {w_image}")
            
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding["text_embedding"]],
                n_results=top_k * 3,
                include=["metadatas", "distances"]
            )
            image_results = self.image_collection.query(
                query_embeddings=[query_embedding["image_embedding"]],
                n_results=top_k * 3,
                include=["metadatas", "distances"]
            )
            
            all_ids = set(text_results["ids"][0] + image_results["ids"][0])
            
            events = {}
            for i, id in enumerate(text_results["ids"][0]):
                events[id] = {
                    "text_distance": text_results["distances"][0][i],
                    "metadata": text_results["metadatas"][0][i]
                }
            for i, id in enumerate(image_results["ids"][0]):
                if id in events:
                    events[id]["image_distance"] = image_results["distances"][0][i]
                else:
                    events[id] = {
                        "image_distance": image_results["distances"][0][i],
                        "metadata": image_results["metadatas"][0][i]
                    }
            
            for id in all_ids:
                if "text_distance" not in events[id]:
                    events[id]["text_distance"] = 1.0
                if "image_distance" not in events[id]:
                    events[id]["image_distance"] = 1.0
                
                text_sim = 1 - events[id]["text_distance"]
                image_sim = 1 - events[id]["image_distance"]
                similarity_score = w_text * text_sim + w_image * image_sim
                events[id]["similarity_score"] = similarity_score
        else:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3,
                include=["metadatas", "distances"]
            )
            events = {}
            for i, id in enumerate(results["ids"][0]):
                similarity_score = 1 - results["distances"][0][i]
                events[id] = {
                    "similarity_score": similarity_score,
                    "metadata": results["metadatas"][0][i]
                }
        
        scored_events = []
        for id, event in events.items():
            metadata_str = event["metadata"]["original_metadata"]
            metadata = json.loads(metadata_str)
            importance = event["metadata"].get("importance", 0.5)
            discrepancy = metadata.get("scores", {}).get("discrepancy", 0)
            vagueness = metadata.get("scores", {}).get("vagueness", 0)
            confidence = metadata.get("response", {}).get("reasoning", {}).get("confidence", 0.0)
            category = event["metadata"].get("category", "routine")
            
            penalty = (discrepancy * Config.LTM_DISCREPANCY_PENALTY_WEIGHT + 
                       vagueness * Config.LTM_VAGUENESS_PENALTY_WEIGHT)
            if category == "safety_critical":
                penalty *= Config.LTM_SAFETY_CRITICAL_PENALTY_REDUCTION
            
            composite_score = (
                event["similarity_score"] * Config.LTM_COMPOSITE_SIMILARITY_WEIGHT +
                importance * Config.LTM_COMPOSITE_IMPORTANCE_WEIGHT +
                confidence * Config.LTM_COMPOSITE_CONFIDENCE_WEIGHT -
                penalty
            )
            
            scored_events.append({
                "id": id,
                "composite_score": composite_score,
                "metadata": metadata,
                "importance": importance,
                "similarity_score": event["similarity_score"],
                "confidence": confidence
            })
        
        scored_events.sort(key=lambda x: x["composite_score"], reverse=True)
        final_results = scored_events[:top_k]
        
        for event in final_results:
            self._boost_importance(event["id"])
        
        return [{
            "metadata": e["metadata"],
            "composite_score": e["composite_score"],
            "similarity_score": e["similarity_score"],
            "importance": e["importance"],
            "confidence": e["confidence"]
        } for e in final_results]

    def _boost_importance(self, uuid):
        """
        Reinforces frequently accessed memories by boosting their importance.

        Args:
            uuid (str): Unique identifier of the memory event.
        """
        try:
            if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
                collections = [self.text_collection, self.image_collection]
            else:
                collections = [self.collection]
            
            for collection in collections:
                result = collection.get(ids=[uuid], include=["metadatas"])
                if result["ids"]:
                    metadata = result["metadatas"][0]
                    current_importance = metadata.get("importance", 0.5)
                    category = metadata.get("category", "routine")
                    access_count = metadata.get("access_count", 0)
                    
                    threshold = Config.LTM_BOOST_DELAY_THRESHOLD
                    if access_count < threshold:
                        new_count = access_count + 1
                        collection.update(
                            ids=[uuid],
                            metadatas=[{
                                "access_count": new_count,
                                "last_accessed": datetime.now().isoformat()
                            }]
                        )
                        ltm_logger.debug(f"Access count for {uuid} is {new_count}/{threshold}: not boosting importance yet.")
                        continue
                    
                    new_importance = current_importance * Config.LTM_FREQUENCY_BOOST_WEIGHT
                    if category == "safety_critical":
                        new_importance *= Config.LTM_SAFETY_CRITICAL_BOOST_WEIGHT
                    new_importance = min(new_importance, 1.0)
                    
                    new_count = access_count + 1
                    
                    collection.update(
                        ids=[uuid],
                        metadatas=[{
                            "importance": new_importance,
                            "access_count": new_count,
                            "last_accessed": datetime.now().isoformat()
                        }]
                    )
                    ltm_logger.debug(f"Boosted importance for {uuid}: {current_importance:.2f} → {new_importance:.2f}")
        except Exception as e:
            ltm_logger.error(f"Error boosting importance for {uuid}: {str(e)}")

    def decay_importance(self, batch_size=100):
        """
        Applies adaptive importance decay with safety protections.

        Args:
            batch_size (int): Number of entries to process in each batch.
        """
        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            collection = self.text_collection
            collections_to_update = [self.text_collection, self.image_collection]
        else:
            collection = self.collection
            collections_to_update = [self.collection]
        
        offset = 0
        while True:
            result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            if not result["ids"]:
                break
            offset += batch_size
            
            for id, metadata in zip(result["ids"], result["metadatas"]):
                try:
                    timestamp_str = metadata.get("timestamp")
                    if timestamp_str is None:
                        ltm_logger.warning(f"Skipping {id} - Missing timestamp.")
                        continue
                    timestamp = datetime.fromisoformat(timestamp_str)
                    days_elapsed = (datetime.now() - timestamp).days
                    current_importance = metadata.get("importance", 0.5)
                    category = metadata.get("category", "routine")
                    
                    if category == "safety_critical":
                        decay_rate = Config.LTM_DECAY_RATE_SAFETY_CRITICAL
                        min_importance = Config.LTM_MIN_IMPORTANCE_SAFETY_CRITICAL
                    else:
                        decay_rate = Config.LTM_DECAY_RATE_ROUTINE
                        min_importance = Config.LTM_MIN_IMPORTANCE_ROUTINE
                    
                    new_importance = max(
                        current_importance * (1 - decay_rate) ** days_elapsed,
                        min_importance
                    )
                    
                    if abs(new_importance - current_importance) > 0.01:
                        for coll in collections_to_update:
                            coll.update(
                                ids=[id],
                                metadatas=[{"importance": new_importance}]
                            )
                        ltm_logger.debug(f"Decayed importance for {id}: {current_importance:.3f} -> {new_importance:.3f}")
                    else:
                        ltm_logger.debug(f"No significant change for {id}, skipping update.")
                except Exception as e:
                    ltm_logger.error(f"Error processing {id}: {str(e)}")

    def prune(self, importance_threshold=None, protected_categories=["safety_critical"]):
        """
        Safely prunes memory with category protections.

        Args:
            importance_threshold (float, optional): Threshold below which memories are pruned.
            protected_categories (list): Categories of memories that are protected from pruning.
        """
        importance_threshold = importance_threshold or Config.LTM_PRUNE_IMPORTANCE_THRESHOLD
        if Config.USE_SEPARATE_EMBEDDINGS_FOR_RETRIEVAL:
            collection = self.text_collection
            collections_to_delete = [self.text_collection, self.image_collection]
        else:
            collection = self.collection
            collections_to_delete = [self.collection]
        
        offset = 0
        while True:
            result = collection.get(limit=100, offset=offset, include=["metadatas"])
            if not result["ids"]:
                break
            offset += 100
            
            for id, metadata in zip(result["ids"], result["metadatas"]):
                try:
                    current_importance = metadata.get("importance", 0.5)
                    category = metadata.get("category", "routine")
                    
                    if category in protected_categories:
                        ltm_logger.debug(f"Skipping {id} - Protected category: {category}")
                        continue
                    
                    if current_importance < importance_threshold:
                        for coll in collections_to_delete:
                            coll.delete(ids=[id])
                        ltm_logger.debug(f"Deleted {id} - Importance {current_importance:.3f} below threshold {importance_threshold:.3f}")
                    else:
                        ltm_logger.debug(f"Keeping {id} - Importance {current_importance:.3f} above threshold.")
                except Exception as e:
                    ltm_logger.error(f"Error processing {id}: {str(e)}")