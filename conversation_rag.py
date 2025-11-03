"""
Conversation RAG Framework for Local LLM Deployment
Manages long conversation histories by retrieving relevant context instead of using entire history
"""

import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    message_id: str
    metadata: Optional[Dict] = None


class ConversationRAG:
    """
    RAG-based conversation manager that retrieves relevant history
    instead of using entire chat history
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "conversation_history",
        persist_directory: str = "./chroma_db",
        max_history_tokens: int = 2000,
        top_k_retrieval: int = 5
    ):
        """
        Initialize the Conversation RAG system
        
        Args:
            embedding_model_name: HuggingFace model for embeddings
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the vector database
            max_history_tokens: Approximate max tokens for retrieved history
            top_k_retrieval: Number of relevant messages to retrieve
        """
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Conversation history embeddings"}
        )
        
        self.max_history_tokens = max_history_tokens
        self.top_k_retrieval = top_k_retrieval
        self.message_counter = 0
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a message to the conversation history
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata dictionary
            
        Returns:
            message_id: Unique identifier for the message
        """
        message_id = f"msg_{self.message_counter}_{datetime.now().timestamp()}"
        self.message_counter += 1
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            message_id=message_id,
            metadata=metadata or {}
        )
        
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "role": role,
                "timestamp": message.timestamp,
                "message_id": message_id,
                **message.metadata
            }],
            ids=[message_id]
        )
        
        return message_id
    
    def retrieve_relevant_history(
        self,
        query: str,
        top_k: Optional[int] = None,
        time_decay: bool = True,
        include_recent: int = 2
    ) -> List[Dict]:
        """
        Retrieve relevant conversation history for the current query
        
        Args:
            query: Current user query
            top_k: Number of relevant messages to retrieve (uses default if None)
            time_decay: Apply time decay to relevance scores
            include_recent: Number of most recent messages to always include
            
        Returns:
            List of relevant messages with metadata
        """
        if top_k is None:
            top_k = self.top_k_retrieval
        
        # Check if collection is empty
        collection_count = self.collection.count()
        if collection_count == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Retrieve from ChromaDB
        # Ensure n_results is at least 1 to avoid ChromaDB error
        n_results = max(1, min(top_k + include_recent, collection_count))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Process results
        relevant_messages = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                message_data = {
                    'content': doc,
                    'role': results['metadatas'][0][i]['role'],
                    'timestamp': results['metadatas'][0][i]['timestamp'],
                    'message_id': results['metadatas'][0][i]['message_id'],
                    'distance': results['distances'][0][i]
                }
                relevant_messages.append(message_data)
        
        # Get most recent messages to always include context continuity
        recent_messages = self._get_recent_messages(include_recent)
        
        # Combine and deduplicate
        combined = self._merge_and_deduplicate(relevant_messages, recent_messages)
        
        # Sort by timestamp
        combined.sort(key=lambda x: x['timestamp'])
        
        return combined
    
    def _get_recent_messages(self, n: int) -> List[Dict]:
        """Get the n most recent messages"""
        # Get all messages and sort by timestamp
        all_data = self.collection.get()
        
        if not all_data['documents'] or len(all_data['documents']) == 0:
            return []
        
        messages = []
        for i, doc in enumerate(all_data['documents']):
            messages.append({
                'content': doc,
                'role': all_data['metadatas'][i]['role'],
                'timestamp': all_data['metadatas'][i]['timestamp'],
                'message_id': all_data['metadatas'][i]['message_id']
            })
        
        # Sort by timestamp and get last n
        if len(messages) == 0:
            return []
        
        messages.sort(key=lambda x: x['timestamp'], reverse=True)
        return messages[:n]
    
    def _merge_and_deduplicate(
        self,
        relevant: List[Dict],
        recent: List[Dict]
    ) -> List[Dict]:
        """Merge relevant and recent messages, removing duplicates"""
        seen_ids = set()
        merged = []
        
        for msg in relevant + recent:
            if msg['message_id'] not in seen_ids:
                seen_ids.add(msg['message_id'])
                merged.append(msg)
        
        return merged
    
    def format_context_for_llm(
        self,
        current_query: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format retrieved context for LLM input
        
        Args:
            current_query: Current user query
            system_prompt: Optional system prompt
            
        Returns:
            Formatted messages list for LLM
        """
        # Retrieve relevant history
        relevant_history = self.retrieve_relevant_history(current_query)
        
        # Format for LLM
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add relevant history
        for msg in relevant_history:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add current query
        messages.append({"role": "user", "content": current_query})
        
        return messages
    
    def get_conversation_summary(self) -> Dict:
        """Get summary statistics about the conversation"""
        count = self.collection.count()
        
        if count == 0:
            return {"total_messages": 0, "user_messages": 0, "assistant_messages": 0}
        
        all_data = self.collection.get()
        user_count = sum(1 for m in all_data['metadatas'] if m['role'] == 'user')
        assistant_count = sum(1 for m in all_data['metadatas'] if m['role'] == 'assistant')
        
        return {
            "total_messages": count,
            "user_messages": user_count,
            "assistant_messages": assistant_count
        }
    
    def clear_history(self):
        """Clear all conversation history"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Conversation history embeddings"}
        )
        self.message_counter = 0
    
    def export_history(self, filepath: str):
        """Export conversation history to JSON file"""
        all_data = self.collection.get()
        
        messages = []
        if all_data['documents']:
            for i, doc in enumerate(all_data['documents']):
                messages.append({
                    'content': doc,
                    'role': all_data['metadatas'][i]['role'],
                    'timestamp': all_data['metadatas'][i]['timestamp'],
                    'message_id': all_data['metadatas'][i]['message_id']
                })
        
        messages.sort(key=lambda x: x['timestamp'])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(messages)} messages to {filepath}")