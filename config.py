"""
Configuration file for Conversation RAG Framework
Modify these settings according to your needs
"""

# RAG Configuration
RAG_CONFIG = {
    # Embedding model from HuggingFace
    # Options: 
    # - "all-MiniLM-L6-v2" (fast, 80MB, good for most cases)
    # - "all-mpnet-base-v2" (better quality, 420MB)
    # - "multi-qa-mpnet-base-dot-v1" (optimized for Q&A)
    "embedding_model_name": "google/embeddinggemma-300m",
    
    # ChromaDB collection name
    "collection_name": "conversation_history",
    
    # Directory to persist vector database
    "persist_directory": "./chroma_db",
    
    # Maximum tokens for retrieved history (approximate)
    "max_history_tokens": 2000,
    
    # Number of relevant messages to retrieve
    "top_k_retrieval": 5,
    
    # Number of most recent messages to always include
    "include_recent": 2,
}


# LLM Backend Configuration
LLM_CONFIG = {
    # Backend type: "ollama", "llamacpp", "lmstudio", or "mock"
    "backend_type": "lmstudio",
    
    # Ollama settings
    "ollama": {
        "model_name": "llama2",  # Options: llama2, mistral, codellama, etc.
        "base_url": "http://localhost:11434"
    },
    
    # llama.cpp settings
    "llamacpp": {
        "base_url": "http://localhost:8080"
    },
    
    # LM Studio settings
    "lmstudio": {
        "base_url": "http://localhost:1234"  # Default LM Studio port
    },
    
    # Generation parameters
    "temperature": 0.7,  # 0.0 to 2.0 (lower = more focused, higher = more creative)
    "max_tokens": 2000,  # Maximum tokens to generate
}


# System Prompts for different use cases
SYSTEM_PROMPTS = {
    "default": "You are a helpful AI assistant.",
    
    "coding": "You are an expert programming assistant. Provide clear, well-commented code examples and explain technical concepts thoroughly.",
    
    "technical": "You are a technical documentation expert. Provide detailed, accurate information with examples.",
    
    "casual": "You are a friendly, conversational AI assistant. Keep responses natural and engaging.",
    
    "customer_support": "You are a professional customer support assistant. Be helpful, patient, and solution-oriented.",
    
    "tutor": "You are an educational tutor. Explain concepts clearly, use examples, and check for understanding.",
}


# Chat Interface Configuration
CHAT_CONFIG = {
    # System prompt to use (key from SYSTEM_PROMPTS)
    "system_prompt_type": "default",
    
    # Show retrieved context in verbose mode
    "verbose": False,
    
    # Enable conversation statistics display
    "show_stats": True,
    
    # Auto-save conversation history
    "auto_save": True,
    "save_filepath": "conversation_history.json",
    
    # Auto-save interval (in number of messages, 0 to disable)
    "auto_save_interval": 10,
}


# Performance Configuration
PERFORMANCE_CONFIG = {
    # Batch size for embedding generation
    "embedding_batch_size": 32,
    
    # Cache embeddings (reduces repeated computation)
    "cache_embeddings": True,
    
    # Number of threads for parallel processing
    "num_threads": 4,
}


# Logging Configuration
LOGGING_CONFIG = {
    # Enable logging
    "enabled": True,
    
    # Log level: "DEBUG", "INFO", "WARNING", "ERROR"
    "level": "INFO",
    
    # Log file path (None for console only)
    "log_file": None,
}


def get_system_prompt(prompt_type: str = "default") -> str:
    """Get system prompt by type"""
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate RAG config
    if RAG_CONFIG["top_k_retrieval"] < 1:
        errors.append("top_k_retrieval must be at least 1")
    
    if RAG_CONFIG["max_history_tokens"] < 100:
        errors.append("max_history_tokens should be at least 100")
    
    # Validate LLM config
    if LLM_CONFIG["backend_type"] not in ["ollama", "llamacpp", "lmstudio", "mock"]:
        errors.append("backend_type must be 'ollama', 'llamacpp', 'lmstudio', or 'mock'")
    
    if not (0 <= LLM_CONFIG["temperature"] <= 2.0):
        errors.append("temperature must be between 0.0 and 2.0")
    
    if LLM_CONFIG["max_tokens"] < 1:
        errors.append("max_tokens must be at least 1")
    
    # Validate chat config
    if CHAT_CONFIG["system_prompt_type"] not in SYSTEM_PROMPTS:
        errors.append(f"system_prompt_type '{CHAT_CONFIG['system_prompt_type']}' not found in SYSTEM_PROMPTS")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Run validation on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Warning: {e}")