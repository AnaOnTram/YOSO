"""
Local LLM Integration for Conversation RAG
Supports Ollama, llama.cpp, and other local LLM backends
"""

from typing import List, Dict, Optional, Generator
import requests
import json
from abc import ABC, abstractmethod


class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """Generate a response from the LLM"""
        pass


class OllamaBackend(BaseLLMBackend):
    """Backend for Ollama local LLM deployment"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama backend
        
        Args:
            model_name: Name of the Ollama model (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Generate response using Ollama
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(self.chat_url, json=payload, timeout=120)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result['message']['content']
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming responses"""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    yield data['message']['content']


class LlamaCppBackend(BaseLLMBackend):
    """Backend for llama.cpp server"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize llama.cpp backend
        
        Args:
            base_url: Base URL for llama.cpp server
        """
        self.base_url = base_url
        self.completion_url = f"{base_url}/v1/chat/completions"
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Generate response using llama.cpp server
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(self.completion_url, json=payload, timeout=120)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result['choices'][0]['message']['content']
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to llama.cpp: {str(e)}"
    
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming responses"""
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]
                    if json_str.strip() != '[DONE]':
                        data = json.loads(json_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']


class LocalLLMChat:
    """
    Chat interface combining ConversationRAG with Local LLM
    """
    
    def __init__(
        self,
        rag_manager,
        llm_backend: BaseLLMBackend,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize chat interface
        
        Args:
            rag_manager: ConversationRAG instance
            llm_backend: LLM backend instance
            system_prompt: Optional system prompt
        """
        self.rag = rag_manager
        self.llm = llm_backend
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
    
    def chat(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        verbose: bool = False
    ) -> str:
        """
        Process a chat message with RAG-enhanced context
        
        Args:
            user_message: User's message
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens to generate
            verbose: Print retrieved context info
            
        Returns:
            Assistant's response
        """
        # Get formatted context with relevant history
        messages = self.rag.format_context_for_llm(
            current_query=user_message,
            system_prompt=self.system_prompt
        )
        
        if verbose:
            print(f"\n--- Retrieved Context ---")
            print(f"Total messages in context: {len(messages)}")
            for i, msg in enumerate(messages):
                print(f"{i+1}. [{msg['role']}]: {msg['content'][:100]}...")
            print("------------------------\n")
        
        # Generate response
        response = self.llm.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Store both user message and assistant response
        self.rag.add_message("user", user_message)
        self.rag.add_message("assistant", response)
        
        return response
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.rag.clear_history()
        print("Conversation history cleared.")
    
    def show_stats(self):
        """Show conversation statistics"""
        stats = self.rag.get_conversation_summary()
        print(f"\n--- Conversation Statistics ---")
        print(f"Total messages: {stats['total_messages']}")
        print(f"User messages: {stats['user_messages']}")
        print(f"Assistant messages: {stats['assistant_messages']}")
        print("------------------------------\n")


class LMStudioBackend(BaseLLMBackend):
    """Backend for LM Studio local LLM deployment"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        """
        Initialize LM Studio backend
        
        Args:
            base_url: Base URL for LM Studio API (default: http://localhost:1234)
        """
        self.base_url = base_url
        self.completion_url = f"{base_url}/v1/chat/completions"
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Generate response using LM Studio
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            response = requests.post(self.completion_url, json=payload, timeout=120)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result['choices'][0]['message']['content']
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to LM Studio: {str(e)}"
    
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming responses"""
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]
                    if json_str.strip() != '[DONE]':
                        try:
                            data = json.loads(json_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            pass


class MockLLMBackend(BaseLLMBackend):
    """Mock backend for testing without a real LLM"""
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """Generate mock response"""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg['role'] == 'user':
                last_user_msg = msg['content']
                break
        
        return f"[Mock Response] I received your message: '{last_user_msg[:50]}...' with {len(messages)} messages in context."