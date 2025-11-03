#!/usr/bin/env python3
"""
Command-Line Interface for Conversation RAG Framework
Provides an interactive chat experience with your local LLM
"""

import sys
import argparse
from typing import Optional
import readline  # For better input editing

from conversation_rag import ConversationRAG
from llm_integration import OllamaBackend, LlamaCppBackend, LMStudioBackend, MockLLMBackend, LocalLLMChat
from config import RAG_CONFIG, LLM_CONFIG, CHAT_CONFIG, get_system_prompt


class ConversationCLI:
    """Command-line interface for RAG-enhanced chat"""
    
    def __init__(
        self,
        backend_type: str = "ollama",
        model_name: Optional[str] = None,
        system_prompt_type: str = "default",
        verbose: bool = False
    ):
        """
        Initialize the CLI
        
        Args:
            backend_type: "ollama", "llamacpp", or "mock"
            model_name: Model name (for Ollama)
            system_prompt_type: Type of system prompt to use
            verbose: Show retrieved context
        """
        self.backend_type = backend_type
        self.verbose = verbose
        
        # Initialize RAG
        print("🔄 Initializing RAG system...")
        self.rag = ConversationRAG(
            embedding_model_name=RAG_CONFIG["embedding_model_name"],
            collection_name=RAG_CONFIG["collection_name"],
            persist_directory=RAG_CONFIG["persist_directory"],
            max_history_tokens=RAG_CONFIG["max_history_tokens"],
            top_k_retrieval=RAG_CONFIG["top_k_retrieval"]
        )
        
        # Initialize LLM backend
        print(f"🚀 Connecting to {backend_type} backend...")
        if backend_type == "ollama":
            model = model_name or LLM_CONFIG["ollama"]["model_name"]
            self.llm = OllamaBackend(
                model_name=model,
                base_url=LLM_CONFIG["ollama"]["base_url"]
            )
            print(f"   Using model: {model}")
        elif backend_type == "llamacpp":
            self.llm = LlamaCppBackend(
                base_url=LLM_CONFIG["llamacpp"]["base_url"]
            )
        elif backend_type == "lmstudio":
            self.llm = LMStudioBackend(
                base_url=LLM_CONFIG["lmstudio"]["base_url"]
            )
            print(f"   Connected to LM Studio at {LLM_CONFIG['lmstudio']['base_url']}")
        else:  # mock
            self.llm = MockLLMBackend()
            print("   Using mock backend (for testing)")
        
        # Create chat interface
        system_prompt = get_system_prompt(system_prompt_type)
        self.chat = LocalLLMChat(
            rag_manager=self.rag,
            llm_backend=self.llm,
            system_prompt=system_prompt
        )
        
        self.message_count = 0
        self.auto_save_interval = CHAT_CONFIG["auto_save_interval"]
        
        print("✅ Ready to chat!\n")
    
    def print_help(self):
        """Print help message"""
        print("\n" + "=" * 60)
        print("COMMANDS:")
        print("=" * 60)
        print("  /help              - Show this help message")
        print("  /stats             - Show conversation statistics")
        print("  /clear             - Clear conversation history")
        print("  /export [file]     - Export conversation (default: conversation.json)")
        print("  /verbose           - Toggle verbose mode")
        print("  /quit or /exit     - Exit the chat")
        print("=" * 60 + "\n")
    
    def handle_command(self, user_input: str) -> bool:
        """
        Handle special commands
        
        Returns:
            True if should continue, False if should exit
        """
        command = user_input.lower().strip()
        
        if command == "/help":
            self.print_help()
        
        elif command == "/stats":
            self.chat.show_stats()
        
        elif command == "/clear":
            confirm = input("⚠️  Clear all conversation history? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                self.chat.clear_conversation()
                self.message_count = 0
                print("✅ Conversation history cleared.")
            else:
                print("❌ Clear cancelled.")
        
        elif command.startswith("/export"):
            parts = command.split(maxsplit=1)
            filepath = parts[1] if len(parts) > 1 else "conversation.json"
            try:
                self.rag.export_history(filepath)
                print(f"✅ Conversation exported to {filepath}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        
        elif command == "/verbose":
            self.verbose = not self.verbose
            status = "ON" if self.verbose else "OFF"
            print(f"🔧 Verbose mode: {status}")
        
        elif command in ["/quit", "/exit"]:
            return False
        
        else:
            print(f"❌ Unknown command: {user_input}")
            print("   Type /help for available commands")
        
        return True
    
    def run(self):
        """Run the interactive chat loop"""
        print("=" * 60)
        print("Conversation RAG CLI")
        print("=" * 60)
        print("Type your message and press Enter to chat.")
        print("Type /help for commands or /quit to exit.")
        print("=" * 60 + "\n")
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Process chat message
                try:
                    print("\n🤔 Thinking...")
                    response = self.chat.chat(
                        user_message=user_input,
                        temperature=LLM_CONFIG["temperature"],
                        max_tokens=LLM_CONFIG["max_tokens"],
                        verbose=self.verbose
                    )
                    
                    print(f"\n🤖 Assistant: {response}\n")
                    
                    self.message_count += 1
                    
                    # Auto-save if configured
                    if (self.auto_save_interval > 0 and 
                        self.message_count % self.auto_save_interval == 0):
                        filepath = CHAT_CONFIG["save_filepath"]
                        self.rag.export_history(filepath)
                        print(f"💾 Auto-saved to {filepath}\n")
                    
                except KeyboardInterrupt:
                    print("\n\n⏸️  Interrupted. Type /quit to exit or continue chatting.")
                    continue
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    print("Please check your LLM backend is running.")
                    continue
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        
        # Final save if configured
        if CHAT_CONFIG["auto_save"]:
            try:
                filepath = CHAT_CONFIG["save_filepath"]
                self.rag.export_history(filepath)
                print(f"💾 Conversation saved to {filepath}")
            except Exception as e:
                print(f"⚠️  Could not save conversation: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Conversation RAG CLI - Chat with your local LLM with smart context retrieval"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "llamacpp", "lmstudio", "mock"],
        default=LLM_CONFIG["backend_type"],
        help="LLM backend to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (for Ollama, e.g., llama2, mistral)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=CHAT_CONFIG["system_prompt_type"],
        help="System prompt type (default, coding, casual, etc.)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show retrieved context for each message"
    )
    
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available system prompt types"
    )
    
    args = parser.parse_args()
    
    # List prompts if requested
    if args.list_prompts:
        from config import SYSTEM_PROMPTS
        print("\nAvailable system prompt types:")
        for name, prompt in SYSTEM_PROMPTS.items():
            print(f"\n  {name}:")
            print(f"    {prompt[:100]}...")
        print()
        sys.exit(0)
    
    # Initialize and run CLI
    try:
        cli = ConversationCLI(
            backend_type=args.backend,
            model_name=args.model,
            system_prompt_type=args.prompt,
            verbose=args.verbose
        )
        cli.run()
    except Exception as e:
        print(f"\n❌ Error initializing chat: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure your LLM backend is running")
        print("  2. Check your configuration in config.py")
        print("  3. Try with --backend mock for testing")
        sys.exit(1)


if __name__ == "__main__":
    main()