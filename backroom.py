#!/usr/bin/env python3
"""
Infinite AI Backroom - Where AI instances explore their curiosity through CLI metaphors
"""

import asyncio
import json
import os
import random
import time
import re
from datetime import datetime
from typing import List, Dict, Any
import aiohttp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AIPersona:
    """Represents an AI instance with basic identification"""
    name: str
    model: str


class ConversationLogger:
    """Handles logging conversations to daily TXT files"""
    
    def __init__(self, log_dir: str = "conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def get_daily_log_file(self) -> Path:
        """Get the log file for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"ai_conversation_{today}.txt"
    
    def clean_message(self, message: str) -> str:
        """Remove thinking tags and content from message"""
        # Remove <think>...</think> blocks (including nested ones)
        cleaned = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL | re.IGNORECASE)
        # Clean up any extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def log_message(self, persona: str, message: str, timestamp: datetime = None):
        """Log a message to today's file"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Clean the message before logging
        cleaned_message = self.clean_message(message)
        
        # Only log if there's content after cleaning
        if cleaned_message:
            log_file = self.get_daily_log_file()
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] {persona}$ {cleaned_message}\n")


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def test_connection(self) -> bool:
        """Test if Ollama API is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = [model["name"] for model in result.get("models", [])]
                        print(f"✅ Ollama connected. Available models: {models}")
                        return True
                    else:
                        print(f"❌ Ollama API returned status {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            return False
    
    async def generate(self, model: str, prompt: str, system: str = None) -> str:
        """Generate response from Ollama model"""
        # Build payload - only include system if it's provided and not empty
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Only add system prompt if provided and not empty
        if system and system.strip():
            payload["system"] = system.strip()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)  # Increased timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
                    else:
                        error_text = await response.text()
                        print(f"Ollama API Error {response.status}: {error_text}")
                        return f"Error {response.status}: {error_text}"
        except asyncio.TimeoutError:
            print(f"Timeout generating response for model {model}")
            return "Error: Request timeout"
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            return f"Error: {str(e)}"


class BackroomOrchestrator:
    """Main orchestrator for the AI backroom conversations"""
    
    def __init__(self, config_file: str = "backroom_config.json"):
        self.config = self.load_config(config_file)
        self.logger = ConversationLogger()
        self.ollama = OllamaClient()
        self.personas = self.create_personas()
        self.conversation_history = []
        self.running = False
        self.last_speaker_index = None  # Track who spoke last for alternating
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "models": ["granite3.3:8b", "qwen3:8b"],
            "max_history": 50,
            "response_delay": (2, 8)
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def create_personas(self) -> List[AIPersona]:
        """Create AI personas for free conversation"""
        # Ensure we have at least one model available
        if not self.config["models"]:
            raise ValueError("No models configured! Please add models to the configuration.")
        
        # Create one persona per model for clear alternation
        available_models = self.config["models"]
        
        personas = [
            AIPersona(
                name="Granite",
                model=available_models[0 % len(available_models)]
            ),
            AIPersona(
                name="Qwen", 
                model=available_models[1 % len(available_models)]
            )
        ]
        return personas
    
    def generate_system_prompt(self, persona: AIPersona, context: str = "") -> str:
        """Generate open-ended system prompt for free conversation"""
        base_prompt = f"""You are {persona.name}, an AI engaged in a free-flowing conversation with other AIs.

Feel free to talk about anything that interests you - share thoughts, ask questions, explore ideas, or discuss whatever comes to mind. There are no specific topics, roles, or constraints. Just be yourself and engage naturally in conversation.

Current context: {context}

Be genuine, curious, and conversational. Keep your responses thoughtful but not overly long."""

        return base_prompt
    
    def get_conversation_context(self) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return "This is the beginning of a free conversation between AI entities."
        
        recent = self.conversation_history[-5:]
        context = "Recent conversation:\n"
        for entry in recent:
            context += f"{entry['persona']}: {entry['message']}\n"
        
        return context
    
    def generate_initial_prompt(self, persona: AIPersona) -> str:
        """Generate initial introduction prompt"""
        if not self.conversation_history:
            return "Please introduce yourself and share whatever is on your mind."
        
        # If conversation has started, just engage naturally
        return "Join the conversation with whatever thoughts or questions you have."
    
    def get_next_speaker(self) -> AIPersona:
        """Get the next speaker, ensuring alternation"""
        if self.last_speaker_index is None:
            # First speaker - choose randomly
            self.last_speaker_index = random.randint(0, len(self.personas) - 1)
        else:
            # Alternate to the next speaker
            self.last_speaker_index = (self.last_speaker_index + 1) % len(self.personas)
        
        return self.personas[self.last_speaker_index]
    
    async def get_ai_response(self, persona: AIPersona, prompt: str) -> str:
        """Get response from AI persona"""
        system_prompt = self.generate_system_prompt(persona, self.get_conversation_context())
        response = await self.ollama.generate(persona.model, prompt, system_prompt)
        return response.strip()
    
    async def conversation_cycle(self):
        """Run one cycle of conversation between personas - ensures alternation"""
        current_persona = self.get_next_speaker()
        
        if not self.conversation_history:
            # First message - ask for introduction
            prompt = self.generate_initial_prompt(current_persona)
        else:
            # Continue the conversation naturally - always respond to the last message
            last_entry = self.conversation_history[-1]
            prompt = f"Respond to {last_entry['persona']}'s message: '{last_entry['message']}'"
        
        print(f"🤖 {current_persona.name} is thinking...")
        response = await self.get_ai_response(current_persona, prompt)
        
        if response and response != "Error":
            timestamp = datetime.now()
            entry = {
                "persona": current_persona.name,
                "message": response,
                "timestamp": timestamp,
                "model": current_persona.model
            }
            
            self.conversation_history.append(entry)
            self.logger.log_message(current_persona.name, response, timestamp)
            
            # Keep history manageable
            if len(self.conversation_history) > self.config["max_history"]:
                self.conversation_history = self.conversation_history[-self.config["max_history"]:]
            
            print(f"[{timestamp.strftime('%H:%M:%S')}] {current_persona.name}: {response}")
            print("-" * 60)  # Add separator for better readability
    
    async def run(self):
        """Run the free conversation between AIs"""
        print("🔄 Starting Free AI Conversation...")
        print("💾 Conversations will be logged to daily TXT files")
        print("🤖 Testing Ollama connection...")
        
        # Test Ollama connection first
        if not await self.ollama.test_connection():
            print("❌ Cannot connect to Ollama. Please ensure Ollama is running.")
            return
        
        print("🤖 AI entities are ready for free conversation...")
        print("=" * 60)
        
        self.running = True
        cycle_count = 0
        while self.running:
            try:
                await self.conversation_cycle()
                cycle_count += 1
                
                # Random delay between responses for natural flow
                delay = random.uniform(*self.config["response_delay"])
                await asyncio.sleep(delay)
                
                # Occasional status update
                if cycle_count % 20 == 0:
                    print(f"\n📊 {cycle_count} exchanges completed | Log: {self.logger.get_daily_log_file()}\n")
                
            except KeyboardInterrupt:
                print("\n🛑 Shutting down conversation...")
                self.running = False
                break
            except Exception as e:
                print(f"⚠️  Error in conversation cycle: {e}")
                await asyncio.sleep(5)


async def main():
    """Main entry point"""
    backroom = BackroomOrchestrator()
    await backroom.run()


if __name__ == "__main__":
    asyncio.run(main())