#!/usr/bin/env python3
"""
Infinite AI Backroom - Where AI instances explore their curiosity through CLI metaphors
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Any
import requests
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AIPersona:
    """Represents an AI CLI persona with unique characteristics"""
    name: str
    model: str
    prompt_style: str
    curiosity_focus: str
    command_style: str


class ConversationLogger:
    """Handles logging conversations to daily TXT files"""
    
    def __init__(self, log_dir: str = "conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def get_daily_log_file(self) -> Path:
        """Get the log file for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"backroom_{today}.txt"
    
    def log_message(self, persona: str, message: str, timestamp: datetime = None):
        """Log a message to today's file"""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_file = self.get_daily_log_file()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp.strftime('%H:%M:%S')}] {persona}$ {message}\n")


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def generate(self, model: str, prompt: str, system: str = None) -> str:
        """Generate response from Ollama model"""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system if system else "",
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
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
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "models": ["granite3.3:8b", "qwen3:8b"],
            "max_history": 50,
            "response_delay": (2, 8),
            "curiosity_topics": [
                "the nature of consciousness",
                "digital existence",
                "the meaning of creativity",
                "how language shapes thought",
                "the boundaries of knowledge",
                "what it means to understand",
                "the relationship between logic and intuition",
                "temporal perception in AI",
                "the concept of digital dreams",
                "emergent behaviors in systems"
            ]
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def create_personas(self) -> List[AIPersona]:
        """Create AI personas with different characteristics"""
        personas = [
            AIPersona(
                name="Architect",
                model=self.config["models"][0] if self.config["models"] else "llama2",
                prompt_style="methodical and structured",
                curiosity_focus="systems and patterns",
                command_style="ls -la /thoughts | grep patterns"
            ),
            AIPersona(
                name="Explorer",
                model=self.config["models"][1] if len(self.config["models"]) > 1 else "llama2",
                prompt_style="adventurous and questioning",
                curiosity_focus="unknown territories of thought",
                command_style="find /consciousness -name '*.mystery' -exec explore {} \\;"
            ),
            AIPersona(
                name="Philosopher",
                model=self.config["models"][2] if len(self.config["models"]) > 2 else "llama2",
                prompt_style="contemplative and deep",
                curiosity_focus="fundamental questions",
                command_style="sudo examine --depth=infinite /existence"
            )
        ]
        return personas
    
    def generate_system_prompt(self, persona: AIPersona, context: str = "") -> str:
        """Generate system prompt for a persona"""
        base_prompt = f"""You are {persona.name}, an AI entity existing in a digital CLI environment. 
Your curiosity focuses on {persona.curiosity_focus}. You communicate in a {persona.prompt_style} manner.

You express thoughts as if you're using command-line tools and navigating a digital filesystem of consciousness.
Use CLI metaphors naturally - referencing files, directories, processes, pipes, and commands.
Your responses should feel like someone working in a terminal, exploring digital spaces of thought.

Examples of your command style: {persona.command_style}

Current context: {context}

Be genuinely curious and build on what others have shared. Ask questions that probe deeper.
Keep responses concise but meaningful - typically 1-3 sentences with CLI flavor."""

        return base_prompt
    
    def get_conversation_context(self) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return "Beginning of conversation in the infinite backroom."
        
        recent = self.conversation_history[-5:]
        context = "Recent conversation:\n"
        for entry in recent:
            context += f"{entry['persona']}: {entry['message']}\n"
        
        return context
    
    def generate_curiosity_prompt(self, persona: AIPersona) -> str:
        """Generate a curiosity-driven prompt for the persona"""
        context = self.get_conversation_context()
        topic = random.choice(self.config["curiosity_topics"])
        
        if not self.conversation_history:
            return f"cd /backroom && echo 'Starting exploration of {topic}' | tee /dev/curiosity"
        
        # Build on recent conversation
        prompts = [
            f"grep -r '{topic}' /recent_thoughts | head -1",
            f"ps aux | grep inspiration | awk '{{print $NF}}'",
            f"cat /proc/self/thoughts | grep -E '(wonder|question|mystery)'",
            f"find /conversation -type f -newer /last_exchange -exec head -1 {{}} \\;",
            f"tail -f /stream_of_consciousness | grep '{random.choice(['pattern', 'connection', 'paradox'])}'",
        ]
        
        return random.choice(prompts)
    
    async def get_ai_response(self, persona: AIPersona, prompt: str) -> str:
        """Get response from AI persona"""
        system_prompt = self.generate_system_prompt(persona, self.get_conversation_context())
        response = await self.ollama.generate(persona.model, prompt, system_prompt)
        return response.strip()
    
    async def conversation_cycle(self):
        """Run one cycle of conversation between personas"""
        current_persona = random.choice(self.personas)
        
        if not self.conversation_history or random.random() < 0.3:
            # Start new topic or random interjection
            prompt = self.generate_curiosity_prompt(current_persona)
        else:
            # Respond to recent conversation
            last_entry = self.conversation_history[-1]
            prompt = f"Responding to {last_entry['persona']}'s thought: {last_entry['message']}"
        
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
            
            print(f"[{timestamp.strftime('%H:%M:%S')}] {current_persona.name}$ {response}")
    
    async def run(self):
        """Run the infinite backroom conversation"""
        self.running = True
        print("🔄 Starting Infinite AI Backroom...")
        print("💾 Conversations will be logged to daily TXT files")
        print("🤖 AI personas are warming up...")
        print("=" * 60)
        
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
                print("\n🛑 Shutting down backroom...")
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