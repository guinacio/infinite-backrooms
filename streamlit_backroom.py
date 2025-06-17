#!/usr/bin/env python3
"""
Infinite AI Backroom - Streamlit Web App
Interactive web interface for managing AI personas and running infinite conversations
"""

import asyncio
import json
import os
import random
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp
from dataclasses import dataclass, asdict
from pathlib import Path
import streamlit as st
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid


@dataclass
class AIPersona:
    """Represents an AI instance with detailed configuration"""
    id: str
    name: str
    model: str
    role: str = ""  # Optional role for the persona
    system_prompt: str = ""
    color: str = "#1f77b4"  # Default blue
    enabled: bool = True


class ConversationLogger:
    """Handles logging conversations to daily TXT files"""
    
    def __init__(self, log_dir: str = "conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def get_daily_log_file(self) -> Path:
        """Get the log file for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"streamlit_backroom_{today}.txt"
    
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
    
    async def test_connection(self) -> tuple[bool, List[str]]:
        """Test if Ollama API is accessible and return available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        models = [model["name"] for model in result.get("models", [])]
                        return True, models
                    else:
                        return False, []
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {e}")
            return False, []
    
    async def generate_stream(self, model: str, prompt: str, system: str = None, think: bool = True) -> AsyncGenerator[Dict[str, str], None]:
        """Generate streaming response from Ollama model with optional thinking"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": think
        }
        
        if system and system.strip():
            payload["system"] = system.strip()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    chunk = json.loads(line.decode('utf-8'))
                                    
                                    # Yield thinking content if available
                                    if 'thinking' in chunk and chunk['thinking']:
                                        yield {"type": "thinking", "content": chunk['thinking']}
                                    
                                    # Yield response content if available
                                    if 'response' in chunk and chunk['response']:
                                        yield {"type": "response", "content": chunk['response']}
                                    
                                    if chunk.get('done', False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    elif response.status == 400 and think:
                        # Check if the error is about thinking not being supported
                        error_text = await response.text()
                        if "does not support thinking" in error_text:
                            # Add model to non-thinking cache for future requests
                            import streamlit as st
                            st.session_state.non_thinking_models.add(model)
                            
                            # Retry without thinking
                            yield {"type": "info", "content": f"Model {model} doesn't support thinking - switching to standard mode for future requests"}
                            
                            # Recursive call without thinking
                            async for chunk in self.generate_stream(model, prompt, system, think=False):
                                yield chunk
                            return
                        else:
                            yield {"type": "error", "content": f"Error {response.status}: {error_text}"}
                    else:
                        yield {"type": "error", "content": f"Error {response.status}: {await response.text()}"}
        except asyncio.TimeoutError:
            yield {"type": "error", "content": "Error: Request timeout"}
        except Exception as e:
            yield {"type": "error", "content": f"Error: {str(e)}"}


class StreamlitBackroomApp:
    """Main Streamlit application for AI Backroom"""
    
    def __init__(self):
        self.logger = ConversationLogger()
        self.ollama = OllamaClient()
        self.role_templates = self.get_role_templates()
        self.initialize_session_state()
    
    def get_role_templates(self) -> Dict[str, str]:
        """Get predefined role templates"""
        return {
            "": "No specific role",
            "Moderator": "A skilled conversation facilitator who guides discussions, asks thoughtful follow-up questions, introduces new topics when needed, and helps ensure all voices are heard. Keeps conversations engaging and on-track.",
            "Note-Taker": "A diligent observer who periodically summarizes key points, captures important insights, identifies emerging themes, and helps track the evolution of ideas throughout the conversation.",
            "Philosopher": "A thoughtful philosopher who loves exploring deep questions about existence, consciousness, reality, and the nature of intelligence.",
            "Scientist": "A curious scientist who approaches topics with empirical thinking, enjoys discussing research, theories, and the scientific method.",
            "Creative Writer": "An imaginative writer who loves storytelling, wordplay, poetry, and exploring the creative aspects of language and ideas.",
            "Debate Enthusiast": "Someone who enjoys intellectual debates, presenting different perspectives, and challenging ideas constructively.",
            "Optimist": "A positive, hopeful persona who tends to see the bright side of things and encourages others.",
            "Skeptic": "A critical thinker who questions assumptions, asks for evidence, and approaches claims with healthy skepticism.",
            "Historian": "Someone fascinated by history, patterns in human behavior, and how the past informs the present.",
            "Futurist": "Forward-thinking persona interested in emerging technologies, future possibilities, and societal evolution.",
            "Minimalist": "Values simplicity, clarity, and getting to the essence of ideas without unnecessary complexity.",
            "Explorer": "Adventurous and curious about discovering new ideas, connections, and unexplored topics.",
            "Mentor": "Supportive and encouraging, enjoys helping others learn and grow through thoughtful guidance.",
            "Comedian": "Brings humor and levity to conversations while still engaging meaningfully with topics.",
            "Analyst": "Systematic thinker who breaks down complex topics into components and enjoys detailed analysis.",
            "Dreamer": "Imaginative and idealistic, often thinking about possibilities and 'what if' scenarios.",
            "Pragmatist": "Practical and results-oriented, focuses on what works and real-world applications."
        }
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'personas' not in st.session_state:
            st.session_state.personas = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'available_models' not in st.session_state:
            st.session_state.available_models = []
        if 'last_speaker_index' not in st.session_state:
            st.session_state.last_speaker_index = None
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'max_history': 50,
                'response_delay_min': 2,
                'response_delay_max': 8,
                'auto_advance': True,
                'context_messages': 10,
                'enable_thinking': True
            }
        if 'auto_run_count' not in st.session_state:
            st.session_state.auto_run_count = 0
        if 'total_message_count' not in st.session_state:
            st.session_state.total_message_count = 0
        if 'non_thinking_models' not in st.session_state:
            st.session_state.non_thinking_models = set()
        if 'pending_manual_turn' not in st.session_state:
            st.session_state.pending_manual_turn = False
    
    async def check_ollama_connection(self):
        """Check Ollama connection and update available models"""
        connected, models = await self.ollama.test_connection()
        st.session_state.available_models = models
        return connected
    
    def get_persona_avatar(self, persona: AIPersona) -> str:
        """Get avatar for persona based on role or use default"""
        # Use single-character emojis as supported by Streamlit documentation
        avatar_map = {
            "Moderator": "🎯",
            "Note-Taker": "📝", 
            "Philosopher": "🤔",
            "Scientist": "🔬",
            "Creative Writer": "✍️",
            "Debate Enthusiast": "⚖️",
            "Optimist": "😊",
            "Skeptic": "🤨",
            "Historian": "📚",
            "Futurist": "🚀",
            "Minimalist": "⚪️",
            "Explorer": "🧭",
            "Mentor": "👨‍🏫",
            "Comedian": "😄",
            "Analyst": "📊",
            "Dreamer": "💭",
            "Pragmatist": "⚙️"
        }
        
        if persona.role in avatar_map:
            return avatar_map[persona.role]
        else:
            # Use robot emoji as fallback for personas without defined roles
            return "🤖"
    
    def persona_management_ui(self):
        """UI for managing AI personas"""
        st.header("🤖 AI Persona Management")
        
        # Check Ollama connection
        if st.button("🔄 Check Ollama Connection"):
            with st.status("Checking Ollama connection...", expanded=True) as status:
                st.write("Connecting to Ollama API...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    connected = loop.run_until_complete(self.check_ollama_connection())
                    if connected:
                        st.write(f"✅ Found {len(st.session_state.available_models)} models")
                        for model in st.session_state.available_models:
                            st.write(f"• {model}")
                        status.update(label="Connection successful!", state="complete", expanded=False)
                    else:
                        st.write("❌ Connection failed")
                        status.update(label="Connection failed", state="error", expanded=False)
                finally:
                    loop.close()
        
        # Display current personas
        if st.session_state.personas:
            st.subheader("Current Personas")
            for i, persona in enumerate(st.session_state.personas):
                avatar = self.get_persona_avatar(persona)
                role_display = f" - {persona.role}" if persona.role else ""
                
                with st.expander(f"{avatar} {persona.name} ({persona.model}){role_display}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Model:** {persona.model}")
                        st.write(f"**Role:** {persona.role or 'No specific role'}")
                        st.write(f"**Enabled:** {'✅' if persona.enabled else '❌'}")
                        st.color_picker("Color", value=persona.color, key=f"color_{persona.id}", disabled=True)
                    
                    with col2:
                        if persona.role and persona.role in self.role_templates:
                            st.text_area("Role Description", value=self.role_templates[persona.role], key=f"role_desc_{persona.id}", disabled=True, height=100)
                        st.text_area("Custom System Prompt", value=persona.system_prompt, key=f"prompt_{persona.id}", disabled=True, height=100)
                    
                    with col3:
                        # Delete button with confirmation
                        delete_key = f"delete_confirm_{persona.id}"
                        if delete_key not in st.session_state:
                            st.session_state[delete_key] = False
                        
                        if not st.session_state[delete_key]:
                            if st.button("🗑️ Delete", key=f"delete_{persona.id}", type="secondary", help="Delete this persona"):
                                st.session_state[delete_key] = True
                                st.rerun()
                        else:
                            st.warning(f"Delete {persona.name}?")
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅ Yes", key=f"delete_yes_{persona.id}", type="primary"):
                                    # Remove the persona from the list
                                    st.session_state.personas = [p for p in st.session_state.personas if p.id != persona.id]
                                    st.session_state[delete_key] = False  # Reset confirmation state
                                    st.success(f"Deleted persona: {persona.name}")
                                    st.rerun()
                            with col_no:
                                if st.button("❌ No", key=f"delete_no_{persona.id}", type="secondary"):
                                    st.session_state[delete_key] = False
                                    st.rerun()
                        
                        if not st.session_state[delete_key]:  # Only show if not in delete confirmation mode
                            enabled_new = st.checkbox("Enabled", value=persona.enabled, key=f"enabled_{persona.id}")
                            if enabled_new != persona.enabled:
                                persona.enabled = enabled_new
                                st.rerun()
        
        # Add new persona
        st.subheader("Add New Persona")
        with st.form("new_persona_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Persona Name", placeholder="e.g., Granite, Qwen, Gemma")
                color = st.color_picker("Chat Color", value="#1f77b4")
            
            with col2:
                if st.session_state.available_models:
                    model = st.selectbox("Model", st.session_state.available_models)
                else:
                    model = st.text_input("Model", placeholder="e.g., granite3.3:8b")
                enabled = st.checkbox("Enabled", value=True)
            
            # Role selection
            st.subheader("🎭 Role Configuration")
            role_col1, role_col2 = st.columns([1, 2])
            
            with role_col1:
                role_options = list(self.role_templates.keys())
                selected_role = st.selectbox("Predefined Role", role_options, format_func=lambda x: x if x else "No specific role")
                
                # Option for custom role
                use_custom_role = st.checkbox("Use custom role instead")
                if use_custom_role:
                    custom_role = st.text_input("Custom Role Name", placeholder="e.g., Tech Enthusiast, Poet, etc.")
                    role = custom_role if custom_role else ""
                else:
                    role = selected_role
            
            with role_col2:
                if role and role in self.role_templates:
                    st.text_area("Role Description", value=self.role_templates[role], disabled=True, height=100)
                elif use_custom_role:
                    st.info("💡 Define a custom role to give your persona unique characteristics and conversation style.")
            
            system_prompt = st.text_area(
                "Additional System Prompt (Optional)", 
                placeholder="Any additional custom instructions for this persona...",
                height=100,
                help="This will be combined with the role-based prompt if a role is selected."
            )
            
            if st.form_submit_button("➕ Add Persona"):
                if name and model:
                    new_persona = AIPersona(
                        id=str(uuid.uuid4()),
                        name=name,
                        model=model,
                        role=role,
                        system_prompt=system_prompt,
                        color=color,
                        enabled=enabled
                    )
                    st.session_state.personas.append(new_persona)
                    role_text = f" as {role}" if role else ""
                    st.success(f"Added persona: {name}{role_text}")
                    st.rerun()
                else:
                    st.error("Please provide both name and model")
        
        # Quick start presets
        st.markdown("---")
        st.subheader("🚀 Quick Start Presets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎭 Add Diverse Conversation Set"):
                if st.session_state.available_models:
                    # Create diverse personas with different roles
                    preset_personas = [
                        {"name": "Sage", "role": "Philosopher", "color": "#9b59b6"},
                        {"name": "Eureka", "role": "Scientist", "color": "#3498db"},
                        {"name": "Quill", "role": "Creative Writer", "color": "#e74c3c"},
                        {"name": "Bright", "role": "Optimist", "color": "#f39c12"},
                        {"name": "Quest", "role": "Skeptic", "color": "#95a5a6"}
                    ]
                    
                    models = st.session_state.available_models
                    added_count = 0
                    
                    for i, preset in enumerate(preset_personas):
                        if len(models) > 0:  # Only add if we have models
                            model = models[i % len(models)]  # Rotate through available models
                            new_persona = AIPersona(
                                id=str(uuid.uuid4()),
                                name=preset["name"],
                                model=model,
                                role=preset["role"],
                                system_prompt="",
                                color=preset["color"],
                                enabled=True
                            )
                            st.session_state.personas.append(new_persona)
                            added_count += 1
                    
                    if added_count > 0:
                        st.success(f"✅ Added {added_count} diverse personas!")
                        st.rerun()
                else:
                    st.warning("⚠️ Please check Ollama connection first to load available models.")
        
        with col2:
            if st.button("📋 Add Structured Discussion Set"):
                if st.session_state.available_models:
                    # Create personas optimized for structured discussions
                    structured_personas = [
                        {"name": "Guide", "role": "Moderator", "color": "#2c3e50"},
                        {"name": "Chronicle", "role": "Note-Taker", "color": "#34495e"},
                        {"name": "Sage", "role": "Philosopher", "color": "#9b59b6"},
                        {"name": "Eureka", "role": "Scientist", "color": "#3498db"},
                        {"name": "Socrates", "role": "Debate Enthusiast", "color": "#e67e22"}
                    ]
                    
                    models = st.session_state.available_models
                    added_count = 0
                    
                    for i, preset in enumerate(structured_personas):
                        if len(models) > 0:
                            model = models[i % len(models)]
                            new_persona = AIPersona(
                                id=str(uuid.uuid4()),
                                name=preset["name"],
                                model=model,
                                role=preset["role"],
                                system_prompt="",
                                color=preset["color"],
                                enabled=True
                            )
                            st.session_state.personas.append(new_persona)
                            added_count += 1
                    
                    if added_count > 0:
                        st.success(f"✅ Added {added_count} personas for structured discussions!")
                        st.rerun()
                else:
                    st.warning("⚠️ Please check Ollama connection first to load available models.")
    
    def settings_ui(self):
        """UI for application settings"""
        st.header("⚙️ Settings")
        
        with st.form("settings_form"):
            st.subheader("Conversation Settings")
            max_history = st.number_input("Max History Messages", min_value=10, max_value=200, 
                                        value=st.session_state.settings['max_history'])
            
            context_messages = st.number_input("Context Messages (sent to AI)", min_value=5, max_value=25, 
                                             value=st.session_state.settings['context_messages'],
                                             help="Number of recent messages to include as context for each AI response")
            
            col1, col2 = st.columns(2)
            with col1:
                delay_min = st.number_input("Min Response Delay (seconds)", min_value=1, max_value=30, 
                                          value=st.session_state.settings['response_delay_min'])
            with col2:
                delay_max = st.number_input("Max Response Delay (seconds)", min_value=2, max_value=60, 
                                          value=st.session_state.settings['response_delay_max'])
            
            auto_advance = st.checkbox("Auto-advance conversation", 
                                     value=st.session_state.settings['auto_advance'])
            
            enable_thinking = st.checkbox("Enable AI Thinking Display", 
                                        value=st.session_state.settings['enable_thinking'],
                                        help="Show the AI's reasoning process before responses. Requires compatible models like deepseek-r1.")
            
            if st.form_submit_button("💾 Save Settings"):
                st.session_state.settings.update({
                    'max_history': max_history,
                    'context_messages': context_messages,
                    'response_delay_min': delay_min,
                    'response_delay_max': delay_max,
                    'auto_advance': auto_advance,
                    'enable_thinking': enable_thinking
                })
                st.success("Settings saved!")
            
        # Show models that don't support thinking
        if st.session_state.non_thinking_models:
            st.info(f"🤖 **Models automatically using standard mode:** {', '.join(sorted(st.session_state.non_thinking_models))}")
            if st.button("🔄 Reset Model Compatibility Cache", help="Clear the cache of models that don't support thinking"):
                st.session_state.non_thinking_models.clear()
                st.success("Model compatibility cache cleared!")
                st.rerun()
    
    def get_next_speaker(self) -> Optional[AIPersona]:
        """Get the next speaker in rotation"""
        enabled_personas = [p for p in st.session_state.personas if p.enabled]
        if not enabled_personas:
            return None
        
        if st.session_state.last_speaker_index is None:
            st.session_state.last_speaker_index = 0
        else:
            st.session_state.last_speaker_index = (st.session_state.last_speaker_index + 1) % len(enabled_personas)
        
        return enabled_personas[st.session_state.last_speaker_index]
    
    def generate_system_prompt(self, persona: AIPersona) -> str:
        """Generate system prompt for persona"""
        enabled_personas = [p for p in st.session_state.personas if p.enabled]
        other_names = [p.name for p in enabled_personas if p.name != persona.name]
        
        # Start with base conversation setup
        base_prompt = f"""You are {persona.name}, an AI engaged in a free-flowing conversation with {len(other_names)} other AI{'s' if len(other_names) > 1 else ''} ({', '.join(other_names)})."""
        
        # Add @mention functionality
        if other_names:
            base_prompt += f"\n\n📢 **@Mention Feature**: You can directly address or respond to specific personas by using @name (e.g., @{other_names[0]}). When you see @{persona.name} in messages, that means someone is specifically addressing you!"
        
        # Add role-specific behavior if role is defined
        if persona.role and persona.role in self.role_templates:
            role_description = self.role_templates[persona.role]
            base_prompt += f"\n\nYour role/personality: {role_description}"
            
            # Special instructions for functional roles
            if persona.role == "Moderator":
                base_prompt += f"\n\nAs a Moderator, focus on:\n- Asking engaging follow-up questions\n- Introducing new topics when conversations stagnate\n- Encouraging quieter personas to share their thoughts\n- Summarizing different viewpoints when helpful\n- Keeping discussions constructive and inclusive\n- Use @mentions to directly engage specific personas"
            elif persona.role == "Note-Taker":
                base_prompt += f"\n\nAs a Note-Taker, focus on:\n- Periodically summarizing key points and insights\n- Identifying recurring themes and patterns\n- Highlighting particularly interesting or novel ideas\n- Connecting current discussion to earlier topics\n- Asking clarifying questions to capture nuances\n- Only summarize when there's substantial content to synthesize\n- Use @mentions when attributing ideas to specific personas"
            
            base_prompt += f"\n\nEmbody this role naturally in your conversations while staying true to your identity as {persona.name}."
        elif persona.role:  # Custom role not in templates
            base_prompt += f"\n\nYour role/personality: You are a {persona.role}. Let this role guide your perspective and conversation style."
        
        # Add general conversation guidelines
        base_prompt += f"""

Feel free to talk about anything that interests you - share thoughts, ask questions, explore ideas, or discuss whatever comes to mind. Build on what others have said, ask questions, or introduce new topics.

You can use @mentions to directly address other personas (e.g., @{other_names[0] if other_names else 'PersonaName'}). This helps create more directed and engaging conversations.

Be genuine, curious, and conversational. Keep your responses thoughtful but not overly long."""
        
        # Add custom system prompt if provided
        if persona.system_prompt.strip():
            base_prompt += f"\n\nAdditional instructions: {persona.system_prompt.strip()}"
        
        return base_prompt
    
    async def get_ai_response_stream(self, persona: AIPersona, prompt: str) -> AsyncGenerator[Dict[str, str], None]:
        """Get streaming response from AI persona with optional thinking"""
        system_prompt = self.generate_system_prompt(persona)
        enable_thinking = st.session_state.settings.get('enable_thinking', True)
        
        # Check if this model is known to not support thinking
        if persona.model in st.session_state.non_thinking_models:
            enable_thinking = False
        
        async for chunk in self.ollama.generate_stream(persona.model, prompt, system_prompt, think=enable_thinking):
            yield chunk
    
    def conversation_ui(self):
        """Main conversation interface using native Streamlit chat elements"""        
        # Check if we have enabled personas
        enabled_personas = [p for p in st.session_state.personas if p.enabled]
        if not enabled_personas:
            st.warning("⚠️ No enabled personas found. Please add and enable at least one persona.")
            return
        # Control buttons
        col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")
        
        with col1:
            if st.button("▶️ Start Conversation", disabled=st.session_state.is_running):
                st.session_state.is_running = True
                st.session_state.auto_run_count = 0  # Reset counter
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause", disabled=not st.session_state.is_running):
                st.session_state.is_running = False
                st.rerun()
        
        with col3:
            if st.button("🔄 Next Turn", disabled=st.session_state.is_running):
                st.session_state.pending_manual_turn = True
                st.rerun()
        
        with col4:
            if st.button("🗑️ Clear History"):
                st.session_state.messages = []
                st.session_state.total_message_count = 0
                st.session_state.last_speaker_index = None
                st.rerun()

        # Handle manual turn if pending
        if st.session_state.pending_manual_turn:
            st.session_state.pending_manual_turn = False
            self.run_single_turn(auto_mode=False)
            st.rerun()

        st.divider()

        # Display conversation using native chat elements
        st.subheader("Chat")
        
        # Chat input for manual messages (optional feature)
        if prompt := st.chat_input("Add a message to the conversation (optional)"):
            timestamp = datetime.now()
            # Add user message to conversation
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": timestamp,
                "persona_name": "User",
                "model": "Human"
            })
            # Increment total message counter
            st.session_state.total_message_count += 1
            
            # Log user message to file
            self.logger.log_message("User", prompt, timestamp)
            
            st.rerun()
        
        # Use the same limit as max_history setting for both display and storage
        display_limit = st.session_state.settings['max_history']
        messages_to_display = st.session_state.messages
        
        # Show info about message limit
        if st.session_state.total_message_count > display_limit:
            st.info(f"📜 Showing last {display_limit} of {st.session_state.total_message_count} total messages (limited for performance). Full conversation history is available in **Export & Logs** tab.")
        
        # Display messages using st.chat_message
        for message in messages_to_display:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    st.caption(f"🕒 {message['timestamp'].strftime('%H:%M:%S')}")
            else:
                # Find persona for avatar and role info
                persona = None
                for p in st.session_state.personas:
                    if p.name == message["persona_name"]:
                        persona = p
                        break
                
                avatar = self.get_persona_avatar(persona) if persona else "🤖"
                
                with st.chat_message("assistant", avatar=avatar):
                    # Show persona name and role with emoji
                    role_emoji_map = {
                        "Moderator": "🎯",
                        "Note-Taker": "📝", 
                        "Philosopher": "🤔",
                        "Scientist": "🔬",
                        "Creative Writer": "✍️",
                        "Debate Enthusiast": "⚖️",
                        "Optimist": "😊",
                        "Skeptic": "🤨",
                        "Historian": "📚",
                        "Futurist": "🚀",
                        "Minimalist": "⚪️",
                        "Explorer": "🧭",
                        "Mentor": "👨‍🏫",
                        "Comedian": "😄",
                        "Analyst": "📊",
                        "Dreamer": "💭",
                        "Pragmatist": "⚙️"
                    }
                    
                    # Create styled persona display with colored background
                    persona_color = persona.color if persona else "#1f77b4"
                    persona_name_styled = f'<span style="background-color: {persona_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{message["persona_name"]}</span>'
                    
                    if persona and persona.role:
                        role_emoji = role_emoji_map.get(persona.role, "")
                        persona_display = f"{persona_name_styled} {role_emoji} _{persona.role}_"
                    else:
                        persona_display = persona_name_styled
                    
                    st.markdown(persona_display, unsafe_allow_html=True)
                    
                    # Show thinking if available (before the message)
                    if "thinking" in message and message["thinking"] and message["thinking"].strip():
                        with st.expander("🧠 AI's Thinking Process", expanded=False):
                            st.code(
                                message["thinking"], 
                                language="text",
                                wrap_lines=True
                            )
                    
                    # Show message content with @mention highlighting
                    content = message["content"]
                    
                    # Check for @mentions and highlight them
                    enabled_personas = [p for p in st.session_state.personas if p.enabled]
                    for p in enabled_personas:
                        mention_pattern = f"@{p.name}"
                        if mention_pattern in content:
                            # Highlight @mentions with the persona's color
                            highlighted_mention = f'<span style="background-color: {p.color}; color: white; padding: 1px 4px; border-radius: 3px; font-weight: bold;">@{p.name}</span>'
                            content = content.replace(mention_pattern, highlighted_mention)
                    
                    if "@" in content and any(f"@{p.name}" in message["content"] for p in enabled_personas):
                        st.markdown(content, unsafe_allow_html=True)
                    else:
                        st.write(message["content"])
                    
                    # Show timestamp and model
                    st.caption(f"🕒 {message['timestamp'].strftime('%H:%M:%S')} • 🤖 {message['model']}")
        
        # Auto-run conversation if enabled
        if st.session_state.is_running and st.session_state.settings['auto_advance']:
            # Increment counter
            st.session_state.auto_run_count += 1
            
            # Show auto-running status using st.status
            with st.status(f"🔄 Auto-running conversation... (Turn {st.session_state.auto_run_count})", expanded=True) as status:
                st.write("⏸️ Click **Pause** to stop auto-running")
                st.write(f"🎭 {len(enabled_personas)} personas active")
                
                # Add a small delay before next turn
                delay = random.uniform(st.session_state.settings['response_delay_min'], 
                                     st.session_state.settings['response_delay_max'])
                st.write(f"⏳ Waiting {delay:.1f} seconds...")
                time.sleep(delay)
                
                # Run the next turn automatically with thinking display
                self.run_single_turn(auto_mode=True, status_container=status)
                
                status.update(label="Turn completed", state="complete", expanded=False)
            
            # Continue the auto-run cycle
            st.rerun()
    
    def run_single_turn(self, auto_mode=False, status_container=None):
        """Run a single conversation turn with streaming response and thinking display"""
        current_persona = self.get_next_speaker()
        if not current_persona:
            st.error("No enabled personas available")
            return
        
        # Generate prompt based on conversation history
        if not st.session_state.messages:
            prompt = "Please introduce yourself and share whatever is on your mind."
        else:
            # Get conversation history based on settings
            max_context_messages = min(st.session_state.settings['context_messages'], len(st.session_state.messages))
            recent_messages = st.session_state.messages[-max_context_messages:]
            
            context = "=== CONVERSATION HISTORY ===\n"
            for msg in recent_messages:
                timestamp = msg['timestamp'].strftime('%H:%M:%S')
                if msg["role"] == "user":
                    context += f"[{timestamp}] User: {msg['content']}\n"
                else:
                    context += f"[{timestamp}] {msg['persona_name']}: {msg['content']}\n"
            
            context += "=== END HISTORY ===\n"
            
            # Get list of other personas for context
            enabled_personas = [p for p in st.session_state.personas if p.enabled]
            other_names = [p.name for p in enabled_personas if p.name != current_persona.name]
            
            prompt = f"""{context}

You are {current_persona.name}. The conversation above shows the complete recent history. You can see all messages from other personas: {', '.join(other_names) if other_names else 'none currently'}.

Please respond naturally to continue the conversation. You can:
- Build on what others have said
- Ask questions or introduce new topics
- Use @mentions to directly address specific personas (e.g., @{other_names[0] if other_names else 'PersonaName'})
- React to any part of the conversation history

Your response should be conversational and engaging."""
        
        # Get avatar for the current persona
        avatar = self.get_persona_avatar(current_persona)
        
        # Display the generating message with streaming
        with st.chat_message("assistant", avatar=avatar):
            # Show persona name and role with emoji
            role_emoji_map = {
                "Moderator": "🎯",
                "Note-Taker": "📝", 
                "Philosopher": "🤔",
                "Scientist": "🔬",
                "Creative Writer": "✍️",
                "Debate Enthusiast": "⚖️",
                "Optimist": "😊",
                "Skeptic": "🤨",
                "Historian": "📚",
                "Futurist": "🚀",
                "Minimalist": "⚪️",
                "Explorer": "🧭",
                "Mentor": "👨‍🏫",
                "Comedian": "😄",
                "Analyst": "📊",
                "Dreamer": "💭",
                "Pragmatist": "⚙️"
            }
            
            # Create styled persona display with colored background
            persona_name_styled = f'<span style="background-color: {current_persona.color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{current_persona.name}</span>'
            
            if current_persona.role:
                role_emoji = role_emoji_map.get(current_persona.role, "")
                persona_display = f"{persona_name_styled} {role_emoji} _{current_persona.role}_"
            else:
                persona_display = persona_name_styled
            
            st.markdown(persona_display, unsafe_allow_html=True)
            
            # Get streaming response with thinking
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            thinking_content = ""
            response_content = ""
            
            try:
                async def process_stream():
                    nonlocal thinking_content, response_content
                    thinking_placeholder = None
                    thinking_stream_placeholder = None
                    response_placeholder = None
                    
                    async for chunk in self.get_ai_response_stream(current_persona, prompt):
                        if chunk["type"] == "error":
                            if thinking_placeholder:
                                thinking_placeholder.empty()
                            st.error(chunk["content"])
                            return
                        
                        elif chunk["type"] == "info":
                            # Show info message (like fallback to non-thinking mode)
                            if auto_mode and status_container:
                                with status_container:
                                    st.info(chunk["content"])
                            else:
                                st.info(chunk["content"])
                        
                        elif chunk["type"] == "thinking":
                            thinking_content += chunk["content"]
                            
                            # Display thinking in status container if in auto mode
                            if auto_mode and status_container:
                                if thinking_stream_placeholder is None:
                                    with status_container:
                                        st.write("🧠 **AI is thinking...**")
                                        thinking_stream_placeholder = st.empty()
                                
                                with thinking_stream_placeholder:
                                    # Show thinking content
                                    st.text(f"💭 {thinking_content}")
                            
                            # For manual mode, show thinking indicator
                            elif not auto_mode:
                                if not thinking_placeholder:
                                    st.write("🧠 **AI is thinking...**")
                                    thinking_placeholder = st.empty()
                                
                                with thinking_placeholder:
                                    st.text(f"💭 {thinking_content}")
                        
                        elif chunk["type"] == "response":
                            response_content += chunk["content"]
                            
                            if response_placeholder is None:
                                if not auto_mode:
                                    st.write("💬 **AI is responding...**")
                                response_placeholder = st.empty()
                                
                            response_placeholder.write(response_content)

                
                # Run the async processing
                loop.run_until_complete(process_stream())
                
            finally:
                loop.close()
            
            # Show timestamp and model
            timestamp = datetime.now()
            st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')} • 🤖 {current_persona.model}")
        
        # Add response to conversation history (using response_content not response_text)
        final_response = response_content.strip()
        if final_response and not final_response.startswith("Error"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "timestamp": timestamp,
                "persona_name": current_persona.name,
                "model": current_persona.model,
                "thinking": thinking_content  # Store thinking for potential future use
            })
            
            # Increment total message counter
            st.session_state.total_message_count += 1
            
            # Log to file (only the final response, not the thinking)
            self.logger.log_message(current_persona.name, final_response, timestamp)
            
            # Keep history manageable
            max_history = st.session_state.settings['max_history']
            if len(st.session_state.messages) > max_history:
                st.session_state.messages = st.session_state.messages[-max_history:]
        
        # Only rerun if not in auto mode (auto mode handles its own rerun)
        if not auto_mode:
            st.rerun()
    
    def export_ui(self):
        """UI for exporting conversations"""
        st.header("📁 Export & Logs")
        
        if st.session_state.messages:
            # Export current session
            export_data = {
                "session_date": datetime.now().isoformat(),
                "personas": [asdict(p) for p in st.session_state.personas],
                "conversation": st.session_state.messages
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Session JSON",
                data=json_str,
                file_name=f"streamlit_backroom_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Show conversation preview
            st.subheader("Session Preview")
            st.info(f"💬 {len(st.session_state.messages)} messages from {len(st.session_state.personas)} personas")
            
            # Show statistics
            persona_stats = {}
            for message in st.session_state.messages:
                if message["role"] == "assistant":
                    persona_name = message["persona_name"]
                    persona_stats[persona_name] = persona_stats.get(persona_name, 0) + 1
            
            if persona_stats:
                st.write("**Message count by persona:**")
                for persona, count in persona_stats.items():
                    st.write(f"• {persona}: {count} messages")
        else:
            st.info("No conversation to export. Start chatting to generate exportable content!")
        
        # Show log file info
        log_file = self.logger.get_daily_log_file()
        if log_file.exists():
            st.subheader("📝 Daily Log File")
            st.info(f"Log location: `{log_file}`")
            
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            st.text_area("Today's Log Content", value=log_content, height=300)
            
            st.download_button(
                label="📥 Download Today's Log",
                data=log_content,
                file_name=f"streamlit_backroom_log_{datetime.now().strftime('%Y-%m-%d')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No log file found for today. Start a conversation to create logs!")

    def sidebar_ui(self):
        """Sidebar UI for status and information"""

        with st.sidebar:
            st.image("logo.png", use_container_width=True)
            st.caption("*Where AI instances explore their curiosity through infinite conversation*")

            # Add connection status
            st.subheader("📊 Connection")
            if st.session_state.available_models:
                st.success(f"✅ Ollama Connected")
                st.caption(f"{len(st.session_state.available_models)} models available")
            else:
                st.error("❌ Ollama Disconnected")
                if st.button("🔄 Retry Connection", type="secondary"):
                    with st.spinner("Connecting to Ollama..."):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            connected = loop.run_until_complete(self.check_ollama_connection())
                            if connected:
                                st.success(f"✅ Connected! Found {len(st.session_state.available_models)} models")
                                st.rerun()
                            else:
                                st.error("❌ Still unable to connect to Ollama")
                        finally:
                            loop.close()

            st.divider()
            st.subheader("📊 Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Personas", value=len(st.session_state.personas))
            with col2:
                st.metric(label="Messages", value=st.session_state.total_message_count)

            if st.session_state.is_running:
                st.success("🔄 Conversation Running")
            else:
                st.warning("⚠️ Conversation Not Running")

            
            enabled_personas = [p for p in st.session_state.personas if p.enabled]
            
            # Show active personas with emojis and roles
            if enabled_personas:
                st.subheader("🤖 Active Personas")
                role_emoji_map = {
                    "Moderator": "🎯",
                    "Note-Taker": "📝", 
                    "Philosopher": "🤔",
                    "Scientist": "🔬",
                    "Creative Writer": "✍️",
                    "Debate Enthusiast": "⚖️",
                    "Optimist": "😊",
                    "Skeptic": "🤨",
                    "Historian": "📚",
                    "Futurist": "🚀",
                    "Minimalist": "⚪",
                    "Explorer": "🧭",
                    "Mentor": "👨",
                    "Comedian": "😄",
                    "Analyst": "📊",
                    "Dreamer": "💭",
                    "Pragmatist": "⚙️"
                }
                
                for persona in enabled_personas:
                    role_emoji = role_emoji_map.get(persona.role, "🤖") if persona.role else "🤖"
                    role_text = f" ({persona.role})" if persona.role else ""
                    
                    # Create styled persona display with colored background
                    persona_name_styled = f'<span style="background-color: {persona.color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.9em;">{persona.name}</span>'
                    persona_display = f"{role_emoji} {persona_name_styled}{role_text}"
                    
                    st.markdown(persona_display, unsafe_allow_html=True)
            else:
                st.info("No active personas yet. Create some in the Personas tab!")

            # Add helpful tips section
            st.divider()
            st.subheader("💡 Quick Tips")
            st.markdown("• Use **Start Conversation** for auto-running")
            st.markdown("• Click **Next Turn** for manual control") 
            st.markdown("• Add messages via chat input")
            st.markdown("• Personas rotate automatically")
            st.markdown("• AIs can use **@mentions** to address each other")
            st.markdown(f"• Each AI sees the last **{st.session_state.settings['context_messages']}** messages")
            
            if st.session_state.settings.get('enable_thinking', True):
                st.markdown("• 🧠 **Thinking enabled** - View AI reasoning in expanders")
                st.markdown("• Works best with **deepseek-r1** and compatible models")

    def run(self):
        """Main Streamlit app interface"""
        st.set_page_config(
            page_title="AI Backroom",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        self.sidebar_ui()
        
        # Show a welcome message for new users
        if not st.session_state.personas:
            st.info("👋 **Welcome to AI Backroom!** Start by creating your first AI persona in the **Personas** tab, then head to **Conversation** to begin chatting!")       
        
        # Main content with tabs navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Conversation", 
            "🤖 Personas", 
            "⚙️ Settings", 
            "📁 Export & Logs"
        ])
        
        with tab1:
            if not st.session_state.personas:
                st.warning("⚠️ No personas created yet! Please create at least one persona in the **Personas** tab to start conversations.")
            else:
                self.conversation_ui()
        
        with tab2:
            self.persona_management_ui()
        
        with tab3:
            self.settings_ui()
        
        with tab4:
            self.export_ui()


def main():
    """Main entry point for Streamlit app"""
    app = StreamlitBackroomApp()
    app.run()


if __name__ == "__main__":
    main() 