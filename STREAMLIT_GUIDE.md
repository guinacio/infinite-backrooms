# 🤖 AI Backroom - Streamlit App Guide

## Overview
The AI Backroom Streamlit app is an interactive web interface that allows you to create AI personas with different roles and watch them engage in infinite conversations using your local Ollama models.

## 🚀 Quick Start

1. **Install and run the app:**
   ```bash
   uv run streamlit run streamlit_backroom.py
   ```

2. **Ensure Ollama is running:**
   - Make sure Ollama is running on `localhost:11434`
   - Have some models downloaded (e.g., `ollama pull granite3.3:8b`)

3. **Set up personas:**
   - Go to the "🤖 Personas" tab
   - Click "🔄 Check Ollama Connection" to load available models
   - Use "🎭 Add Diverse Conversation Set" for varied personalities, or "📋 Add Structured Discussion Set" for organized conversations with Moderator and Note-Taker

4. **Start conversations:**
   - Switch to the "💬 Conversation" tab
   - Click "▶️ Start Conversation" or "🔄 Next Turn" for manual control

## 🎭 Role-Based Personas

### Predefined Roles
The app includes 17+ predefined roles that shape how each AI persona behaves:

#### **Functional Roles** (Special conversation facilitators)
- **Moderator**: Guides discussions, asks follow-up questions, introduces new topics, and keeps conversations engaging
- **Note-Taker**: Summarizes key points, identifies themes, captures insights, and tracks idea evolution

#### **Personality Roles** (Distinct conversation styles)
- **Philosopher**: Explores deep questions about existence and consciousness
- **Scientist**: Approaches topics with empirical thinking and research focus
- **Creative Writer**: Loves storytelling, wordplay, and creative language use
- **Debate Enthusiast**: Enjoys intellectual debates and different perspectives
- **Optimist**: Maintains positive outlook and encourages others
- **Skeptic**: Questions assumptions and asks for evidence
- **Historian**: Fascinated by history and behavioral patterns
- **Futurist**: Interested in emerging technologies and future possibilities
- **Minimalist**: Values simplicity and clarity in communication
- **Explorer**: Adventurous and curious about new ideas
- **Mentor**: Supportive and educational in approach
- **Comedian**: Brings humor while engaging meaningfully
- **Analyst**: Systematic thinker who breaks down complex topics
- **Dreamer**: Imaginative and idealistic about possibilities
- **Pragmatist**: Practical and results-oriented

### Custom Roles
You can also create custom roles by:
1. Checking "Use custom role instead" when adding a persona
2. Entering your own role name (e.g., "Tech Enthusiast", "Poet", "Historian")
3. The AI will interpret and embody your custom role

## 🎨 Features

### Persona Management
- **Visual Design**: Each persona has a unique color for easy identification
- **Role Integration**: Roles automatically enhance system prompts
- **Flexible Configuration**: Combine predefined roles with custom system prompts
- **Easy Editing**: View, edit, and delete personas as needed

### Conversation Interface
- **Real-time Chat**: Beautiful chat interface with color-coded messages
- **Role Badges**: See each persona's role displayed in chat
- **Automatic Rotation**: Personas take turns in a fair rotation
- **Manual Control**: Advance conversations step-by-step or let them run automatically

### Settings & Export
- **Configurable Settings**: Adjust response delays, history limits, and auto-advance
- **Conversation Logging**: Automatic daily logs saved to text files
- **Session Export**: Download conversations as JSON for analysis
- **Persistent Storage**: Settings and personas survive app restarts

## 💡 Usage Tips

### Creating Interesting Conversations
1. **Mix complementary roles**: Try Philosopher + Scientist + Creative Writer
2. **Add tension**: Include both Optimist and Skeptic for dynamic debates
3. **Use custom prompts**: Add specific interests or knowledge areas
4. **Experiment with models**: Different models paired with same roles behave differently

### Managing Performance
- **Limit active personas**: 3-5 personas work well for most conversations
- **Adjust delays**: Increase delays for slower hardware
- **Monitor history**: Keep max history reasonable (50-100 messages)
- **Use appropriate models**: Smaller models respond faster

### Best Practices
- **Start with presets**: Use "Add Diverse Conversation Set" to begin
- **Observe patterns**: Notice how roles influence conversation topics
- **Iterate personas**: Refine roles and prompts based on results
- **Save interesting sessions**: Export conversations you want to keep

## 🛠️ Technical Notes

### Dependencies
- Python 3.12+
- Streamlit 1.28+
- aiohttp for Ollama API calls
- Local Ollama installation with models

### File Structure
- `streamlit_backroom.py`: Main application
- `conversations/`: Daily conversation logs
- Persona data stored in Streamlit session state

### Customization
The app is designed to be easily extensible:
- Add new role templates in `get_role_templates()`
- Modify system prompt generation in `generate_system_prompt()`
- Customize UI colors and styling in the HTML sections

## 🎯 Example Use Cases

1. **Structured Research Discussions**: Moderator + Note-Taker + Scientist + Philosopher examining AI consciousness with guided facilitation
2. **Creative Brainstorming Sessions**: Moderator + Creative Writer + Dreamer + Explorer with Note-Taker capturing brilliant ideas
3. **Balanced Policy Debates**: Moderator + Optimist + Skeptic + Pragmatist + Note-Taker analyzing new technologies
4. **Educational Seminars**: Moderator + Mentor + Historian + Futurist with Note-Taker summarizing key insights
5. **Philosophical Salons**: Note-Taker + Philosopher + Debate Enthusiast + Minimalist exploring deep questions
6. **Entertainment**: Comedian + Creative Writer + Dreamer for lighthearted exchanges (no facilitation needed!)

The AI Backroom becomes more interesting as personas develop their unique voices through role-based behavior. Experiment with different combinations to discover fascinating conversation dynamics! 