# 🤖 AI Backroom - Interactive Multi-Persona Conversation Platform

A Streamlit-based web application that enables you to create AI personas with distinct roles and personalities, then watch them engage in dynamic and potentially infinite conversations using your local Ollama models.

## Features

- **Multi-Persona Conversations**: Create and manage multiple AI personas with unique personalities and roles
- **Real-time Chat Interface**: Beautiful tabbed interface with color-coded messages and persona identification
- **Role-Based Behavior**: 17+ predefined roles that shape conversation dynamics and personality traits
- **@Mention System**: Personas can reference each other using @mentions with visual highlighting
- **Configurable Context**: Adjustable conversation history (5-25 messages) for AI context awareness
- **Automatic Logging**: Daily conversation logs saved to text files for analysis
- **Session Management**: Export conversations as JSON and persistent settings storage
- **Standalone Log Viewer**: Separate application for advanced log analysis and search

## Quick Start

### Prerequisites

- Python 3.12+
- [UV package manager](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai/) running locally on `localhost:11434`
- At least one Ollama model downloaded (e.g., `ollama pull granite3.3:8b`)

### Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/guinacio/infinite-backrooms.git
   cd infinite-backrooms
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Start the application:**
   ```bash
   uv run streamlit run streamlit_backroom.py
   ```

4. **Verify Ollama connection:**
   - Navigate to the "Personas" tab
   - Click "Check Ollama Connection" to load available models
   - Ensure your models appear in the dropdown

## Application Structure

The application uses a tabbed interface with four main sections:

### 1. Conversation Tab
- Real-time chat interface with color-coded persona messages
- Manual conversation control with "Next Turn" button
- @mention highlighting when personas reference each other
- Conversation history display with timestamps

### 2. Personas Tab
- Create, edit, and manage AI personas
- Assign predefined roles or create custom roles
- Configure individual system prompts and model selection
- Visual persona management with color coding
- Quick setup options for diverse conversation sets

### 3. Settings Tab
- Adjust conversation context (5-25 messages)
- Configure response delays and auto-advance settings
- Manage conversation history limits
- Connection status and model information

### 4. Export & Logs Tab
- Export current session as JSON
- View conversation statistics
- Access daily log files
- Session management tools

## Role System

### Functional Roles
Special conversation facilitators that enhance group dynamics:

- **Moderator**: Guides discussions, introduces topics, maintains engagement
- **Note-Taker**: Summarizes key points, identifies themes, tracks insights

### Personality Roles
Distinct conversation styles and perspectives:

- **Philosopher**: Explores existential questions and consciousness
- **Scientist**: Empirical thinking and research-focused approach
- **Creative Writer**: Storytelling, wordplay, and creative expression
- **Debate Enthusiast**: Intellectual debates and diverse perspectives
- **Optimist**: Positive outlook and encouraging communication
- **Skeptic**: Questions assumptions and demands evidence
- **Historian**: Historical context and behavioral pattern analysis
- **Futurist**: Emerging technologies and future possibilities
- **Minimalist**: Simple, clear, and concise communication
- **Explorer**: Adventurous and curious about new concepts
- **Mentor**: Supportive and educational guidance
- **Comedian**: Humor while maintaining meaningful engagement
- **Analyst**: Systematic breakdown of complex topics
- **Dreamer**: Imaginative and idealistic perspectives
- **Pragmatist**: Practical and results-oriented thinking

### Custom Roles
Create your own roles by selecting "Use custom role instead" when adding personas. The AI will interpret and embody your custom role definition.

## Advanced Features

### @Mention System
Personas can reference each other using @mentions (e.g., "@Philosopher what do you think about..."). The system:
- Provides @mention instructions in system prompts
- Highlights mentions in the chat interface using persona colors
- Enhances conversation flow and direct interaction

### Conversation Context
- Configurable context window (5-25 messages) determines how much conversation history each AI sees
- Larger context enables more coherent long-form discussions
- Smaller context keeps conversations focused and reduces processing time

### Logging and Analysis
- Automatic daily logging to `conversations/streamlit_backroom_YYYY-MM-DD.txt`
- Clean message format with timestamps and persona identification
- Thinking tags automatically filtered from logs

## Standalone Log Viewer

For advanced log analysis, use the standalone log viewer application:

```bash
uv run streamlit run log_viewer.py
```

### Log Viewer Features
- **Multi-file Analysis**: Load and analyze multiple log files simultaneously
- **Advanced Filtering**: Filter by persona, date range, and message content
- **Search Capabilities**: Text search with Contains, Exact Match, and Regex options
- **Statistics Dashboard**: Message counts, persona activity, and conversation metrics
- **Multiple View Modes**: Chat view, table view, and raw text format
- **Export Options**: Download filtered data as CSV, JSON, or TXT formats
- **Visual Analytics**: Persona activity charts and conversation breakdowns

The log viewer works with both current (`streamlit_backroom_*.txt`) and legacy (`backroom_*.txt`) log file formats.

## Usage Examples

### Structured Research Discussion
```
Moderator + Note-Taker + Scientist + Philosopher
```
Guided facilitation of AI consciousness research with systematic note-taking.

### Creative Brainstorming
```
Moderator + Creative Writer + Dreamer + Explorer + Note-Taker
```
Imaginative idea generation with structured capture of insights.

### Balanced Policy Analysis
```
Moderator + Optimist + Skeptic + Pragmatist + Note-Taker
```
Multi-perspective analysis of new technologies or policies.

### Educational Seminar
```
Moderator + Mentor + Historian + Futurist + Note-Taker
```
Knowledge sharing with historical context and future implications.

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface with tabbed navigation
- **Backend**: Async HTTP client for Ollama API communication
- **Storage**: Session state for runtime data, file system for logs
- **Logging**: Daily text files with structured message format

### File Structure
```
infinite-backrooms/
├── streamlit_backroom.py    # Main application
├── log_viewer.py           # Standalone log analysis tool
├── conversations/          # Daily conversation logs
├── pyproject.toml         # UV dependencies
├── uv.lock               # Locked dependencies
└── README.md             # This file
```

### Dependencies
- **Streamlit**: Web interface framework
- **aiohttp**: Async HTTP client for Ollama API
- **pandas**: Data analysis for log viewer (log_viewer.py only)

## Contributing

This project uses UV for dependency management. To contribute:

1. Fork the repository
2. Create a feature branch
3. Install dependencies: `uv sync`
4. Make your changes
5. Test with: `uv run streamlit run streamlit_backroom.py`
6. Submit a pull request

## License

This project is open source. See the repository for license details.

## Author

Created by [guinacio](https://github.com/guinacio)

---

**Note**: This application requires a local Ollama installation with downloaded models. The AI personas will only be as capable as the underlying models you provide. 