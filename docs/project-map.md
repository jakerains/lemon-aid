# Lemon-Aid Project Map

## Overview
Lemon-Aid (v1.0.2) is a specialized tool designed for generating high-quality training data using multiple LLM providers. It features a unified interface for working with various language models, robust error handling, and an intuitive console interface.

## Core Components

### Main Script (src/lemonaid.py)
- Entry point and core functionality
- Asynchronous processing with rate limiting
- Rich console interface with progress tracking
- Dynamic batch size adjustment
- Comprehensive error handling and recovery

### LLM Providers (src/llm_providers.py)
- OpenAI (GPT-4o and GPT-4o-mini)
- DeepSeek (Chat and code models)
- Groq (Llama 3.x and Mixtral models)
- Ollama (Local models with dynamic discovery)
- Provider-specific API handling and rate limiting

### Root Launcher (lemonaid.py)
- Provides easy access from root directory
- Handles Python path configuration
- Maintains proper imports and error handling

## Core Features

### Data Generation
- Asynchronous batch processing
- Duplicate detection and filtering
- Progress saving and recovery
- Dynamic prompt adaptation
- Customizable response lengths

### User Interface
- Rich console display with ASCII art
- Interactive provider/model selection
- Progress bars and status updates
- Detailed error reporting
- Generation statistics

### Error Handling
- Graceful shutdown on interruption
- Automatic retry mechanisms
- Rate limit handling
- Progress preservation
- Detailed error tracking

## Technical Architecture

### Dependencies
- aiohttp: Async HTTP client
- openai: OpenAI API client
- rich: Console interface
- python-dotenv: Environment management
- backoff: Rate limiting

### File Structure
- src/: Source code
  - lemonaid.py: Main application
  - llm_providers.py: Provider implementations
- assets/: Images and resources
  - lemon-aid-big-ascii-art.txt: ASCII art for header
  - Various image files (.png, .jpg)
- data/: Generated data files
  - training_data.jsonl: Generated training data
  - Other JSONL files
- docs/: Documentation
  - CHANGELOG.md: Version history
  - project-map.md: This file
- lemonaid.py: Root launcher script
- .env.example: Environment setup guide
- requirements.txt: Python dependencies
- LICENSE: Project license
- README.md: Project documentation
- .gitignore: Git ignore rules

## Implementation Details

### Provider Integration
- Unified API interface
- Provider-specific rate limiting
- Dynamic model discovery (Ollama)
- API key validation
- Error handling per provider

### Output Format
- JSONL format with special tokens
- System/User/Assistant structure
- Duplicate detection
- Quality filtering
- Length control

### Configuration
- Environment-based setup
- Provider-specific settings
- Model selection
- Rate limiting configuration
- Output customization 