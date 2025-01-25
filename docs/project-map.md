# Lemon-Aid Project Map

## Core Components

### Main Script (lemonaid.py)
- Entry point for the application
- Handles user interaction and configuration
- Manages the training data generation pipeline

### Key Functions

#### Data Generation
- `generate_training_data()`: Main async function orchestrating the generation process
- `generate_qa_pair_async()`: Individual Q&A pair generation
- `generate_batch()`: Parallel generation of multiple Q&A pairs

#### Text Processing
- `clean_text()`: Sanitizes and normalizes generated text
- `format_instruction()`: Formats Q&A pairs for fine-tuning
- `get_dynamic_aspect()`: Generates varied aspects for questions
- `adapt_prompt_strategy()`: Adjusts prompts based on generation quality

#### Display and Progress
- Rich console interface for progress tracking
- Progress saving and recovery system
- Detailed statistics and formatting

### LLM Provider System
- **Location**: `llm_providers.py`
- **Purpose**: Manages multiple LLM providers with OpenAI-compatible APIs
- **Supported Providers**:
  - OpenAI (GPT-4o models)
  - DeepSeek (V3 and R1 models)
  - Groq (Llama 3.x models)
  - Hugging Face (Various open source models)
- **Features**:
  - Provider selection and management
  - Model selection within each provider
  - Unified client interface
  - Support for both OpenAI and Hugging Face APIs
  - Context window management
  - Function calling support
  - Streaming capabilities

### Configuration
- **Location**: `.env` and `.env.example`
- **Purpose**: Manages API keys and provider configurations
- **Features**:
  - API key management
  - Base URL configurations
  - Model-specific settings
  - Batch processing parameters

## Technical Architecture

### API Integration
- Asynchronous DeepSeek API calls
- Rate limiting and concurrency management
- Error handling and retries

### Data Management
- JSONL file format for training data
- Progress saving with temporary files
- Duplicate detection system

### User Interface
- Interactive prompt approval system
- Rich console output with formatting
- Progress bars and statistics

## Dependencies
- OpenAI API client for DeepSeek integration
- Rich for console formatting
- aiohttp for async operations
- python-dotenv for configuration
- backoff for retry logic
- tqdm for progress tracking
- OpenAI SDK (>=1.0.0)
- Hugging Face Hub (>=0.20.3)
- Async HTTP support
- Environment management

## File Structure
```
lemon-aid/
├── lemonaid.py          # Main application file
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
├── docs/
│   └── project-map.md  # This file
└── .env                # Environment configuration (not in repo)
```

## Implementation Details

### Generation Strategy
1. System prompt generation and approval
2. Initial examples generation and validation
3. Batch generation with dynamic adaptation
4. Duplicate detection and avoidance
5. Progress tracking and saving

### Error Handling Strategy
1. Automatic retries with exponential backoff
2. Progress saving on interruption
3. Graceful degradation
4. Informative error messages

### Output Format
Standardized JSONL format compatible with fine-tuning:
- Instruction/Response format
- Clear delineation of sections
- Consistent end-of-text markers 