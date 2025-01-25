# 🍋 Lemon-Aid v1.0.2

Easy Training Data infused with Citrus!

## Overview

Lemon-Aid is a powerful tool for generating high-quality training data using multiple LLM providers. It features a unified interface for working with various language models, robust error handling, and an intuitive console interface.

## Features

- **Multi-Provider Support**
  - OpenAI (GPT-4o and GPT-4o-mini)
  - DeepSeek (Chat and code models)
  - Groq (Llama 3.x and Mixtral models)
  - Ollama (Local models with dynamic discovery)

- **Advanced Generation**
  - Asynchronous batch processing
  - Dynamic prompt adaptation
  - Duplicate detection and filtering
  - Progress saving and recovery
  - Customizable response lengths

- **Rich Interface**
  - Interactive provider/model selection
  - Real-time progress tracking
  - Detailed error reporting
  - Generation statistics
  - Colorful console display

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/jakerains/lemon-aid.git
   cd lemon-aid
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Copy `.env.example` to `.env`
   - Add your API keys for desired providers
   - For Ollama, ensure the service is running locally

4. Run the tool:
   ```bash
   python run.py
   ```

## Project Structure

```
lemon-aid/
├── src/               # Source code
│   ├── lemonaid.py    # Main application
│   └── llm_providers.py # Provider implementations
├── assets/            # Images and resources
│   └── lemon-aid-big-ascii-art.txt
├── data/              # Generated data files
│   └── training_data.jsonl
├── docs/              # Documentation
│   ├── CHANGELOG.md
│   └── project-map.md
├── run.py            # Root launcher script
├── requirements.txt   # Python dependencies
└── .env.example      # Environment setup guide
```

## Output Format

The tool generates training data in JSONL format with special tokens:

```json
{
  "text": "<|im_start|>system\nYou are a knowledgeable assistant...<|im_end|>\n<|im_start|>user\nQuestion here...<|im_end|>\n<|im_start|>assistant\nAnswer here...<|im_end|>"
}
```

## Error Handling

- Graceful shutdown on interruption
- Automatic retry mechanisms
- Rate limit handling
- Progress preservation
- Detailed error tracking

## Requirements

- Python 3.8+
- aiohttp>=3.9.3
- backoff>=2.2.1
- openai>=1.12.0
- rich>=13.7.0
- python-dotenv>=1.0.0
- tqdm>=4.66.2

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- OpenAI, DeepSeek, Groq, and Ollama for their APIs
- All contributors and users of this project 