# 🍋 Lemon-Aid

A powerful tool for generating high-quality training data using multiple LLM providers, designed specifically for fine-tuning language models.

## ✨ Features

- 🤖 Multi-provider support:
  - OpenAI (GPT-4o, GPT-4o-mini)
  - DeepSeek (chat, coder)
  - Groq (Llama 3.x, Mixtral)
  - Ollama (local models)
- 🎯 Dynamic response length control
- 💾 Progress saving and recovery
- 🔄 Interactive example generation
- 📝 System prompt customization
- 🎨 Rich console interface
- 📊 Detailed progress tracking
- ⚡ Parallel batch processing
- 🛡️ Comprehensive error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- API keys for your chosen providers:
  - OpenAI ([Get key](https://platform.openai.com))
  - DeepSeek ([Get key](https://platform.deepseek.com))
  - Groq ([Get key](https://console.groq.com))
  - Ollama ([Install locally](https://ollama.ai))

### Installation

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
```bash
cp .env.example .env
# Edit .env with your API keys
```

## 💻 Usage

Run the script:
```bash
python lemonaid.py
```

You'll be guided through:
1. Selecting your LLM provider
2. Choosing a specific model
3. Configuring your training data requirements:
   - Purpose/context
   - Answer length preference
   - Number of examples
   - Output filename

The tool will then:
1. Generate and let you approve a system prompt
2. Show example Q&A pairs for your approval
3. Generate the requested entries with progress tracking
4. Save results in JSONL format for fine-tuning

## 📄 Output Format

The tool generates JSONL files formatted for fine-tuning:
```json
{
    "text": "<|im_start|>system\nYou are a knowledgeable assistant...<|im_end|>\n<|im_start|>user\n[Question]<|im_end|>\n<|im_start|>assistant\n[Answer]<|im_end|>"
}
```

## 🛡️ Error Handling

- Automatic retries with exponential backoff
- Progress saving on interruption
- Rate limit management
- Graceful shutdown
- Detailed error messages

## 📦 Dependencies

See [requirements.txt](requirements.txt) for the full list:
- openai>=1.12.0
- python-dotenv>=1.0.0
- rich>=13.7.0
- aiohttp>=3.9.3
- backoff>=2.2.1
- tqdm>=4.66.2

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Commons Clause License with additional restrictions - see the [LICENSE](LICENSE) file for details. This means:

- ✅ You can download and use the software for personal use
- ❌ You cannot modify or adapt the software
- ❌ You cannot sell or redistribute the software
- ❌ You cannot use it for commercial purposes

All rights reserved by Jake Rains.

## 🙏 Acknowledgments

- OpenAI, DeepSeek, Groq, and Ollama for their APIs
- All contributors and users of this project 