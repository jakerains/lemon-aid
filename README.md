# Lemon-Aid 🍋

<p align="center">
  <img src="lemonaidlogo.png" alt="Lemon-Aid Logo" width="400"/>
</p>

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: Commons Clause](https://img.shields.io/badge/License-Commons%20Clause-red.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A powerful tool to help generate high-quality training data for fine-tuning language models. Lemon-Aid supports multiple LLM providers including OpenAI, DeepSeek, Groq, and Hugging Face to generate diverse, high-quality question-answer pairs while maintaining consistency and avoiding common pitfalls.

## ✨ Features

- 🌐 Multi-provider support with unified interface
- 🤖 Latest models from OpenAI, DeepSeek, Groq, and Hugging Face
- 🚀 Asynchronous generation of training data
- 🔍 Automatic duplicate detection and avoidance
- 🧠 Dynamic prompt adaptation based on generation quality
- 💾 Progress saving and recovery
- 📊 Rich console interface with detailed progress tracking
- ⚙️ Configurable system prompts with user approval workflow
- 📄 JSONL output format compatible with fine-tuning

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- API keys for your chosen providers:
  - OpenAI ([Get key](https://platform.openai.com))
  - DeepSeek ([Get key](https://platform.deepseek.com))
  - Groq ([Get key](https://console.groq.com))
  - Hugging Face ([Get key](https://huggingface.co/settings/tokens))

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
# Edit .env with your API keys and preferences
```

## 💻 Usage

Run the script:
```bash
python lemonaid.py
```

You'll be prompted to:
1. Select your preferred LLM provider
2. Choose a specific model from the provider
3. Enter your topic for training data generation
4. Specify the number of entries you want
5. Provide an output filename (defaults to training_data.jsonl)

The tool will then:
1. Generate and let you approve a system prompt
2. Show example Q&A pairs for your approval
3. Generate the requested number of entries with progress tracking

## 🤖 Supported Models

### OpenAI
- GPT-4o (Latest large GA model, 128k context)
- GPT-4o-mini (Latest small GA model, 128k context)

### DeepSeek
- DeepSeek-chat (Latest V3 model)
- DeepSeek-reasoner (Latest R1 model)

### Groq
- Llama-3.3-70b-versatile (Latest Llama 3.3 for general use)
- Llama-3.3-70b-specdec (Latest Llama 3.3 for specialized tasks)
- Llama-3.1-8b-instant (Smaller, faster Llama 3.1 model)
- Llama-3.2-90b-vision-preview (Latest vision-enabled Llama model)

### Hugging Face
- Latest open source models including Llama, Mixtral, Gemma, Yi, and more
- Code-specialized models like StarCoder
- Embedding models for specialized tasks

## 📄 Output Format

The tool generates JSONL files with entries formatted for fine-tuning:
```json
{
    "text": "Below is an instruction that describes a task...\n\n### Instruction:\n[Question]\n\n### Response:\n[Answer]\n<|end_of_text|>"
}
```

## 🛡️ Error Handling

- ♻️ Automatic retry with exponential backoff for API failures
- 💾 Progress saving on interruption
- 🔍 Duplicate detection and avoidance
- 🚨 Graceful error handling with informative messages

## 📦 Dependencies

- OpenAI SDK (>=1.0.0)
- Hugging Face Hub (>=0.20.3)
- See [requirements.txt](requirements.txt) for the full list of dependencies

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

- OpenAI, DeepSeek, Groq, and Hugging Face for their APIs
- All contributors and users of this project 