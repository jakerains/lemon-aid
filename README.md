# Lemon-Aid

<div align="center">
  <img src="assets/lemonaidlogo.png" alt="Lemon-Aid Logo" width="200"/>
  
  <h3>🍋 Easy Training Data Generation infused with Citrus! 🍋</h3>
  
  <p><strong>Lemon-Aid v1.0.3</strong></p>

  <p>
    <a href="https://github.com/jakerains/lemon-aid/releases/latest">
      <img alt="Version" src="https://img.shields.io/badge/version-1.0.3-brightgreen.svg"/>
    </a>
    <a href="LICENSE">
      <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
    </a>
    <a href="requirements.txt">
      <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8+-blue.svg"/>
    </a>
    <img alt="Last Updated" src="https://img.shields.io/badge/Last%20Updated-March%202024-green"/>
    <br/>
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"/>
    <img alt="Maintained" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg"/>
  </p>

  <p>
    <a href="#overview">Overview</a> •
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#documentation">Docs</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>

<div align="center">
  <img src="assets/screengrab.png" alt="Lemon-Aid Demo" width="600"/>
  <p><em>Generate high-quality training data with a beautiful interface</em></p>
</div>

## 🌟 Highlights

> 🚀 **Fast & Efficient**: Asynchronous batch processing with smart rate limiting
>
> 🎯 **Multi-Provider**: Support for OpenAI, DeepSeek, Groq, and local Ollama models
>
> 💡 **Smart Generation**: Dynamic prompt adaptation and duplicate detection
>
> 🛡️ **Robust**: Built-in error handling and progress preservation

<details>
<summary>🎥 Show Demo</summary>
<br>

![Demo Animation](assets/demo.gif)

</details>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Output Format](#output-format)
- [Error Handling](#error-handling)
- [Requirements](#requirements)
- [Contributing](#-contributing)
- [License](#license)
- [Acknowledgments](#-acknowledgments)

## 🔍 Overview

Lemon-Aid is a powerful tool for generating high-quality training data using multiple LLM providers. It features a unified interface for working with various language models, robust error handling, and an intuitive console interface.

## ✨ Features

<table>
<tr>
<td>

### 🔌 Provider Support

- OpenAI (GPT-4o and GPT-4o-mini)
- DeepSeek (Chat and code models)
- Groq (Llama 3.x and Mixtral models)
- Ollama (Local models)

</td>
<td>

### ⚡ Advanced Features

- Async batch processing
- Dynamic prompt adaptation
- Duplicate detection
- Progress saving
- Response length control

</td>
</tr>
</table>

## 🚀 Quick Start

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

## 📁 Project Structure

```plaintext
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

## 📤 Output Format

The tool generates training data in JSONL format with special tokens:

```jsonc
{
  "text": "<|im_start|>system\nYou are a knowledgeable assistant...<|im_end|>\n<|im_start|>user\nQuestion here...<|im_end|>\n<|im_start|>assistant\nAnswer here...<|im_end|>"
}
```

## ⚡ Error Handling

- Graceful shutdown on interruption
- Automatic retry mechanisms
- Rate limit handling
- Progress preservation
- Detailed error tracking

## 📋 Requirements

- Python 3.8+
- Dependencies:
  ```plaintext
  aiohttp>=3.9.3
  backoff>=2.2.1
  openai>=1.12.0
  rich>=13.7.0
  python-dotenv>=1.0.0
  tqdm>=4.66.2
  ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI, DeepSeek, Groq, and Ollama for their APIs
- All contributors and users of this project 

## 📚 Documentation

<details>
<summary>📖 Detailed Documentation</summary>

- [Full Documentation](docs/README.md)
- [Changelog](docs/CHANGELOG.md)
- [Project Map](docs/project-map.md)
- [Contributing Guidelines](CONTRIBUTING.md)

</details>

---

<div align="center">
  <p>Made with 🍋 by <a href="https://github.com/jakerains">Jake Rains</a></p>
  <p>
    <a href="https://github.com/jakerains/lemon-aid/issues">Report Bug</a>
    ·
    <a href="https://github.com/jakerains/lemon-aid/issues">Request Feature</a>
  </p>
</div> 