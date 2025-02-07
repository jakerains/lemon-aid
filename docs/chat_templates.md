# Chat Templates in Lemon-Aid

## Overview

Lemon-Aid supports multiple chat template formats through its template system, allowing for consistent message formatting across different LLM providers. This system is based on Hugging Face's transformers chat templates.

## Supported Formats

### ChatML
- Used by: OpenAI, DeepSeek
- Format: `<|im_start|>role\ncontent<|im_end|>`
- Example:
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?
<|im_end|>
```

### Llama
- Used by: Llama models, Groq, Ollama
- Format: `<s>[INST] content [/INST]`
- Example:
```
<s>[INST] You are a helpful assistant. [/INST]
[INST] Hello! [/INST]
Hi! How can I help you today?</s>
```

### Alpaca
- Format: Simple instruction/response pairs
- Example:
```
### System:
You are a helpful assistant.

### Instruction:
Hello!

### Response:
Hi! How can I help you today?
```

## Custom Templates

You can define custom templates using JSON format:

```json
{
  "name": "Custom Format",
  "format": "custom",
  "template": "{% for message in messages %}...",
  "bos_token": "<s>",
  "eos_token": "</s>",
  "system_token": "<|system|>",
  "user_token": "<|user|>",
  "assistant_token": "<|assistant|>",
  "stop_token": "<|stop|>"
}
```

## Usage

Templates are automatically selected based on the provider and model being used. The system will:

1. Use the appropriate template for each provider
2. Handle message formatting consistently
3. Apply any necessary tokens or special formatting
4. Ensure compatibility with the model's expected input format

## Adding New Templates

To add a new template format:

1. Define the template in `src/chat_templates.py`
2. Add the format to the `TemplateFormat` enum
3. Create a template instance with appropriate tokens
4. Update the provider configuration to use the new format

## References

- [Hugging Face Chat Templates Documentation](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [ChatML Format Specification](https://github.com/openai/openai-python/blob/main/chatml.md)
- [Llama 2 Chat Format](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) 