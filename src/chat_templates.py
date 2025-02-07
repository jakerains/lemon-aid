"""
Chat template handling for Lemon-Aid.
Implements support for various chat template formats using Hugging Face's transformers chat templates.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum
import json

class TemplateFormat(Enum):
    """Supported chat template formats."""
    CHATML = "chatml"
    LLAMA = "llama"
    ALPACA = "alpaca"
    MISTRAL = "mistral"
    ZEPHYR = "zephyr"
    CUSTOM = "custom"

@dataclass
class ChatTemplate:
    """Chat template configuration."""
    name: str
    format: TemplateFormat
    template: str
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    stop_token: str = "<|stop|>"

    @classmethod
    def from_format(cls, format: Union[str, TemplateFormat]) -> "ChatTemplate":
        """Create a template from a predefined format."""
        if isinstance(format, str):
            format = TemplateFormat(format.lower())

        templates = {
            TemplateFormat.CHATML: ChatTemplate(
                name="ChatML",
                format=TemplateFormat.CHATML,
                template="""{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}""",
                system_token="<|im_start|>system",
                user_token="<|im_start|>user",
                assistant_token="<|im_start|>assistant",
                stop_token="<|im_end|>"
            ),
            TemplateFormat.LLAMA: ChatTemplate(
                name="Llama",
                format=TemplateFormat.LLAMA,
                template="""{% if messages[0]['role'] == 'system' %}
{{ bos_token + '[INST] ' + messages[0]['content'] + ' [/INST]' }}
{% endif %}
{% for message in messages[1:] %}
{% if message['role'] == 'user' %}
{{ '[INST] ' + message['content'] + ' [/INST]' }}
{% elif message['role'] == 'assistant' %}
{{ message['content'] + '</s>' }}
{% endif %}
{% endfor %}""",
                bos_token="<s>",
                eos_token="</s>",
                system_token="[INST]",
                user_token="[INST]",
                assistant_token="[/INST]",
                stop_token="</s>"
            ),
            TemplateFormat.ALPACA: ChatTemplate(
                name="Alpaca",
                format=TemplateFormat.ALPACA,
                template="""{% if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] }}
{% endif %}
{% for message in messages[1:] %}
{% if message['role'] == 'user' %}
### Instruction:
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
### Response:
{{ message['content'] }}
{% endif %}
{% endfor %}""",
                system_token="### System:",
                user_token="### Instruction:",
                assistant_token="### Response:",
                stop_token="\n\n"
            ),
            # Add more template formats as needed
        }

        return templates.get(format, templates[TemplateFormat.CHATML])

    def apply(self, messages: List[Dict[str, str]]) -> str:
        """Apply the template to a list of messages."""
        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("Jinja2 is required for chat templates. Install with: pip install jinja2")

        template = Template(self.template)
        return template.render(
            messages=messages,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            system_token=self.system_token,
            user_token=self.user_token,
            assistant_token=self.assistant_token,
            stop_token=self.stop_token
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ChatTemplate":
        """Create a template from a JSON string."""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            format=TemplateFormat(data["format"]),
            template=data["template"],
            bos_token=data.get("bos_token", "<s>"),
            eos_token=data.get("eos_token", "</s>"),
            system_token=data.get("system_token", "<|system|>"),
            user_token=data.get("user_token", "<|user|>"),
            assistant_token=data.get("assistant_token", "<|assistant|>"),
            stop_token=data.get("stop_token", "<|stop|>")
        )

    def to_json(self) -> str:
        """Convert template to JSON string."""
        return json.dumps({
            "name": self.name,
            "format": self.format.value,
            "template": self.template,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "system_token": self.system_token,
            "user_token": self.user_token,
            "assistant_token": self.assistant_token,
            "stop_token": self.stop_token
        }, indent=2) 