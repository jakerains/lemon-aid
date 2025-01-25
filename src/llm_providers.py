"""
LLM Provider configurations and utilities for Lemon-Aid.
Handles multiple providers that support the OpenAI schema.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import re
from openai import AsyncOpenAI
import aiohttp
import asyncio
import time
from rich.live import Live
from rich.layout import Layout
import sys
import msvcrt
import requests

console = Console()

def hidden_input(prompt: str = "") -> str:
    """Get hidden input from user that allows pasting."""
    if prompt:
        console.print(prompt, end="")
    chars = []
    while True:
        char = msvcrt.getwch()
        if char == '\r':  # Enter key
            console.print()
            return ''.join(chars)
        elif char == '\x03':  # Ctrl+C
            raise KeyboardInterrupt
        elif char == '\b':  # Backspace
            if chars:
                chars.pop()
                # Move cursor back and clear the character
                sys.stdout.write('\b \b')
        else:
            chars.append(char)
            # Print * for each character
            sys.stdout.write('*')

@dataclass
class LLMProvider:
    name: str
    base_url: str
    api_key_env: str
    available_models: List[str]
    default_model: str
    max_tokens: int = 8192
    supports_functions: bool = True
    description: str = ""
    context_window: Optional[int] = None
    model_type: Optional[str] = None
    client_type: str = "openai"  # Can be "openai" or "ollama"
    _api_key: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment or stored value."""
        if self._api_key:
            return self._api_key
        return os.getenv(self.api_key_env)

    @api_key.setter
    def api_key(self, value: str):
        """Set API key value."""
        self._api_key = value

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has API key or is local)."""
        if self.client_type == "ollama":
            try:
                # Check if Ollama is running locally
                response = requests.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    # Update available models list dynamically
                    models = response.json()
                    self.available_models = sorted([model['name'] for model in models['models']])
                    return True
            except:
                return False
        return bool(self.api_key)

    def get_key_pattern(self) -> Tuple[str, str]:
        """Get the expected pattern and example for the provider's API key."""
        patterns = {
            "OpenAI": (r"^sk-[A-Za-z0-9-_]{32,}$", "sk-proj-xxxxxxxxxxxx..."),
            "DeepSeek": (r"^sk-[A-Za-z0-9]{32}$", "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"),
            "Groq": (r"^gsk_[A-Za-z0-9_-]{48,}$", "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
            "Ollama": (r".*", "No API key needed - local service"),
        }
        return patterns.get(self.name, (r".+", "any-valid-key-format"))

    async def validate_key(self, key: str) -> Tuple[bool, str]:
        """Validate the API key format and test the connection."""
        if self.client_type == "ollama":
            try:
                # Check if Ollama service is running
                response = requests.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    return True, "Ollama service is running"
                return False, "Ollama service is not running"
            except Exception as e:
                return False, f"Cannot connect to Ollama service: {str(e)}"
        
        pattern, example = self.get_key_pattern()
        
        # First check the format
        if not re.match(pattern, key):
            return False, f"Invalid key format. Expected format like: {example}"
        
        # Then test the connection with provider-specific checks
        try:
            # For OpenAI-compatible APIs (OpenAI, DeepSeek, Groq)
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {key}"}
                
                # For Groq, skip the models list check and go straight to chat completion
                if self.name == "Groq":
                    test_data = {
                        "model": "mixtral-8x7b-32768",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 1
                    }
                    async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=test_data) as resp:
                        if resp.status != 200:
                            return False, f"API key validation failed: {resp.status} {resp.reason}"
                        return True, "API key validated successfully"
                
                # For other providers, try models list first
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    if response.status != 200:
                        return False, f"API key validation failed: {response.status} {response.reason}"
                    
                    # For each provider, test with a specific model
                    if self.name == "OpenAI":
                        test_data = {
                            "model": "gpt-3.5-turbo",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1
                        }
                        async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=test_data) as resp:
                            if resp.status != 200:
                                return False, "API key validation failed: Unable to make a test completion request"
                    
                    elif self.name == "DeepSeek":
                        test_data = {
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1
                        }
                        async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=test_data) as resp:
                            if resp.status != 200:
                                return False, "API key validation failed: Unable to access DeepSeek chat model"
                
                return True, "API key validated successfully"
            
        except aiohttp.ClientError as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"API key validation failed: {str(e)}"

    async def prompt_for_api_key(self) -> bool:
        """Prompt user for API key if not found in environment."""
        if self.client_type == "ollama":
            # For Ollama, just check if the service is running
            is_valid, message = await self.validate_key("")
            if is_valid:
                console.print(Panel(
                    "[green]✓ Ollama service detected and running![/green]",
                    border_style="green",
                    box=box.ROUNDED
                ))
                return True
            else:
                console.print(Panel(
                    f"[red]✗ {message}[/red]\n"
                    "[yellow]Please make sure Ollama is installed and running.[/yellow]\n"
                    "Visit: [link=https://ollama.ai]https://ollama.ai[/link] for installation instructions.",
                    border_style="red",
                    box=box.ROUNDED
                ))
                return False
        
        pattern, example = self.get_key_pattern()
        
        # Create a styled panel with provider-specific information
        info_panel = Panel(
            f"[yellow]No API key found for {self.name}.[/yellow]\n\n"
            f"[cyan]Where to get your key:[/cyan]\n"
            f"→ Visit: [link=https://platform.{self.name.lower()}.com]https://platform.{self.name.lower()}.com[/link]\n\n"
            f"[cyan]Expected format:[/cyan]\n"
            f"→ Example: [dim]{example}[/dim]",
            title="🔑 [bold yellow]API Key Required[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(info_panel)
        
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            remaining = max_attempts - attempt
            
            console.print(f"\n[yellow]Please enter your {self.name} API key:[/yellow]")
            try:
                key = input().strip()
            except KeyboardInterrupt:
                console.print("\n[yellow]API key input cancelled.[/yellow]")
                return False
            
            if not key:
                if remaining > 0:
                    console.print(f"[red]No key entered. {remaining} attempts remaining.[/red]")
                continue
            
            # Show validation progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task("[cyan]Validating API key...", total=None)
                is_valid, message = await self.validate_key(key)
            
            if is_valid:
                self.api_key = key
                console.print(Panel(
                    f"[green]✓ {self.name} API key validated and saved successfully![/green]",
                    border_style="green",
                    box=box.ROUNDED
                ))
                return True
            else:
                if remaining > 0:
                    console.print(Panel(
                        f"[red]✗ {message}[/red]\n"
                        f"[yellow]{remaining} attempts remaining[/yellow]",
                        border_style="red",
                        box=box.ROUNDED
                    ))
                else:
                    console.print(Panel(
                        f"[red]✗ {message}[/red]\n"
                        "[red]Maximum attempts reached. Please try again later.[/red]",
                        border_style="red",
                        box=box.ROUNDED
                    ))
        
        return False

PROVIDERS = {
    "openai": LLMProvider(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        available_models=[
            "gpt-4o",  # Latest large GA model
            "gpt-4o-mini",  # Latest small GA model
        ],
        default_model="gpt-4o-mini",
        max_tokens=16384,
        context_window=128000,
        description="OpenAI's GPT models",
        model_type="Text Generation, Reasoning",
        supports_functions=True,
        client_type="openai"
    ),
    "ollama": LLMProvider(
        name="Ollama",
        base_url="http://localhost:11434",
        api_key_env="",  # No API key needed for local Ollama
        available_models=[],  # Empty list, will be populated dynamically
        default_model="",  # Will be set after fetching models
        max_tokens=4096,
        context_window=8192,
        description="Local Ollama models",
        model_type="Text Generation",
        supports_functions=False,
        client_type="ollama"
    ),
    "deepseek": LLMProvider(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        available_models=[
            "deepseek-chat",
            "deepseek-coder",
        ],
        default_model="deepseek-chat",
        max_tokens=8192,
        context_window=16000,
        description="DeepSeek's chat and code models",
        model_type="Text Generation, Code",
        supports_functions=True,
        client_type="openai"
    ),
    "groq": LLMProvider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        available_models=[
            # Latest Llama 3.3 models
            "llama-3.3-70b-specdec",
            "llama-3.3-70b-versatile",
            # Llama 3.1 models
            "llama-3.1-70b-versatile",
            # Mixtral models
            "mixtral-8x7b-32768",
            # Tool-use specialized models
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview",
            # Google models
            "gemma-7b-it",
            "gemma2-9b-it"
        ],
        default_model="llama-3.3-70b-versatile",  # Updated default to latest model
        max_tokens=32768,
        context_window=32768,
        description="Ultra-fast inference with Groq",
        model_type="Text Generation",
        supports_functions=True,
        client_type="openai"
    ),
}

async def select_provider() -> Optional[LLMProvider]:
    """Display available providers and let user select one."""
    # Create a table to display provider information
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title="[bold]Available LLM Providers[/bold]"
    )
    
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Provider", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("Status", style="cyan")
    
    # Add providers to table
    for i, (key, provider) in enumerate(PROVIDERS.items(), 1):
        status = "[green]Available[/green]" if provider.is_available else "[red]Needs Setup[/red]"
        table.add_row(
            str(i),
            provider.name,
            provider.description,
            status
        )
    
    console.print(table)
    console.print("\n[yellow]Select a provider by number:[/yellow]")
    
    while True:
        try:
            choice = int(Prompt.ask("Choice", choices=[str(i) for i in range(1, len(PROVIDERS) + 1)]))
            provider = list(PROVIDERS.values())[choice - 1]
            
            if not provider.is_available:
                if not await provider.prompt_for_api_key():
                    if not Prompt.ask(
                        "\n[yellow]Provider setup failed. Try another?[/yellow]",
                        choices=["y", "n"],
                        default="y"
                    ) == "y":
                        return None
                    console.print(table)
                    continue
            
            return provider
            
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            return None

def select_model(provider: LLMProvider) -> Optional[str]:
    """Display available models for the selected provider and let user select one."""
    if not provider:
        return None
        
    # For Ollama, refresh the model list
    if provider.client_type == "ollama":
        try:
            response = requests.get(f"{provider.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                provider.available_models = sorted([model['name'] for model in models['models']])
                if not provider.default_model and provider.available_models:
                    # Set default model to first available model
                    provider.default_model = provider.available_models[0]
        except Exception as e:
            console.print(f"[red]Error fetching Ollama models: {str(e)}[/red]")
            return None
    
    if not provider.available_models:
        console.print("[red]No models available for this provider.[/red]")
        return None
        
    # Create a table to display model information
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        title=f"[bold]{provider.name} Models[/bold]"
    )
    
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Model", style="green")
    if provider.client_type != "ollama":
        table.add_column("Max Tokens", style="yellow")
        table.add_column("Context", style="cyan")
    
    # Add models to table
    for i, model in enumerate(provider.available_models, 1):
        if provider.client_type == "ollama":
            table.add_row(
                str(i),
                model
            )
        else:
            table.add_row(
                str(i),
                model,
                str(provider.max_tokens),
                str(provider.context_window or "Default")
            )
    
    console.print(table)
    console.print("\n[yellow]Select a model by number:[/yellow]")
    
    while True:
        try:
            choice = int(Prompt.ask("Choice", choices=[str(i) for i in range(1, len(provider.available_models) + 1)]))
            return provider.available_models[choice - 1]
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            return None

def create_client(provider: LLMProvider) -> Optional[Union[AsyncOpenAI, Any]]:
    """Create an API client for the selected provider."""
    if not provider or not provider.is_available:
        return None
        
    try:
        if provider.client_type == "ollama":
            # For Ollama, we don't need to create a persistent client
            # We'll create sessions as needed in the generate function
            return "ollama"
        else:
            # For OpenAI-compatible APIs
            return AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url
            )
    except Exception as e:
        console.print(f"[red]Error creating client: {str(e)}[/red]")
        return None 