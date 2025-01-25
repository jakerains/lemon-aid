"""
LLM Provider configurations and utilities for Lemon-Aid.
Handles multiple providers that support the OpenAI schema.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
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
from huggingface_hub import InferenceClient
import aiohttp
import asyncio
import time
from rich.live import Live
from rich.layout import Layout
import sys
import msvcrt

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
    client_type: str = "openai"  # Can be "openai" or "huggingface"
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
        """Check if provider is available (has API key)."""
        return bool(self.api_key)

    def get_key_pattern(self) -> Tuple[str, str]:
        """Get the expected pattern and example for the provider's API key."""
        patterns = {
            "OpenAI": (r"^sk-[A-Za-z0-9-_]{32,}$", "sk-proj-xxxxxxxxxxxx..."),
            "DeepSeek": (r"^[A-Za-z0-9]{32,}$", "abcdefgh12345678ijklmnop90123456"),
            "Groq": (r"^gsk_[A-Za-z0-9]{32,}$", "gsk_abcdefgh12345678ijklmnop90123456"),
            "Hugging Face": (r"^hf_[A-Za-z0-9]{32,}$", "hf_abcdefgh12345678ijklmnop90123456"),
        }
        return patterns.get(self.name, (r".+", "any-valid-key-format"))

    async def validate_key(self, key: str) -> Tuple[bool, str]:
        """Validate the API key format and test the connection."""
        pattern, example = self.get_key_pattern()
        
        # First check the format
        if not re.match(pattern, key):
            return False, f"Invalid key format. Expected format like: {example}"
        
        # Then test the connection with provider-specific checks
        try:
            if self.client_type == "huggingface":
                client = InferenceClient(api_key=key)
                # Test with a simple model info request
                await client.get_model_info("gpt2")
                return True, "API key validated successfully"
                
            # For OpenAI-compatible APIs (OpenAI, DeepSeek, Groq)
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {key}"}
                
                # First try a models list request
                async with session.get(f"{self.base_url}/models", headers=headers) as response:
                    if response.status != 200:
                        return False, f"API key validation failed: {response.status} {response.reason}"
                    
                    # For each provider, test with a specific model
                    if self.name == "OpenAI":
                        # Test with a simple completion request
                        test_data = {
                            "model": "gpt-3.5-turbo",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1
                        }
                        async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=test_data) as resp:
                            if resp.status != 200:
                                return False, "API key validation failed: Unable to make a test completion request"
                    
                    elif self.name == "DeepSeek":
                        # Test with deepseek-chat model
                        test_data = {
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1
                        }
                        async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=test_data) as resp:
                            if resp.status != 200:
                                return False, "API key validation failed: Unable to access DeepSeek chat model"
                    
                    elif self.name == "Groq":
                        # Test with a Llama model
                        test_data = {
                            "model": "llama-3.1-8b-instant",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1
                        }
                        async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=test_data) as resp:
                            if resp.status != 200:
                                return False, "API key validation failed: Unable to access Groq models"
                
                return True, "API key validated successfully"
            
        except aiohttp.ClientError as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"API key validation failed: {str(e)}"

    async def prompt_for_api_key(self) -> bool:
        """Prompt user for API key if not found in environment."""
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
            "gpt-4o",       # Latest large GA model (2024-11-20)
            "gpt-4o-mini",  # Latest small GA model (2024-07-18)
        ],
        default_model="gpt-4o",
        max_tokens=16384,
        context_window=128000,
        description="Latest GPT-4o models with 128k context window",
        model_type="Text Generation, Reasoning",
        supports_functions=True,
        client_type="openai"
    ),
    "deepseek": LLMProvider(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",  # OpenAI-compatible endpoint
        api_key_env="DEEPSEEK_API_KEY",
        available_models=[
            "deepseek-chat",      # Latest V3 model (2024/12/26)
            "deepseek-reasoner",  # Latest R1 model (2025/01/20)
        ],
        default_model="deepseek-chat",
        max_tokens=8192,
        description=(
            "Latest DeepSeek models including V3 and R1. "
            "Supports streaming, JSON mode, function calling, context caching, "
            "chat prefix completion, and FIM completion."
        ),
        model_type="Text Generation, Reasoning, Chat",
        context_window=8192,
        supports_functions=True,
        client_type="openai"
    ),
    "groq": LLMProvider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        available_models=[
            "llama-3.3-70b-versatile",     # Latest Llama 3.3 for general use
            "llama-3.3-70b-specdec",       # Latest Llama 3.3 for specialized tasks
            "llama-3.1-8b-instant",        # Smaller, faster Llama 3.1 model
            "llama-3.2-90b-vision-preview" # Latest vision-enabled Llama model
        ],
        default_model="llama-3.3-70b-versatile",
        max_tokens=4096,
        description="Ultra-fast inference optimized Llama models with specialized variants",
        model_type="Text Generation, Chat, Vision",
        supports_functions=True,
        client_type="openai"
    ),
    "huggingface": LLMProvider(
        name="Hugging Face",
        base_url="https://api-inference.huggingface.co/models",
        api_key_env="HF_API_KEY",
        available_models=[
            # Latest Large Language Models
            "meta-llama/Llama-3.1-70B-Instruct",  # Latest Llama 3.1
            "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Latest Mixtral
            "google/gemma-7b-it",  # Latest Gemma
            "01-ai/Yi-34b-chat",  # Latest Yi
            "Qwen/Qwen1.5-72B-Chat",  # Latest Qwen
            
            # Code Models
            "bigcode/starcoder2-15b",  # Latest StarCoder
            
            # Embeddings
            "BAAI/bge-large-en-v1.5",  # Latest BGE embeddings
        ],
        default_model="meta-llama/Llama-3.1-70B-Instruct",
        max_tokens=4096,
        description="Direct access to latest open source models",
        model_type="Text Generation, Code, Embeddings",
        client_type="huggingface"
    ),
}

def get_available_providers() -> Dict[str, LLMProvider]:
    """Returns a dictionary of available providers (those with API keys configured)."""
    return {k: v for k, v in PROVIDERS.items() if v.is_available}

async def select_provider() -> Optional[LLMProvider]:
    """Interactive provider selection with API key prompting."""
    try:
        # Clear screen once at the start
        console.clear()

        # First display the ASCII art line by line
        try:
            with open('lemon-aid-big-ascii-art.txt', 'r', encoding='utf-8') as f:
                ascii_art = f.read().splitlines()
                # Filter empty lines at start and end
                while ascii_art and not ascii_art[0].strip():
                    ascii_art.pop(0)
                while ascii_art and not ascii_art[-1].strip():
                    ascii_art.pop()
                
                # Display each line with color
                for i, line in enumerate(ascii_art):
                    if "AID" in line:
                        parts = line.split("AID")
                        text = Text()
                        text.append(parts[0], style="yellow")
                        text.append("AID", style="green")
                        if len(parts) > 1:
                            text.append(parts[1], style="yellow")
                    elif 2 <= i <= 4 and "⣿" in line:  # Only color the leaf area
                        text = Text(line, style="green")
                    else:
                        text = Text(line, style="yellow")
                    console.print(text)
                    time.sleep(0.02)  # Slight delay between lines
        except FileNotFoundError:
            console.print(Panel(
                "🍋 [bold yellow]Welcome to Lemon-Aid![/bold yellow]",
                subtitle="[italic]Training Data Generation Tool[/italic]",
                box=box.ROUNDED,
                border_style="yellow",
                padding=(1, 2)
            ))

        # Add some spacing and pause
        console.print()
        time.sleep(0.5)  # Let user see the ASCII art

        # Create providers table
        table = Table(
            title="🤖 [bold]Available LLM Providers[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
            row_styles=["", "dim"]
        )
        
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Status", style="green", justify="center")
        table.add_column("Description", style="yellow", max_width=50)
        table.add_column("Types", style="magenta", no_wrap=True)

        # Show table with animation
        providers_list = list(PROVIDERS.values())
        with Live(table, console=console, refresh_per_second=8) as live:
            for provider in providers_list:
                status = "[green]✓ Connected[/green]" if provider.is_available else "[yellow]⚠ Needs API Key[/yellow]"
                table.add_row(
                    f"[bold]{provider.name}[/bold]",
                    status,
                    Text(provider.description, style="yellow", justify="left"),
                    Text(provider.model_type or "General Purpose", style="magenta")
                )
                time.sleep(0.15)  # Smooth animation for table rows

        # Show provider selection menu
        console.print()
        console.print("[cyan]Select a provider:[/cyan]")
        
        # Handle provider selection
        while True:
            try:
                # Display numbered list of providers
                for i, provider in enumerate(providers_list, 1):
                    status = "[green]✓[/green]" if provider.is_available else "[yellow]⚠[/yellow]"
                    console.print(f"[cyan]{i}[/cyan]. {status} {provider.name}")

                choice = int(input("\nChoice: ")) - 1
                if 0 <= choice < len(providers_list):
                    selected = providers_list[choice]
                    
                    # If provider needs API key, prompt for it
                    if not selected.is_available and not await selected.prompt_for_api_key():
                        console.print("[red]Cannot proceed without API key.[/red]")
                        return None
                    
                    return selected
                    
                console.print("[red]Invalid choice. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Provider selection cancelled. Exiting gracefully...[/yellow]")
        return None
    except Exception as e:
        console.print(f"\n[red]An error occurred during provider selection: {e}[/red]")
        return None

def create_client(provider: LLMProvider) -> Union[AsyncOpenAI, InferenceClient]:
    """Creates an appropriate client based on the provider type."""
    if provider.client_type == "huggingface":
        return InferenceClient(api_key=provider.api_key)
    else:
        return AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
        )

def select_model(provider: LLMProvider) -> Optional[str]:
    """Interactive model selection for the chosen provider."""
    try:
        console.print(f"\n[cyan]Available models for {provider.name}:[/cyan]")
        
        for i, model in enumerate(provider.available_models, 1):
            console.print(f"[cyan]{i}[/cyan]. {model}")

        while True:
            try:
                choice = int(input("\nSelect a model (enter number): ")) - 1
                if 0 <= choice < len(provider.available_models):
                    return provider.available_models[choice]
                console.print("[red]Invalid choice. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Model selection cancelled. Exiting gracefully...[/yellow]")
        return None
    except Exception as e:
        console.print(f"\n[red]An error occurred during model selection: {e}[/red]")
        return None 