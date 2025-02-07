"""CLI prompt utilities."""

from typing import List, Tuple
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.panel import Panel
from rich.table import Table

console = Console()

def display_menu_options(title: str, options: List[str]) -> None:
    """Display menu options in a nicely formatted table."""
    table = Table(title=title, show_header=True, header_style="bold yellow")
    table.add_column("Option", style="cyan")
    table.add_column("Description", style="green")
    
    for i, option in enumerate(options, 1):
        table.add_row(f"[bold cyan]{i}[/bold cyan]", option)
    
    console.print(Panel(table, border_style="yellow"))

def get_provider_selection(providers: List[str]) -> str:
    """Get provider selection from user with validation."""
    while True:
        console.print("\n[bold yellow]Available Providers[/bold yellow]")
        display_menu_options("Select a Provider", providers)
        
        choice = Prompt.ask(
            "[bold cyan]Enter provider number[/bold cyan]",
            choices=[str(i) for i in range(1, len(providers) + 1)],
            show_choices=False
        )
        
        selected = providers[int(choice) - 1]
        if Confirm.ask(f"You selected [cyan]{selected}[/cyan]. Is this correct?"):
            return selected
        console.print("[yellow]Let's try again...[/yellow]")

def get_model_selection(models: List[str]) -> str:
    """Get model selection from user with validation."""
    while True:
        console.print("\n[bold yellow]Available Models[/bold yellow]")
        display_menu_options("Select a Model", models)
        
        choice = Prompt.ask(
            "[bold cyan]Enter model number[/bold cyan]",
            choices=[str(i) for i in range(1, len(models) + 1)],
            show_choices=False
        )
        
        selected = models[int(choice) - 1]
        if Confirm.ask(f"You selected [cyan]{selected}[/cyan]. Is this correct?"):
            return selected
        console.print("[yellow]Let's try again...[/yellow]")

def get_batch_settings() -> Tuple[int, int]:
    """Get batch processing settings from user with validation."""
    while True:
        console.print("\n[bold yellow]Batch Generation Settings[/bold yellow]")
        
        settings_panel = Panel(
            "[cyan]Configure your generation batch settings:[/cyan]\n"
            "• Batch size: Number of prompts to generate in parallel\n"
            "• Total samples: Total number of training examples to generate",
            border_style="yellow"
        )
        console.print(settings_panel)
        
        batch_size = IntPrompt.ask(
            "[bold cyan]Enter batch size[/bold cyan]",
            default=10,
            show_default=True
        )
        
        total_samples = IntPrompt.ask(
            "[bold cyan]Enter total samples to generate[/bold cyan]",
            default=100,
            show_default=True
        )
        
        # Show summary and confirm
        console.print(f"\nSettings Summary:")
        console.print(f"• Batch Size: [cyan]{batch_size}[/cyan]")
        console.print(f"• Total Samples: [cyan]{total_samples}[/cyan]")
        
        if Confirm.ask("Are these settings correct?"):
            return batch_size, total_samples
        console.print("[yellow]Let's configure the settings again...[/yellow]") 