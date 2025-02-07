#!/usr/bin/env python3
"""
Simple setup script for Lemon-Aid.
Creates a virtual environment and installs dependencies.
"""

import os
import subprocess
import sys
import venv
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.prompt import Prompt
from rich import box
from rich.theme import Theme

# Create a custom theme for Lemon-Aid
custom_theme = Theme({
    "progress.description": "bright_yellow",
    "progress.percentage": "green",
    "progress.remaining": "bright_yellow",
    "bar.complete": "bright_green",
    "bar.finished": "bright_green",
    "bar.pulse": "bright_yellow",
    "info": "bright_yellow",
    "warning": "yellow",
    "error": "red",
    "success": "bright_green",
})

# Initialize rich console with theme
console = Console(theme=custom_theme)

def run_command(command, description=None, show_output=False):
    """Run a command with a progress bar."""
    try:
        with Progress(
            SpinnerColumn(style="bright_yellow"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                complete_style="bright_green",
                finished_style="bright_green",
                pulse_style="bright_yellow"
            ),
            TaskProgressColumn(style="bright_yellow"),
            TextColumn("[bright_yellow]{task.elapsed:.2f}s"),
            console=console,
            transient=True,
            expand=True
        ) as progress:
            task = progress.add_task(description or command, total=None)
            
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and show_output:
                    # Style the output based on content
                    line = output.strip()
                    if "Successfully installed" in line:
                        console.print(f"[success]{line}[/success]")
                    elif "Requirement already satisfied" in line:
                        console.print(f"[info]{line}[/info]")
                    elif any(word in line.lower() for word in ["warning", "warn"]):
                        console.print(f"[warning]{line}[/warning]")
                    elif any(word in line.lower() for word in ["error", "fail"]):
                        console.print(f"[error]{line}[/error]")
                    else:
                        console.print(f"[bright_yellow]{line}[/bright_yellow]")
                progress.update(task)
            
            returncode = process.poll()
            
            if returncode == 0:
                progress.update(task, completed=100)
                return True
            else:
                error = process.stderr.read()
                console.print(f"[error]Error: {error}[/error]")
                return False
                
    except Exception as e:
        console.print(f"[error]Error: {str(e)}[/error]")
        return False

def activate_venv():
    """Activate the virtual environment in the current process."""
    venv_path = os.path.abspath("venv")
    
    if sys.platform == "win32":
        activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        command = f"call {activate_script} && set"
    else:
        activate_script = os.path.join(venv_path, "bin", "activate")
        command = f"source {activate_script} && env"
    
    try:
        with Progress(
            SpinnerColumn(style="bright_yellow"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                complete_style="bright_green",
                finished_style="bright_green",
                pulse_style="bright_yellow"
            ),
            TaskProgressColumn(style="bright_yellow"),
            TextColumn("[bright_yellow]{task.elapsed:.2f}s"),
            console=console,
            transient=True,
            expand=True
        ) as progress:
            task = progress.add_task("Activating virtual environment...", total=None)
            output = subprocess.check_output(command, shell=True, text=True)
            
            # Parse and apply the new environment variables
            for line in output.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            
            progress.update(task, completed=100)
            return True
            
    except subprocess.CalledProcessError as e:
        console.print(f"[error]Failed to activate virtual environment: {e}[/error]")
        return False

def start_app():
    """Start the Lemon-Aid application."""
    if sys.platform == "win32":
        activate_cmd = ".\\venv\\Scripts\\activate && "
    else:
        activate_cmd = "source ./venv/bin/activate && "
    
    # Start a new shell with the virtual environment activated and run the app
    cmd = f"{activate_cmd}python run.py"
    
    try:
        if sys.platform == "win32":
            # On Windows, we need to use cmd.exe to handle the activation
            subprocess.run(["cmd", "/c", cmd], check=True)
        else:
            # On Unix, we can use the shell directly
            subprocess.run(["bash", "-c", cmd], check=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start application: {e}[/red]")
        return False

def show_menu():
    """Show the post-setup menu."""
    console.print(Panel.fit(
        "[bold green]🎉 Setup Complete![/bold green]\n\n"
        "[white]What would you like to do?[/white]\n\n"
        "[cyan]1.[/cyan] Start Lemon-Aid now\n"
        "[cyan]2.[/cyan] Exit",
        title="[bold yellow]🍋 Lemon-Aid[/bold yellow]",
        border_style="green",
        box=box.ROUNDED
    ))
    
    choice = Prompt.ask(
        "Choose an option",
        choices=["1", "2"],
        default="1"
    )
    
    if choice == "1":
        console.print("\n[green]Starting Lemon-Aid...[/green]")
        if start_app():
            return True
        else:
            console.print(Panel(
                "[yellow]To start Lemon-Aid manually:[/yellow]\n"
                f"[cyan]1. {'    .\\venv\\Scripts\\activate' if sys.platform == 'win32' else '    source venv/bin/activate'}[/cyan]\n"
                "[cyan]2. python run.py[/cyan]",
                title="[bold red]Manual Start Required[/bold red]",
                border_style="yellow"
            ))
    else:
        console.print("\n[green]Setup complete! Run Lemon-Aid later with:[/green]")
        console.print(Panel(
            f"[cyan]1. {'    .\\venv\\Scripts\\activate' if sys.platform == 'win32' else '    source venv/bin/activate'}[/cyan]\n"
            "[cyan]2. python run.py[/cyan]",
            title="[bold yellow]Quick Start[/bold yellow]",
            border_style="green"
        ))

def main():
    # Show welcome message
    console.print(Panel.fit(
        "[yellow]Welcome to[/yellow] [bold green]Lemon-Aid[/bold green] [yellow]Setup![/yellow]",
        border_style="green"
    ))
    
    # Create virtual environment
    console.print("\n[bold cyan]Step 1:[/bold cyan] Creating virtual environment")
    if not run_command(
        "python -m venv venv",
        "Creating virtual environment...",
    ):
        console.print("[red]❌ Failed to create virtual environment[/red]")
        return

    # Activate the virtual environment
    console.print("\n[bold cyan]Step 2:[/bold cyan] Activating environment")
    if not activate_venv():
        console.print("[red]❌ Failed to activate virtual environment[/red]")
        return

    # Install dependencies
    console.print("\n[bold cyan]Step 3:[/bold cyan] Installing dependencies")
    if not run_command(
        "python -m pip install --upgrade pip",
        "Upgrading pip...",
    ):
        console.print("[red]❌ Failed to upgrade pip[/red]")
        return

    if not run_command(
        "pip install -r requirements.txt",
        "Installing packages...",
        show_output=True
    ):
        console.print("[red]❌ Failed to install dependencies[/red]")
        return

    # Show menu and handle user choice
    show_menu()

if __name__ == "__main__":
    main() 