"""Setup script for Lemon-Aid using UV."""
import os
import subprocess
import sys
from pathlib import Path

def run_command(command: str) -> None:
    """Run a command and print output."""
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def setup_environment():
    """Set up the development environment using UV."""
    # Get project root
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "venv"

    try:
        # Install UV if not already installed
        print("Checking for UV installation...")
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Installing UV...")
            run_command("pip install uv")

        # Create virtual environment using UV
        print("Creating virtual environment...")
        run_command(f"uv venv {venv_path}")

        # Install dependencies using UV pip
        print("Installing dependencies...")
        if sys.platform == "win32":
            pip_cmd = f"{venv_path}\\Scripts\\uv.exe"
        else:
            pip_cmd = f"{venv_path}/bin/uv"
            
        run_command(f"{pip_cmd} pip install --upgrade pip")  # Upgrade pip first
        run_command(f"{pip_cmd} pip install -r {project_root}/requirements.txt")
        
        # Install dev dependencies
        print("Installing development dependencies...")
        run_command(f"{pip_cmd} pip install -r {project_root}/requirements.txt[dev]")

        print("\nSetup complete! To activate the environment:")
        print(f"Windows: {venv_path}\\Scripts\\activate")
        print(f"Unix/Mac: source {venv_path}/bin/activate")

    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment() 