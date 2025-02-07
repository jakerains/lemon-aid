#!/usr/bin/env python3
"""
Lemon-Aid Launcher Script
Provides the main entry point for the Lemon-Aid application.
"""

import os
import sys
import asyncio
from pathlib import Path

async def async_main():
    # Add the src directory to Python path
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    sys.path.insert(0, src_dir)
    
    try:
        # Import and run the main application
        from lemonaid import main as run_lemonaid
        await run_lemonaid()
    except ImportError as e:
        print(f"\033[91mError: Could not import Lemon-Aid. Make sure you're in the correct directory and the virtual environment is activated.\033[0m")
        print(f"Details: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\033[93mLemon-Aid shutdown requested. Goodbye! 🍋\033[0m")
        sys.exit(0)
    except Exception as e:
        print(f"\033[91mError running Lemon-Aid: {str(e)}\033[0m")
        sys.exit(1)

def main():
    """Entry point that runs the async main function."""
    if sys.platform == "win32":
        # Set the event loop policy for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\033[93mLemon-Aid shutdown requested. Goodbye! 🍋\033[0m")
        sys.exit(0)

if __name__ == "__main__":
    main() 