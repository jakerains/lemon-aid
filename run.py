"""
Lemon-Aid Launcher Script
This script launches the Lemon-Aid training data generation tool from the root directory.
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Import and run the main script
from lemonaid import main
import asyncio

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("\nProcess completed.")
        sys.exit(0) 