#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lemon-Aid Launcher Script
This script launches the Lemon-Aid training data generation tool from the root directory.
"""

import os
import sys

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

from lemonaid import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main()) 