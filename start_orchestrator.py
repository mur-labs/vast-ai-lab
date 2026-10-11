#!/usr/bin/env python3
"""Convenience top-level launcher for the Rental LLM Orchestrator."""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from orchestrator.main import main

if __name__ == "__main__":
    main()
