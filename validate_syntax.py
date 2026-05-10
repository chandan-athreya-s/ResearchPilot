#!/usr/bin/env python3
"""
Quick syntax validation for new agent modules.
Tests that all files can be imported without errors.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

print("="*60)
print("SYNTAX VALIDATION FOR AGENT MODULES")
print("="*60)

modules_to_test = [
    ("Query Agent", "app.agents.query_agent"),
    ("Retrieval Agent", "app.agents.retrieval_agent"),
    ("Citation Agent", "app.agents.citation_agent"),
    ("Orchestrator", "app.agents.orchestrator"),
]

passed = 0
failed = 0

for name, module_path in modules_to_test:
    try:
        __import__(module_path)
        print(f"✓ {name:30} — Successfully imported")
        passed += 1
    except Exception as e:
        print(f"✗ {name:30} — Import failed: {str(e)}")
        failed += 1

print("="*60)
print(f"Result: {passed}/{len(modules_to_test)} modules valid")
print("="*60)

if failed == 0:
    print("✓ All modules have valid syntax!")
    sys.exit(0)
else:
    print(f"✗ {failed} module(s) have issues")
    sys.exit(1)
