```python
# src/commands/ignore.py

import os
import re
from typing import List, Optional
from pathlib import Path

class Ignore:
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.ignore_file = os.path.join(repo_root, '.gitignore')
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> List[str]:
        """Load patterns from .gitignore file"""
        patterns = []
        if os.path.exists(self.ignore_file):
            with open(self.ignore_file, 'r') as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return patterns

    def _match_pattern(self, pattern: str, path: str) -> bool:
        """Check if path matches the given pattern"""
        if pattern.startswith('/'):
            pattern = pattern[1:]

        if pattern.endswith('/'):
            pattern = pattern[:-1]

        if '/' in pattern:
            return re.match(pattern.replace('/', os.sep), path)
        else:
            return re.match(pattern, os.path.basename(path))

    def is_ignored(self, path: str) -> bool:
        """Check if a path should be ignored"""
        relative_path = os.path.relpath(path, self.repo_root)
        for pattern in self.patterns:
            if self._match_pattern(pattern, relative_path):
                return True
        return False

    def add_pattern(self, pattern: str) -> None:
        """Add a new pattern to .gitignore"""
        if pattern not in self.patterns:
            self.patterns.append(pattern)
            with open(self.ignore_file, 'a') as f:
                f.write(pattern + '\n')

    def remove_pattern(self, pattern: str) -> None:
        """Remove a pattern from .gitignore"""
        if pattern in self.patterns:
            self.patterns.remove(pattern)
            with open(self.ignore_file, 'w') as f:
                for p in self.patterns:
                    f.write(p + '\n')

    def list_patterns(self) -> List[str]:
        """List all patterns in .gitignore"""
        return self.patterns.copy()

    def execute(self, args: List[str]) -> int:
        """Execute ignore command"""
        if not args:
            print("Usage: mygit ignore [add|remove|list] [pattern]")
            return 1

        command = args[0]
        if command == 'add':
            if len(args) < 2:
                print("Error: pattern required for add command")
                return 1
            self.add_pattern(args[1])
        elif command == 'remove':
            if len(args) < 2:
                print("Error: pattern required for remove command")
                return 1
            self.remove_pattern(args[1])
        elif command == 'list':
            for pattern in self.list_patterns():
                print(pattern)
        else:
            print(f"Error: unknown command '{command}'")
            return 1

        return 0
```