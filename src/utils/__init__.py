"""
Utility functions and helper classes.

This module contains common utilities used across the project.
"""

# Import helpers only when needed to avoid circular imports

__all__ = [
    "setup_notebook_path"
]

def setup_notebook_path():
    """Setup notebook path - simplified version."""
    import sys
    from pathlib import Path
    
    # Find project root
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / 'src').exists():
            src_path = current_path / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            return str(src_path)
        current_path = current_path.parent
    
    return None
