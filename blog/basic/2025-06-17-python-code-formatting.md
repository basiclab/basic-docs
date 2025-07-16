---
title: Basic Series 3 - Python Code Formatting with Black and isort
description: A guide to using Black and isort to automatically format your Python code.
slug: basic-series-3
tags: [basic-series]
---

This guide shows you how to automatically format your Python code using **Black** and **isort**, so your project stays clean, consistent, and easy to read (even at 3 a.m. after too much coffee).

## What Are Black and isort?

* **Black**: An opinionated Python code formatter that auto-formats your code to follow PEP 8 style (so you can stop arguing about whitespace with your teammates).
* **isort**: Automatically sorts and organizes your imports into neat, logical sections.

> ⚠️ **Heads up**: Black and isort can sometimes fight over formatting. To avoid this, always configure isort to use Black’s style.

## Installing Black and isort

```bash
# Install both tools
pip install black isort

# Or add them as dev dependencies with uv
uv add black isort --dev
```

## How to Use Black and isort

### Using Black

```bash
# Format a single file
black your_file.py

# Format an entire directory
black your_directory/

# Format specific files
black file1.py file2.py

# Check what would be changed (without modifying)
black --check your_file.py

# Format all Python files in the current directory
black .
```

### Using isort

```bash
# Sort imports in a single file
isort your_file.py

# Sort imports in an entire directory
isort your_directory/

# Sort imports in specific files
isort file1.py file2.py

# Check for changes without actually changing files
isort --check-only your_file.py
```

## Configuring Black and isort

### Black

Create a `pyproject.toml` in your project root (if you’re using `uv`, it may already exist):

```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

### isort

Add this to the same `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
```

> Setting `profile = "black"` makes sure isort and Black play nicely together.

## IDE Integration

### VS Code

1. Install the [Black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) and [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) extensions.
2. Add this to your `settings.json`:

```json
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        },
        "editor.defaultFormatter": "black"
    }
}
```

### PyCharm

1. Install the BlackConnect plugin.
2. Install the isort plugin.
3. Configure both under Settings → Tools.

## Before and After Example

### Before

```python
from datetime import datetime
import os
import sys
from typing import List,Dict
def my_function(  x: int,y: int  )->int:
    return x+y
```

### After Black and isort

```python
import os
import sys
from datetime import datetime
from typing import Dict, List


def my_function(x: int, y: int) -> int:
    return x + y
```

## Best Practices

1. **Use the same config across your team**: Avoid “it works on my machine” chaos.
2. **Set up pre-commit hooks**: Auto-format code before you even push.
3. **Add checks to CI/CD**: Stop bad formatting from sneaking in.
4. **Document your rules**: Put them in the README so everyone stays on the same page.

## Troubleshooting

### Common Issues

1. **Black and isort fighting?**

   * Use `profile = "black"` in isort.
   * Run isort first, then Black.

2. **Line length conflicts?**

   * Set the same `line-length` in both tools.

3. **Imports not sorting right?**

   * Use `--profile black` with isort.
   * Double-check your config sections.

## Advanced Usage with Ruff

Ruff is a lightning-fast linter and formatter (written in Rust) that can replace Black, isort, and other tools — all in one go.

### Why Use Ruff?

1. **Super fast**: Like, really fast.
2. **All-in-one**: Formatting and linting together.
3. **Compatible with Black style**: No fighting.
4. **Tons of rules**: Supports over 800 lint rules.
5. **Modern and reliable**: Built for speed and correctness.

### Installing Ruff

```bash
# Install with pip
pip install ruff

# Or with uv
uv add ruff --dev
```

### Using Ruff

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Auto-fix issues
ruff check --fix .

# Fix and format in one go
ruff check --fix . && ruff format .
```

### Ruff Configuration

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
select = ["ALL"]
ignore = ["E501", "D203", "D212"]
target-version = "py38"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

### VS Code Integration

1. Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
2. Add to your `settings.json`:

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

### Before and After with Ruff

* **Before**

```python
from datetime import datetime
import os
import sys
from typing import List,Dict
def my_function(  x: int,y: int  )->int:
    return x+y
```

* **After**

```python
import os
import sys
from datetime import datetime
from typing import Dict, List


def my_function(x: int, y: int) -> int:
    return x + y
```

## Conclusion

Whether you stick with Black and isort or switch to Ruff for an all-in-one approach, keeping your code consistently formatted will make your life (and your teammates’ lives) so much easier. No more nitpicking over spaces or import orders — just clean, readable code every time.

Now go make your code beautiful and keep those diffs clean!

:::note[Link to other basic series]

1. [Upload to GitHub](blog/basic/2025-06-15-upload-to-github-repo.md)
2. [Python Mangement with uv](blog/basic/2025-06-16-python-environment-with-uv.md)
3. [Code formatting](blog/basic/2025-06-17-python-code-formatting.md)
4. [Tmux is all you need](blog/basic/2025-06-18-tmux-usage.md)
5. [Configuration with tryo](blog/basic/2025-06-19-configuration-with-tyro.md)

:::
