---
title: Basic Series 2 - Using uv for Python Package Management
description: A guide to using uv for Python package management.
slug: basic-series-2
tags: [basic-series]
---

:::note
This post may be updated as better alternatives to `uv` emerge.
:::

`uv` is a super-fast Python package installer and resolver, built in **Rust**. Think of it as a modern, turbocharged alternative to `pip` and other Python package managers.

## Key Features

* **Speed**: `uv` is blazing fast compared to traditional tools.
* **Automatic Python Management**: It can handle downloading and managing Python versions for you.
* **Compatibility**: Works nicely with existing Python installs.
* **Modern CLI**: Clean, intuitive commands that are easy to remember.

<!-- truncate -->

## Installation

Want to try it out? Just run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Basic Usage

### Installing Python Versions

Let `uv` manage your Python versions so you don’t have to.

* Install the latest Python version:

```bash
uv python install
```

* Install a specific version:

```bash
uv python install 3.12
```

* Install multiple versions at once:

```bash
uv python install 3.11 3.12
```

### Managing Packages

* Initialize a new project:

```bash
uv init project_name

# or in the current directory
uv init

# specify a Python version (recommended)
uv init --python 3.12
```

* Add a package to your project:

```bash
uv add package_name

# or specify a version
uv add 'package_name==1.2.3'
```

* Add dependencies from `requirements.txt`:

```bash
uv add -r requirements.txt
```

* Remove a package:

```bash
uv remove package_name
```

* Upgrade a specific package:

```bash
uv lock --upgrade-package package_name
```

### Project Structure

A typical `uv` project might look like this:

```plaintext
.
├── .venv/              # Virtual environment
├── .python-version     # Python version for this project
├── README.md
├── main.py
├── pyproject.toml      # Project metadata & dependencies
└── uv.lock             # Lockfile for reproducible installs
```

<details>
<summary>📄 Project Files Explained (click to expand)</summary>

#### pyproject.toml

This file holds your project’s metadata and dependencies. It’s where you define project name, version, and dependency list. Example:

```toml
[project]
name = "your-project"
version = "0.1.0"
description = "Your project description"
dependencies = []
```

#### .python-version

Specifies which Python version to use for the project. This is auto-generated when you initialize with a specific Python version.

#### .venv

Contains your isolated virtual environment. All project dependencies get installed here. It’s created automatically when you run commands like `uv run` or `uv sync`.

#### uv.lock

This file captures exactly which versions of packages are installed, ensuring consistent setups across different machines. Commit this file to version control — no need to edit it manually.

</details>

### Running Scripts

* Run directly using `uv`:

```bash
uv run script.py
```

or

* Activate your virtual environment first:

```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

python main.py
```

## Using PyTorch with uv

PyTorch has some special packaging quirks since its wheels live on separate indices and vary based on your accelerator (CPU, CUDA, etc.).

### Basic PyTorch Installation

Here’s an example `pyproject.toml` setup for CPU-only builds:

```toml
[project]
name = "your-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.7.0",
    "torchvision>=0.22.0",
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu" },
]
torchvision = [
    { index = "pytorch-cpu" },
]
```

> **Note**
> You can find the correct PyTorch index URLs [here](https://pytorch.org/get-started/locally/).

### Using CUDA Builds

For CUDA support, use the right index URL for your CUDA version. Example for CUDA 12.1:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

### Quick Install with uv pip

Prefer something simpler? You can also use `uv pip` directly:

```bash
# CPU-only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Additional Resources

Want to dig deeper? Check out the official docs at [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).

## Conclusion

`uv` makes managing Python projects feel way less painful — faster installs, automatic Python version management, clean project structures, and smooth handling of PyTorch (even with tricky CUDA builds).

Whether you're working on small personal scripts or big research projects, uv can help you keep things tidy and reproducible without all the typical package management headaches.

If you’re curious, definitely check out the official docs and give it a spin. Who knows, it might just become your new favorite tool!

:::note[Link to other basic series]

1. [Upload to GitHub](blog/basic/2025-06-15-upload-to-github-repo.md)
2. [Python Mangement with uv](blog/basic/2025-06-16-python-environment-with-uv.md)
3. [Code formatting](blog/basic/2025-06-17-python-code-formatting.md)
4. [Tmux is all you need](blog/basic/2025-06-18-tmux-usage.md)
5. [Configuration with tryo](blog/basic/2025-06-19-configuration-with-tyro.md)

:::
