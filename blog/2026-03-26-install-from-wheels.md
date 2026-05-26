---
title: Installing Packages from Wheels
description: Understanding why some packages are a nightmare to install and how to sidestep the pain with prebuilt wheels.
slug: install-from-wheels
tags: [package-management]
---

You type `pip install flash-attn`. You hit Enter. You wait. And wait. Your fan spins up like a jet engine. 10 minutes later: **compilation error**. Congratulations, you've experienced one of the most classic rites of passage in the ML world.

This guide explains what's going on, why some packages are a nightmare to install, and how to sidestep the pain with prebuilt wheels.

<!-- truncate -->

---

## What Even Is a Wheel?

A Python **wheel** (`.whl` file) is a prebuilt binary distribution. Think of it like a zip file that contains already-compiled code, ready to be dropped into your Python environment.

When you do a normal `pip install some-package`, pip first checks if a wheel exists for your platform. If it does — great, it's fast. If it doesn't, pip falls back to downloading the source and **compiling it on your machine**. That's where things get messy.

```
some_package-1.0.0-cp310-cp310-linux_x86_64.whl
               │         │         │
             Python    ABI tag   Platform
             version
```

The filename encodes exactly what it was built for. `cp310` means CPython 3.10. `linux_x86_64` means 64-bit Linux. If your environment doesn't match, pip won't even try to install it.

---

## Issue Behind the Scene

`flash-attn`, `xformers`, `bitsandbytes`, `apex` — these packages share a common trait: they have **CUDA kernels** baked in. That means they need to be compiled against:

1. A specific **CUDA version** (e.g., 11.8, 12.1, 12.4)
2. A specific **PyTorch version** (e.g., 2.1.0, 2.3.0)
3. Your **Python version**

When you `pip install flash-attn` from source, your machine has to compile thousands of lines of CUDA code. This takes **15–40 minutes**, requires `nvcc` (the NVIDIA CUDA compiler) to be installed and on your PATH, and will fail spectacularly if any version is mismatched.

The error usually looks like one of these:

```
# The "I don't even have a compiler" error
error: command 'gcc' failed: No such file or directory

# The "CUDA version mismatch" error  
RuntimeError: CUDA error: no kernel image is available for execution on the device

# The "nvcc not found" classic
nvcc: command not found

# The cryptic one that sends you to Stack Overflow at 2am
ninja: build stopped: subcommand failed.
```

---

## The Smart Way: Install from a Prebuilt Wheel

Most popular CUDA packages maintain a repo of prebuilt wheels for common CUDA + PyTorch + Python combinations. Instead of compiling, you download the exact binary you need.

### Step 1: Know Your Environment

Before hunting for a wheel, figure out exactly what you're working with:

```bash
# Python version
python --version

# PyTorch version + CUDA it was built with
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# CUDA toolkit version on your system
nvcc --version
# or, if nvcc isn't installed:
nvidia-smi  # shows driver's max supported CUDA version
```

Example output you might see:

```
Python 3.10.12
2.3.0+cu121
12.1
```

So you need: **Python 3.10**, **PyTorch 2.3.0**, **CUDA 12.1**.

### Step 2: Find the Right Wheel

**For `flash-attn`**, the prebuilt wheels live on GitHub [releases](https://github.com/Dao-AILab/flash-attention/releases).

Look for a filename that matches your setup. For the example above, you'd grab something like:

```
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

### Step 3: Install It

Once you have the URL or have downloaded the file:

```bash
# Install directly from URL (no download needed)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# or with uv
uv add https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Or install from a local file you downloaded
pip install flash_attn-2.6.3+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# or with uv
uv add flash_attn-2.6.3+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Done. No compiler needed. No 30-minute wait. Just a clean, fast install.
