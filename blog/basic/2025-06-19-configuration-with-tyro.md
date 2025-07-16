---
title: Basic Series 5 - Configuration with Tyro
description: A guide to using Tyro for configuration.
slug: basic-series-5
tags: [basic-series]
---

:::warning
**Code Style Note**: The demonstration code examples in this post are intentionally compact for readability. In production code, you should follow PEP 8 style guidelines with proper spacing, line breaks, and formatting.
:::

## Why Use Configuration Tools Anyway?

* **Single source of truth**: No more copy-pasting option definitions across your code, CLI parser, and docs.
* **Less boilerplate**: Automatically handle parsing, defaults, validation, and help text without endless manual work.

## Why Tyro?

* **Pure Python**: Just use type hints and dataclasses — no extra weird config files or DSLs to learn.
* **One-liner CLI**: `tyro.cli(...)` magically figures out flags, defaults, help messages, and even subcommands for you.
* **Better developer experience**: You get IDE autocomplete, type checking, and neat, hierarchical interfaces. (Honestly, this is my favorite part!)

<!-- truncate -->

## How to Use Tyro

The magic happens with `tyro.cli()`, which can take either a **function** or a **dataclass**.

### Option 1: Function (Quick and Simple)

Great for smaller scripts! Tyro will generate CLI flags from your function’s parameters and then call your function directly.

```python
import tyro

def main(input: str, verbose: bool = False) -> None:
    print(f"Input: {input}, verbose={verbose}")

if __name__ == "__main__":
    # highlight-next-line
    tyro.cli(main)
```

When you run:

```bash
python script.py --help
```

You'll see something like:

```bash
usage: script.py [-h] --input STR [--verbose | --no-verbose]

╭─ options ───────────────────────────────╮
│ -h, --help       show help message      │
│ --input STR      (required)             │
│ --verbose, --no-verbose (default: False)│
╰─────────────────────────────────────────╯
```

### Option 2: Dataclass (More Structured)

Perfect if your project has more settings and you want things tidy.

```python
from dataclasses import dataclass
import tyro

# `dataclass` is needed here
# highlight-next-line
@dataclass
class Config:
    input_path: str
    batch_size: int = 32

if __name__ == "__main__":
    # highlight-next-line
    cfg = tyro.cli(Config)
    print(f"Running with config: {cfg}")
```

Now Tyro will parse `--input-path` and `--batch-size` flags, then give you a `Config` instance ready to use.

## Using Default Configs & Overriding Them

### Overriding Defaults in Dataclasses

You can pre-define a dataclass with certain default values and make Tyro only require what’s missing. Use the `default=` argument:

```python
from dataclasses import dataclass
import tyro

@dataclass
class Args:
    string: str
    reps: int = 3

if __name__ == "__main__":
    args = tyro.cli(
        Args,
        default=Args(string="hello", reps=tyro.MISSING),
    )
    print(args)
```

* Here, `--string` defaults to `"hello"`.
* `reps` is marked as required (`MISSING`), so you must specify it via `--reps`.

## Hierarchical Dataclasses

Want to manage more complex configs? You can nest dataclasses and keep things clean and organized.

```python
from dataclasses import dataclass
import tyro

@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

@dataclass
class Config:
    optimizer: OptimizerConfig
    save_dir: str = "logs"
    seed: int = 0

def train(config: Config) -> None:
    # Your training logic here
    pass

if __name__ == "__main__":
    config = tyro.cli(Config)
    train(config)
    print(config)
```

### Help Message Example

```bash
$ python script.py --help
usage: script.py [-h] [OPTIONS]

╭─ options ─────────────────────────────╮
│ -h, --help        show help           │
│ --save-dir STR    (default: logs)     │
│ --seed INT        (default: 0)        │
╰───────────────────────────────────────╯
╭─ optimizer options ───────────────────╮
│ --optimizer.learning-rate FLOAT       │
│                      (default: 0.0003)│
│ --optimizer.weight-decay FLOAT        │
│                      (default: 0.01)  │
╰───────────────────────────────────────╯
```

### Usage Examples

```bash
# Using all defaults
$ python script.py
Config(optimizer=OptimizerConfig(learning_rate=0.0003, weight_decay=0.01), save_dir='logs', seed=0)

# Overwriting some values
$ python script.py --save_dir runs --seed 1234 --optimizer.learning-rate 1e-5
Config(optimizer=OptimizerConfig(learning_rate=1e-05, weight_decay=0.01), save_dir='runs', seed=1234)
```

## Saving Your Config

Saving configs is super helpful if you want to share experiments or reproduce results later. You can save them as JSON or YAML.

```python
from dataclasses import dataclass
import os
import tyro
import yaml

@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

@dataclass
class Config:
    optimizer: OptimizerConfig
    save_dir: str = "logs"
    seed: int = 0

def train(config: Config) -> None:
    # Your training logic here
    pass

if __name__ == "__main__":
    config = tyro.cli(Config)
    train(config)
    # highlight-start
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f)
    # highlight-end
```

## Other Alternatives

If you need something even fancier (like managing tons of nested configs or big experiments), check out [Hydra](https://github.com/facebookresearch/hydra). It’s built by Meta and great for complex projects.

## Conclusion

Tyro is a super handy way to build clean, type-safe, and user-friendly CLIs without drowning in boilerplate code. Whether you're working on a quick experiment or a big research project, it helps you stay organized, reduce repetitive code, and make your scripts easier for others (and future you) to run.

:::tip
If you want even more advanced features, nested configs, or extra tricks, be sure to dive into the [official Tyro docs](https://brentyi.github.io/tyro/) — there’s a lot more to explore.
:::

:::note[Link to other basic series]

1. [Upload to GitHub](blog/basic/2025-06-15-upload-to-github-repo.md)
2. [Python Mangement with uv](blog/basic/2025-06-16-python-environment-with-uv.md)
3. [Code formatting](blog/basic/2025-06-17-python-code-formatting.md)
4. [Tmux is all you need](blog/basic/2025-06-18-tmux-usage.md)
5. [Configuration with tryo](blog/basic/2025-06-19-configuration-with-tyro.md)

:::
