---
title: Basic Series 4 - Mastering tmux for Efficient Terminal Management
description: A guide to using tmux for efficient terminal management.
slug: basic-series-4
tags: [basic-series]
---

This guide explains how to use tmux (Terminal Multiplexer) to enhance your terminal productivity and manage multiple terminal sessions effectively.

## What is tmux?

tmux is a terminal multiplexer that allows you to:

- Create multiple terminal sessions within a single window
- Detach from sessions without closing them
- Reattach to sessions from different terminals
- Split your terminal into multiple panes
- Share terminal sessions with other users

<!-- truncate -->

## Installation

> [!NOTE]
> If you are using our servers, tmux is already installed.

```bash
# On Ubuntu/Debian
sudo apt-get install tmux

# On macOS with Homebrew
brew install tmux

# On Windows (via WSL)
sudo apt-get install tmux
```

## Basic Usage

### Starting tmux

```bash
# Start a new session
tmux

# Start a new named session
tmux new -s mysession

# List existing sessions
tmux ls
```

### Essential Keyboard Shortcuts

All tmux commands start with a prefix key (default: <kbd>Ctrl</kbd>+<kbd>b</kbd>). Here are the most common commands:

#### Session Management

- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>d</kbd> - Detach from current session
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>$</kbd> - Rename current session
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>s</kbd> - Switch between sessions

#### Window Management

- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>c</kbd> - Create new window
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>n</kbd> - Next window
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>p</kbd> - Previous window
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>&</kbd> - Kill current window
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>,</kbd> - Rename current window

#### Pane Management

- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>%</kbd> - Split pane vertically
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>"</kbd> - Split pane horizontally
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>arrow</kbd> - Move between panes
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>z</kbd> - Zoom in/out of current pane
- <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>x</kbd> - Kill current pane

## Configuration

Create or edit `~/.tmux.conf`:

```bash
# Enable mouse mode
set -g mouse on

# Start window numbering at 1
set -g base-index 1

# Start pane numbering at 1
setw -g pane-base-index 1

# Increase scrollback buffer size
set -g history-limit 50000

# Use vim keybindings in copy mode
setw -g mode-keys vi

# Update status bar every 15 seconds
set -g status-interval 15

# Enable focus events
set -g focus-events on

# Set terminal title
set -g set-titles on
set -g set-titles-string "#T"

# Status bar customization
set -g status-style bg=black,fg=white
set -g status-left "#[fg=green]#H #[fg=black]• #[fg=green]#(uname -r | cut -c 1-6)#[default]"
set -g status-right "#[fg=green]#(cut -d ' ' -f 1-3 /proc/loadavg)#[default]"
```

## Common Use Cases

### 1. Remote Development

```bash
# Start a new session on remote server
ssh user@server
tmux new -s dev

# do somthing here ...

# Detach from session
Ctrl+b d

# Reattach to session
tmux attach -t dev
```

### 2. Multiple Terminal Windows

```bash
# Create new window
Ctrl+b c

# Switch between windows
Ctrl+b n  # Next window
Ctrl+b p  # Previous window
```

### 3. Split Screen Work

```bash
# Split screen vertically
Ctrl+b %

# Split screen horizontally
Ctrl+b "

# Move between panes
Ctrl+b arrow keys
```

## Best Practices

1. **Session Naming**: Always use descriptive names for your sessions
2. **Window Organization**: Use windows for different projects or tasks
3. **Pane Layout**: Use panes for related tasks within the same project
4. **Configuration**: Customize your `.tmux.conf` to match your workflow
5. **Session Persistence**: Use `tmux-resurrect` or `tmux-continuum` for session persistence

## Advanced Features (Optional)

### Session Persistence

Install tmux-resurrect:

```bash
# Install tmux plugin manager
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# Add to .tmux.conf
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-resurrect'

# Install plugins
Ctrl+b I
```

### Copy Mode

1. Enter copy mode: <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>[</kbd>
2. Use vim keys to navigate
3. Space to start selection
4. Enter to copy
5. <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>]</kbd> to paste

### Synchronize Panes

```bash
# Toggle synchronize-panes
Ctrl+b :setw synchronize-panes on
Ctrl+b :setw synchronize-panes off
```

## Troubleshooting

### Common Issues

1. **Mouse Mode Not Working**:
   - Ensure `set -g mouse on` is in your `.tmux.conf`
   - Restart tmux after configuration changes

2. **Copy/Paste Issues**:
   - Use <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>[</kbd> to enter copy mode
   - Use vim keys to navigate
   - Space to start selection
   - Enter to copy
   - <kbd>Ctrl</kbd>+<kbd>b</kbd> then <kbd>]</kbd> to paste

3. **Session Recovery**:
   - Use `tmux attach` to reattach to existing sessions
   - Use `tmux ls` to list available sessions

## Additional Resources

- [Official tmux Documentation](https://github.com/tmux/tmux/wiki)
- [tmux Cheat Sheet](https://tmuxcheatsheet.com/)
- [tmux Plugin Manager](https://github.com/tmux-plugins/tpm)
- [tmux-resurrect](https://github.com/tmux-plugins/tmux-resurrect)

:::note[Link to other basic series]

1. [Upload to GitHub](blog/basic/2025-06-15-upload-to-github-repo.md)
2. [Python Mangement with uv](blog/basic/2025-06-16-python-environment-with-uv.md)
3. [Code formatting](blog/basic/2025-06-17-python-code-formatting.md)
4. [Tmux is all you need](blog/basic/2025-06-18-tmux-usage.md)
5. [Configuration with tryo](blog/basic/2025-06-19-configuration-with-tyro.md)

:::
