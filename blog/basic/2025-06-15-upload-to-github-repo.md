---
title: Basic Series 1 - Uploading Code to a GitHub Repository
description: A guide to uploading your local code to a private GitHub repository.
slug: basic-series-1
tags: [basic-series]
---

This guide will show you how to push your local code to a private GitHub repository step by step — no stress, no cryptic errors (hopefully).

## Prerequisites

Before we jump in, make sure you have:

* Git installed on your computer
* A GitHub account
* An SSH key set up (recommended) or a Personal Access Token

<!-- truncate -->

## Steps

### 1. Create a Private Repository on GitHub

1. Go to [GitHub](https://github.com)
2. Click the "+" icon in the top right
3. Choose "New repository"
4. Name your repository
5. Pick "Private" if you don’t want the whole world seeing your code
6. Click "Create repository"

> **Public vs Private**
>
> * **Public**: Visible to everyone. Good for open-source or public research code.
> * **Private**: Only you and invited collaborators can see it. Perfect for work-in-progress or sensitive projects.

### 2. Initialize Git in Your Local Project

If you haven't done this before, no worries. Just run:

```bash
# Move into your project folder
cd your-project-directory

# Initialize a Git repository
git init
```

### 3. Add Your Files

It’s better to add files selectively so you don’t accidentally track things you don’t want.
If you do use `git add .`, make sure you set up a `.gitignore` first so you don't upload, say, your secret keys or huge datasets. See [Setting Up .gitignore](#setting-up-gitignore).

```bash
# Add everything (be careful with this!)
git add .

# Or, add specific files
git add file1.py file2.py
```

### 4. Commit Your Changes

Write clear, meaningful commit messages so you know what changed later on.

```bash
git commit -m "Add initial code"
```

:::tip
You can even spice them up with emojis (using [gitmoji](https://gitmoji.dev/)) if you want — just remember to use the text format like `:sparkles:` so nothing breaks.
:::

### 5. Link Your Local Repository to GitHub

Before pushing, you need to tell Git where your repo lives online.

```bash
git remote add origin git@github.com:username/repository-name.git
```

:::info
If you haven’t set up authentication yet, check [Authentication Methods](#authentication-methods).
:::

### 6. Push Your Code

```bash
# Push to main branch
git push origin main

# Push to a different remote branch
git push origin local_branch:remote_branch
```

**Example**: push your local `feature` branch to the remote `dev` branch.

```bash
git push -u origin feature:dev
```

The `-u` flag sets the upstream branch, so future pushes can be simpler. You can check upstream branches with `git branch -vv`.

## Setting Up .gitignore

A `.gitignore` file tells Git which files or folders it should ignore. This keeps your repo clean and avoids accidentally uploading huge files, build artifacts, or sensitive info.

Create one like this:

```bash
touch .gitignore
```

<details>
<summary>Common .gitignore patterns for Python projects (click to expand)</summary>

```plaintext
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
env/
.venv/
venv/

# Distribution / packaging
build/
dist/
*.egg-info/

# Jupyter notebooks
.ipynb_checkpoints

# IDEs and editors
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Environment files
.env
```

</details>

:::tip
You can find more templates at [gitignore.io](https://www.toptal.com/developers/gitignore) or from [GitHub’s gitignore repo](https://github.com/github/gitignore).
There’s also a handy VS Code extension: [gitignore-generator](https://github.com/piotrpalarz/vscode-gitignore-generator).
:::

## Authentication Methods

### Using SSH (Recommended)

1. Generate an SSH key if you don’t have one yet:

   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. Copy your public key:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. Go to GitHub → Settings → SSH and GPG keys → New SSH key, paste it, and save.

### Using Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate a new token with the `repo` scope
3. Use this token instead of your password when pushing code

## Common Issues and Quick Fixes

### Authentication Failed

* Double-check your SSH key setup
* Make sure your Personal Access Token is still valid
* Verify you have access rights to the repo

### Remote Already Exists

If you get an error like "remote origin already exists":

```bash
git remote remove origin
git remote add origin git@github.com:username/repository-name.git
```

## Best Practices

* Use `.gitignore` to keep unnecessary stuff out of your repo
* Write meaningful commit messages so future-you knows what’s going on
* Keep private code private
* Push your changes often so your remote stays in sync

## Conclusion

That’s it! You’re ready to show off your code (or keep it safely hidden in a private repo). No more "wait, where did my changes go?" moments — just smooth sailing from here.

:::note[Link to other basic series]

1. [Upload to GitHub](blog/basic/2025-06-15-upload-to-github-repo.md)
2. [Python Mangement with uv](blog/basic/2025-06-16-python-environment-with-uv.md)
3. [Code formatting](blog/basic/2025-06-17-python-code-formatting.md)
4. [Tmux is all you need](blog/basic/2025-06-18-tmux-usage.md)
5. [Configuration with tryo](blog/basic/2025-06-19-configuration-with-tyro.md)

:::
