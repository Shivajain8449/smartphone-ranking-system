# Contributing to Smartphone Feature Ranking System

Thank you for your interest in contributing! This project is part of **GirlScript Summer of Code (GSSoC)**.

---

## Table of Contents
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Commit Message Format](#commit-message-format)
- [Code Style Guidelines](#code-style-guidelines)
- [GSSoC Labels](#gssoc-labels)

---

## Getting Started

### 1. Fork and Clone
```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/smartphone-ranking-system.git
cd smartphone-ranking-system
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project
```bash
python smartphone_ranker.py
```

---

## How to Contribute

1. Browse [open issues](https://github.com/Shivajain8449/smartphone-ranking-system/issues)
2. Comment on the issue you want to work on — wait for it to be assigned to you
3. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and commit (see format below)
5. Push and open a Pull Request against the `main` branch
6. Link the issue number in your PR description (e.g., `Fixes #12`)

> **Note**: Do not start working before the issue is assigned to you.

---

## Commit Message Format

Use this format for all commits:

```
type: short description (#issue-number)
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

**Examples**:
```
feat: add argparse CLI support (#5)
fix: correct PROCESSOR column values in sample data (#8)
docs: update README installation steps (#3)
test: add unit tests for normalize_data method (#11)
```

---

## Code Style Guidelines

- Follow **PEP 8** style guide
- Add **docstrings** to every function and class
- Keep functions focused — one responsibility per function
- Use **meaningful variable names**
- Run `pylint smartphone_ranker.py` before submitting

---

## GSSoC Labels

| Label | Description |
|-------|-------------|
| `gssoc` | Issues available for GSSoC contributors |
| `good-first-issue` | Great for first-time contributors |
| `level1` | Easy — documentation, dataset additions |
| `level2` | Medium — new features, refactoring |
| `level3` | Hard — web UI, API, advanced ML features |
| `gssoc-merged` | Applied when a GSSoC PR is merged |

---

## Questions?

Open a [GitHub Discussion](https://github.com/Shivajain8449/smartphone-ranking-system/discussions) or reach out via email: shivajain299@gmail.com