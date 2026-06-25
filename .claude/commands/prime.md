---
description: Prime agent with codebase understanding
---

# Prime: Load Project Context

## Objective

Build comprehensive understanding of the codebase by analyzing structure, documentation, and key files.

## Process

### 1. Analyze Project Structure

Show directory structure (if tree is available):
On Linux, run: `tree -L 4 -I 'node_modules|__pycache__|.git|dist|build|*.pyc'`

If tree is not available, list files recursively with: `find . -not -path '*/.git/*' -not -path '*/node_modules/*' -not -name '*.pyc' | sort`

### 2. Read Core Documentation

- Read PRD files in `Orchestration/PRDs/` (list and read all `.md` files there)
- Read `CLAUDE.md` at project root if it exists
- Read README files at project root and major directories
- Read any architecture documentation
- Read plan files in `Orchestration/Plans/` for current implementation intent
- Read reference docs in `Orchestration/References/` for project-specific rules and context

### 3. Identify Key Files

Based on the structure, identify and read:
- Main entry points (main.py, index.ts, app.py, etc.)
- Core configuration files (pyproject.toml, package.json, tsconfig.json)
- Key model/schema definitions
- Important service or controller files

### 4. Understand Current State

If this is a git repository, check recent activity and current status:
!`git log -10 --oneline`
!`git status`

If not a git repository, note that version control is not initialized.

## Output Report

Provide a concise summary covering:

### Project Overview
- Purpose and type of application
- Primary technologies and frameworks
- Current version/state

### Architecture
- Overall structure and organization
- Key architectural patterns identified
- Important directories and their purposes

### Tech Stack
- Languages and versions
- Frameworks and major libraries
- Build tools and package managers
- Testing frameworks

### Core Principles
- Code style and conventions observed
- Documentation standards
- Testing approach

### Current State
- Active branch (if git repo)
- Recent changes or development focus
- Plans in progress (`Orchestration/Plans/`)
- Any immediate observations or concerns

**Make this summary easy to scan - use bullet points and clear headers.**
