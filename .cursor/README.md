# Cursor IDE Configuration

This directory contains Cursor IDE specific configuration and rules.

## Rules Directory

The `rules/` directory contains `.mdc` (Markdown with metadata) files that define AI assistant behavior for this project.

### Available Rules

1. **`no_test_files.mdc`** - Prevents automatic test file generation/modification
2. **`markdown_in_docs.mdc`** - Enforces documentation placement in `docs/` folder
3. **`dependencies.mdc`** - Mandates use of `pyproject.toml` for dependency management
4. **`project_structure.mdc`** - Defines proper file organization
5. **`python_style.mdc`** - Python code style conventions
6. **`qlora_training.mdc`** - QLoRA training specific guidelines

### Rule File Format

Each `.mdc` file has YAML frontmatter followed by markdown content:

```mdc
---
description: Brief description of the rule
globs: 
  - '**/*.py'
  - '**/*.md'
alwaysApply: true
---

# Rule Title

Rule content in markdown format...
```

### How Rules Work

- **`description`**: Short summary of what the rule does
- **`globs`**: File patterns this rule applies to
- **`alwaysApply`**: Whether the rule is always active (true) or contextual (false)

### Modifying Rules

To add or modify rules:
1. Create/edit `.mdc` files in the `rules/` directory
2. Follow the established format with frontmatter + markdown
3. Use clear, actionable language
4. Test the rules to ensure they work as expected

### Override Behavior

User explicit instructions always override these rules. Rules are guidelines to improve AI assistance, not restrictions.

