# Contributing to Causal Bayesian Networks

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add: description of your changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a pull request

## Code Standards

### Python Style Guide

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes (Google style)
- Keep functions focused and modular
- Maximum line length: 100 characters

### Example Docstring

```python
def compute_treatment_effect(treatment: str, outcome: str) -> float:
    """
    Compute the average treatment effect.

    Args:
        treatment: Name of the treatment variable
        outcome: Name of the outcome variable

    Returns:
        Average treatment effect as a float

    Raises:
        ValueError: If variables are not in the network
    """
    pass
```

### Testing

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Place tests in `tests/` directory
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`

### Documentation

- Update README.md if adding major features
- Add docstrings to all public APIs
- Update theory.md for new theoretical concepts
- Add examples in `examples/` for new use cases

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of changes in the PR
4. Link any related issues
5. Request review from maintainers

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: <description>` - New features
- `Fix: <description>` - Bug fixes
- `Update: <description>` - Updates to existing features
- `Docs: <description>` - Documentation changes
- `Test: <description>` - Test additions/changes
- `Refactor: <description>` - Code refactoring

## Areas for Contribution

### High Priority

- Structure learning algorithms
- Additional inference methods
- Performance optimization
- More real-world examples
- Better visualization tools

### Good First Issues

- Documentation improvements
- Adding type hints
- Writing tests
- Fixing typos
- Adding examples

### Feature Requests

Before working on major features:
1. Open an issue to discuss the idea
2. Get feedback from maintainers
3. Outline the implementation approach
4. Start development after approval

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the project

## Questions?

Open an issue with the "question" label or reach out to maintainers.

Thank you for contributing!
