"""Versioned prompt templates.

Prompt text lives in `<name>/<version>.md` files rather than in Python string
literals, so a prompt change is a reviewable diff and every call can record exactly
which text produced it. See `registry.py`.
"""
