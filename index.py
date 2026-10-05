"""Vercel entrypoint: Vercel's Python runtime looks for an ASGI `app` in index.py.

Locally / in Docker use:  uvicorn backend.app.main:app
"""
from backend.app.main import app  # noqa: F401
