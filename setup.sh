#!/bin/bash
set -euo pipefail

uv venv -p 3.12
uv sync
