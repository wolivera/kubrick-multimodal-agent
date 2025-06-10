FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install application dependencies
COPY . /app
WORKDIR /app
RUN uv sync --frozen --no-cache

# Run the Chainlit CLI
CMD ["/app/.venv/bin/chainlit", "run", "chainlit_app.py", "--port", "8501", "--host", "0.0.0.0"]
