.PHONY: install install-dev test test-cov lint format run chaos clean

# Install the library
install:
	pip install -e .

# Install with all dev + server dependencies
install-dev:
	pip install -e ".[all]"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=llm_shield --cov-report=term-missing --cov-report=html

# Lint
lint:
	ruff check llm_shield/ server/ tests/

# Format
format:
	ruff format llm_shield/ server/ tests/

# Run the development server
run:
	uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run chaos testing (opt-in, deliberate fault injection)
# Usage:
#   make chaos                          # 20 requests, 40% fault rate
#   make chaos ARGS="--requests 100"    # 100 requests
#   make chaos ARGS="--seed 42"         # reproducible run
#   make chaos ARGS="--fault-rate 0.6"  # 60% of calls get a fault
ARGS ?=
chaos:
	@echo "⚠️  Chaos mode — fault injection ACTIVE. Not for production."
	python3 -m tests.chaos_runner $(ARGS)

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
