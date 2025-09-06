# Variables
PYTHON = python3
POETRY = poetry
PYTEST = pytest
PYTHON_MODULES = cogents_tools tests examples
COVERAGE_MODULES = cogents_tools
TEST_DIR = tests
LINE_LENGTH = 120

# Colors for output
BLUE = \033[34m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
RESET = \033[0m

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# SETUP COMMANDS
# =============================================================================

.PHONY: install setup check-env

install: ## Install development dependencies
	@echo "$(BLUE)🔧 Installing development dependencies...$(RESET)"
	@$(POETRY) install --with dev,tools

setup: install ## Setup development environment
	@echo "$(GREEN)✅ Development environment ready$(RESET)"

check-env: ## Check environment setup
	@echo "$(BLUE)🔍 Checking environment...$(RESET)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Poetry version: $$($(POETRY) --version)"
	@echo "Working directory: $$(pwd)"
	@echo "Python modules: $(PYTHON_MODULES)"

# =============================================================================
# TEST COMMANDS
# =============================================================================

.PHONY: test test-unit test-integration test-coverage test-watch

test: ## Run all tests.
	@echo "$(BLUE)🧪 Running all tests...$(RESET)"
	$(POETRY) run $(PYTEST) $(TEST_DIR) -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)🧪 Running unit tests...$(RESET)"
	$(POETRY) run $(PYTEST) $(TEST_DIR) -v -m "not integration"

test-integration: ## Run integration tests only
	@echo "$(BLUE)🧪 Running integration tests...$(RESET)"
	$(POETRY) run $(PYTEST) $(TEST_DIR) -v -m "integration"

test-coverage: ## Run tests with coverage
	@echo "$(BLUE)🧪 Running tests with coverage...$(RESET)"
	$(POETRY) run $(PYTEST) $(TEST_DIR) --cov=$(COVERAGE_MODULES) --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)👀 Running tests in watch mode...$(RESET)"
	$(POETRY) run pytest-watch $(TEST_DIR) -- -v

# =============================================================================
# CODE QUALITY COMMANDS
# =============================================================================

.PHONY: format format-check lint lint-fix quality autofix

format: ## Format code (black, isort, autoflake)
	@echo "$(BLUE)🎨 Formatting code...$(RESET)"
	@$(POETRY) run python -m autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables $(PYTHON_MODULES)
	@$(POETRY) run isort $(PYTHON_MODULES) --line-length $(LINE_LENGTH)
	@$(POETRY) run black $(PYTHON_MODULES) --line-length $(LINE_LENGTH)

format-check: ## Check if code is properly formatted
	@echo "$(BLUE)🔍 Checking code formatting...$(RESET)"
	@$(POETRY) run black --check $(PYTHON_MODULES) || (echo "$(RED)❌ Code formatting check failed. Run 'make format' to fix.$(RESET)" && exit 1)
	@$(POETRY) run isort --check-only $(PYTHON_MODULES) || (echo "$(RED)❌ Import sorting check failed. Run 'make format' to fix.$(RESET)" && exit 1)

lint: ## Lint code
	@echo "$(BLUE)🔍 Running linters...$(RESET)"
	@$(POETRY) run flake8 --max-line-length=$(LINE_LENGTH) --extend-ignore=E203,W503 $(PYTHON_MODULES)

lint-fix: ## Auto-fix linting issues where possible
	@echo "$(BLUE)🔧 Auto-fixing linting issues...$(RESET)"
	@$(POETRY) run python -m autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables $(PYTHON_MODULES)

quality: format-check lint ## Run all quality checks
	@echo "$(GREEN)🎉 All quality checks passed!$(RESET)"

autofix: lint-fix format ## Auto-fix all code quality issues

# =============================================================================
# BUILD COMMANDS
# =============================================================================

.PHONY: build build-wheel build-sdist package

build: ## Build package
	@echo "$(BLUE)🔨 Building package...$(RESET)"
	@$(POETRY) build

build-wheel: ## Build wheel
	@echo "$(BLUE)🔨 Building wheel...$(RESET)"
	@$(POETRY) build --format wheel

build-sdist: ## Build source distribution
	@echo "$(BLUE)🔨 Building source distribution...$(RESET)"
	@$(POETRY) build --format sdist

package: clean build ## Build and package for distribution

# =============================================================================
# CLEAN COMMANDS
# =============================================================================

.PHONY: clean clean-all

clean: ## Clean Python cache and build artifacts
	@echo "$(BLUE)🧹 Cleaning Python cache and build artifacts...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ 2>/dev/null || true

clean-all: clean ## Clean everything including dependencies
	@echo "$(GREEN)✅ Complete cleanup finished!$(RESET)"

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

.PHONY: shell requirements version

shell: ## Activate development shell
	@echo "$(BLUE)🐚 Activating development shell...$(RESET)"
	@$(POETRY) shell

requirements: ## Generate requirements files
	@echo "$(BLUE)📋 Generating requirements files...$(RESET)"
	@$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	@$(POETRY) export -f requirements.txt --output requirements-prod.txt --without-hashes --only main

version: ## Show current version
	@echo "$(BLUE)Current version:$(RESET)"
	@echo "Cogents Tools (pyproject.toml): $(shell grep '^version = ' pyproject.toml | cut -d'"' -f2)"

# =============================================================================
# DEVELOPMENT COMMANDS
# =============================================================================

.PHONY: dev dev-check full-check

dev-check: quality test-unit ## Quick development check (quality + unit tests)

full-check: format-check lint test build ## Full development check (all checks + all tests + build)

# =============================================================================
# DOCUMENTATION COMMANDS
# =============================================================================

.PHONY: docs docs-build docs-serve docs-clean

docs: ## Build documentation
	@echo "$(BLUE)📚 Building documentation...$(RESET)"
	@cd docs && make html
	@echo "$(GREEN)Documentation built: docs/_build/html/index.html$(RESET)"

docs-serve: docs ## Build and serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(RESET)"
	@cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build artifacts
	@rm -rf docs/_build/

# =============================================================================
# RELEASE COMMANDS
# =============================================================================

.PHONY: release publish publish-test check-publish-prereqs

release: clean build ## Build all release artifacts

publish: check-publish-prereqs ## Publish package to PyPI
	@echo "$(BLUE)📦 Publishing to PyPI...$(RESET)"
	@$(POETRY) publish --build
	@echo "$(GREEN)✅ Package published to PyPI$(RESET)"

publish-test: check-publish-prereqs ## Publish package to TestPyPI
	@echo "$(BLUE)📦 Publishing to TestPyPI...$(RESET)"
	@$(POETRY) publish --repository testpypi --build
	@echo "$(GREEN)✅ Package published to TestPyPI$(RESET)"

check-publish-prereqs: ## Check prerequisites for publishing
	@echo "$(BLUE)🔍 Checking publishing prerequisites...$(RESET)"
	@$(POETRY) --version >/dev/null 2>&1 || (echo "$(RED)❌ Poetry not found. Install with: pip install poetry$(RESET)" && exit 1)
	@if [ -z "$${POETRY_PYPI_TOKEN_PYPI}" ] && [ -z "$${POETRY_PYPI_TOKEN_TESTPYPI}" ] && [ ! -f ~/.pypirc ]; then \
		echo "$(YELLOW)⚠️  PyPI credentials not found. Set POETRY_PYPI_TOKEN_PYPI or configure ~/.pypirc$(RESET)"; \
		echo "$(BLUE)💡 You can set credentials with:$(RESET)"; \
		echo "   export POETRY_PYPI_TOKEN_PYPI=pypi-your_token_here"; \
		echo "   export POETRY_PYPI_TOKEN_TESTPYPI=pypi-your_test_token_here"; \
		echo "   OR configure ~/.pypirc file"; \
		echo "$(BLUE)💡 Get tokens from: https://pypi.org/manage/account/token/$(RESET)"; \
	fi

test-auth: ## Test PyPI authentication
	@echo "$(BLUE)🔐 Testing PyPI authentication...$(RESET)"
	@$(POETRY) config --list | grep pypi || echo "$(YELLOW)No Poetry PyPI config found$(RESET)"
	@if [ -n "$${POETRY_PYPI_TOKEN_PYPI}" ]; then \
		echo "$(GREEN)✅ POETRY_PYPI_TOKEN_PYPI is set$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  POETRY_PYPI_TOKEN_PYPI not set$(RESET)"; \
	fi
	@if [ -f ~/.pypirc ]; then \
		echo "$(GREEN)✅ ~/.pypirc file exists$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  ~/.pypirc file not found$(RESET)"; \
	fi

# =============================================================================
# CI/CD COMMANDS
# =============================================================================

.PHONY: ci ci-test ci-quality

ci: ci-quality ci-test ## Run CI pipeline (quality checks + tests)

ci-test: test-unit test-integration ## Run CI tests

ci-quality: format-check lint ## Run CI quality checks

# =============================================================================
# DEFAULT GOAL
# =============================================================================

.DEFAULT_GOAL := help 