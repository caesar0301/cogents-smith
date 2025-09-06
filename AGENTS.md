# Coding Agent Rules of cogents-tools

## General Rules

## Testing

### Testing Cases

- Put test cases under folder `tests`.
- Naming convention: unitest with `test_foo.py` and integration with `test_foo_integration.py`.
- For integrationt tests (depending on external resources, such as web service, service endpoint etc.), mark them as `@pytest.mark.integration`. Optional, mark unittests as `@pytest.mark.unit`.

### Tests Running

- Run unit and integration tests with `make test-unit` and `make test-integration` separately.

## Code Format

- Run `make format` when finishing a task or a feature before commiting.
- Run `make autofix` to fix linting and formatting errors automatically.
