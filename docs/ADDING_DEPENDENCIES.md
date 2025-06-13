# Adding Dependencies

The project uses [Poetry](https://python-poetry.org/) for dependency management. All runtime and development packages are recorded in `pyproject.toml` and locked in `poetry.lock`.

## Adding a Runtime Dependency

1. Install the package with Poetry which updates both files automatically:

```bash
poetry add <package>
```

2. Commit the modified `pyproject.toml` and `poetry.lock` files so others can reproduce the environment.

## Adding a Development Dependency

Use the `--group dev` flag to mark a dependency as development-only:

```bash
poetry add --group dev <package>
```

These packages are installed when running `poetry install` and can be used for testing or tooling.

