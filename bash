# List all files (excluding git and cache)
find . -type f -not -path './.git/*' -not -path './.pytest_cache/*' -not -path './__pycache__/*' | sort

# Check if key missing files exist
ls -la Dockerfile docker-compose.yml .github/workflows/ci.yml .gitignore pyproject.toml 2>/dev/null || echo "SOME FILES MISSING"

# Check for duplicate/artifact files
find . -name '*\(1\)*' -o -name '*\(2\)*' -o -name '*\(3\)*' | sort
