#!/bin/bash

# Ensure ruff is installed
if ! command -v ruff &> /dev/null; then
    echo "ruff not found, installing..."
    pip install ruff
fi

# Ensure reports directory exists
mkdir -p reports

# Run ruff checks
echo "Running style checks..."
ruff check . --exclude=experiments --output-format=github > reports/ruff_report.txt

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ Style checks passed!"
else
    echo "❌ Style checks failed. Please check reports/ruff_report.txt for details"
fi

exit $exit_code 