#!/bin/bash

# Exit on error
set -e

# Extract version from pyproject.toml using grep and cut
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)

if [ -z "$VERSION" ]; then
    echo "Error: Could not extract version from pyproject.toml"
    exit 1
fi

echo "Deploying documentation for version $VERSION"

# Deploy the documentation using mike
mike deploy --push --update-aliases "$VERSION" latest

# Set the default version to latest
# mike set-default --push latest

echo "Documentation deployed successfully for version $VERSION" 
