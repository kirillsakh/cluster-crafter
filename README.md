# Project Setup and Execution Guide

## Project Description
<p>This project aims to create a command-line tool for performing data clustering.</p>

## Local Development

### Prerequisites
<ul><li>Python 3.11</li></ul>

### Create Virtual Environment

```
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment on Linux or macOS
source .venv/bin/activate

# Activate the virtual environment on Windows
.\.venv\Scripts\activate
```
### Install Dependencies

```
# Upgrade pip
pip install --upgrade pip

# Install pip-tools
pip install pip-tools

# Generate requirements.txt
pip-compile requirements.in

# Install dependencies
pip install -r requirements.txt
```
