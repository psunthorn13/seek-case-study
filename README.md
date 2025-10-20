# Seek Case Study

## Overview
This project is a DS case study application & Jupyter notebook built with Python, utilising Streamlit for the UI, Pinecone for vector search, and various data processing libraries.

## Prerequisites

- **Python 3.12**: Ensure Python 3.12 is installed on your system.
- **UV**: Used for managing project dependencies.

# Setup Instructions

### 1. **Clone the Repository**
```bash
   git clone https://github.com/psunthorn13/ai_investment_agent.git
```  

### 2. **Install UV**
#### On macOS and Linux.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


#### On Windows.
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
#### With pip.
```bash
# With pip.
pip install uv
```
See the installation documentation for details and alternative installation methods. https://docs.astral.sh/uv/getting-started/installation/#pypi

Documentation

### 3. **Package Management**
We use [UV](https://docs.astral.sh/uv/getting-started/features/) as a package management tool. Follow these steps
#### Create Virtual Environment (if not exist)
```bash
uv ven
```


### If pyproject.toml is given or you want to install packages
#### Compile requirements into a lockfile.
```bash
# Prod environment
uv pip compile -o requirements.txt pyproject.toml

# Dev environment
uv pip compile --extra dev -o requirements-dev.txt pyproject.toml
```

#### Sync an environment
```bash
# Prod environment
uv pip sync requirements.txt               

# Dev environment
uv pip sync requirements-dev.txt        
```

# Execute application.
## Run Streamlit App
```bash
uv run streamlit run app.py                
```
## Run Pinecone Indexing
```bash
uv run python pinecone_indexing.py               
```

