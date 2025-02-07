# Setting Up Lemon-Aid

## Quick Setup (Recommended)

1. Clone and setup:
   ```bash
   git clone https://github.com/jakerains/lemon-aid.git
   cd lemon-aid
   python setup.py
   ```

2. Activate and run:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Unix/Mac
   source venv/bin/activate

   # Run Lemon-Aid
   python launch.py
   ```

That's it! 🎉

## Troubleshooting

If you encounter any issues:
1. Make sure Python 3.8+ is installed
2. Try: `pip install -U uv`
3. Delete the venv folder and try again

[See detailed setup guide](docs/SETUP_DETAILED.md)

## Using UV (Recommended)

UV is a fast, reliable Python package installer and resolver. Here's how to set up Lemon-Aid using UV:

1. Install UV:
   ```bash
   pip install uv
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/jakerains/lemon-aid.git
   cd lemon-aid
   ```

3. Run the setup script:
   ```bash
   # Windows
   python scripts/setup.py

   # Unix/Mac
   python3 scripts/setup.py
   ```

4. Activate the virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Unix/Mac
   source venv/bin/activate
   ```

5. Verify installation:
   ```bash
   uv pip list  # Should show all installed packages
   ```

6. Run Lemon-Aid:
   ```bash
   python launch.py
   ```

## Manual UV Commands (if needed)

If you need to manually manage dependencies:

```bash
# Create venv with UV
uv venv venv

# Activate venv (same as above)
# Then use UV pip commands:

# Install dependencies
uv pip install -r requirements.txt

# Install dev dependencies
uv pip install -r requirements.txt[dev]

# Add new dependency
uv pip install package_name

# Update dependencies
uv pip compile requirements.txt
```

## Manual Installation (Alternative)

If you prefer not to use UV:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate it:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Unix/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ``` 