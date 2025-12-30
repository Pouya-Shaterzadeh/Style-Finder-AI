# Development Guide - Hot Reloading

## Quick Start

Run the development server with hot reloading:

```bash
source venv/bin/activate
python dev.py
```

The app will automatically restart when you make changes to:
- Python files (`.py`)
- CSS files (`.css`)
- HTML files (`.html`)
- JavaScript files (`.js`)

## Features

- **Automatic Restart**: App restarts automatically when files change
- **File Watching**: Monitors all relevant files in the project
- **Debouncing**: Prevents multiple restarts from rapid file changes
- **Real-time Output**: Shows app output in the terminal
- **Graceful Shutdown**: Handles Ctrl+C cleanly

## Usage

1. **Start Development Server**:
   ```bash
   python dev.py
   ```

2. **Make Changes**: Edit any Python, CSS, or HTML files

3. **Auto-Restart**: The app will automatically detect changes and restart

4. **Stop Server**: Press `Ctrl+C` to stop

## Watched Files

The script watches for changes in:
- `*.py` - Python source files
- `*.css` - Stylesheet files
- `*.html` - HTML template files
- `*.js` - JavaScript files

## Ignored Files/Directories

The following are ignored:
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python files
- `.git/` - Git directory
- `venv/` - Virtual environment
- `.env` - Environment variables
- `*.log` - Log files
- `*.tmp` - Temporary files

## Troubleshooting

### App doesn't restart
- Check that `watchdog` is installed: `pip install watchdog`
- Verify file changes are being saved
- Check terminal output for errors

### Port already in use
- Stop any existing app instances
- Or change the port in `app.py`

### Import errors after restart
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` to ensure all dependencies are installed

## Production vs Development

- **Development**: Use `python dev.py` for hot reloading
- **Production**: Use `python app.py` directly

