#!/usr/bin/env python3
"""
Development script with hot reloading for Style Finder AI
Automatically restarts the app when files change
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Project root
PROJECT_ROOT = Path(__file__).parent
APP_FILE = PROJECT_ROOT / "app.py"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

# Files/directories to watch
WATCH_PATTERNS = [
    "*.py",
    "*.css",
    "*.html",
    "*.js",
]

# Files/directories to ignore
IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    "venv",
    ".env",
    "*.log",
    "*.tmp",
]

class AppReloadHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def __init__(self, restart_callback):
        super().__init__()
        self.restart_callback = restart_callback
        self.last_restart = 0
        self.debounce_seconds = 1  # Wait 1 second before restarting
    
    def should_ignore(self, path):
        """Check if path should be ignored"""
        path_str = str(path)
        for pattern in IGNORE_PATTERNS:
            if pattern in path_str:
                return True
        return False
    
    def on_modified(self, event):
        """Called when a file is modified"""
        if event.is_directory:
            return
        
        if self.should_ignore(event.src_path):
            return
        
        # Check if file matches watch patterns
        file_path = Path(event.src_path)
        if not any(file_path.match(pattern) for pattern in WATCH_PATTERNS):
            return
        
        # Debounce: don't restart too frequently
        current_time = time.time()
        if current_time - self.last_restart < self.debounce_seconds:
            return
        
        self.last_restart = current_time
        print(f"\n🔄 File changed: {file_path.name}")
        print(f"   Path: {event.src_path}")
        print("   Restarting app...\n")
        self.restart_callback()

class HotReloadApp:
    """Manages the app process with hot reloading"""
    
    def __init__(self):
        self.process = None
        self.observer = None
        self.running = True
    
    def start_app(self):
        """Start the Gradio app"""
        if self.process:
            self.stop_app()
        
        print("=" * 60)
        print("🚀 Starting Style Finder AI...")
        print("=" * 60)
        
        # Use venv Python if available, otherwise system Python
        python_cmd = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        
        try:
            self.process = subprocess.Popen(
                [python_cmd, str(APP_FILE)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Print output in real-time
            self._print_output()
            
        except Exception as e:
            print(f"❌ Error starting app: {e}")
            return False
        
        return True
    
    def _print_output(self):
        """Print app output in a separate thread"""
        import threading
        
        def print_loop():
            if not self.process:
                return
            
            try:
                for line in iter(self.process.stdout.readline, ''):
                    if not line:
                        break
                    print(line.rstrip())
            except Exception:
                pass
        
        thread = threading.Thread(target=print_loop, daemon=True)
        thread.start()
    
    def stop_app(self):
        """Stop the Gradio app"""
        if self.process:
            print("\n⏹️  Stopping app...")
            try:
                # Try graceful shutdown
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                print(f"⚠️  Error stopping app: {e}")
            finally:
                self.process = None
    
    def restart_app(self):
        """Restart the app"""
        self.stop_app()
        time.sleep(0.5)  # Brief pause before restart
        self.start_app()
    
    def start_watcher(self):
        """Start the file system watcher"""
        event_handler = AppReloadHandler(self.restart_app)
        self.observer = Observer()
        
        # Watch the project root and subdirectories
        self.observer.schedule(
            event_handler,
            str(PROJECT_ROOT),
            recursive=True
        )
        
        self.observer.start()
        print("\n👀 Watching for file changes...")
        print(f"   Watching: {PROJECT_ROOT}")
        print(f"   Patterns: {', '.join(WATCH_PATTERNS)}")
        print(f"   Ignoring: {', '.join(IGNORE_PATTERNS)}")
        print("\n" + "=" * 60)
        print("💡 Press Ctrl+C to stop")
        print("=" * 60 + "\n")
    
    def stop_watcher(self):
        """Stop the file system watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
    
    def run(self):
        """Run the app with hot reloading"""
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n\n🛑 Shutting down...")
            self.running = False
            self.stop_app()
            self.stop_watcher()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the app
        if not self.start_app():
            return
        
        # Start watching for changes
        self.start_watcher()
        
        # Keep running
        try:
            while self.running:
                time.sleep(1)
                # Check if process is still alive
                if self.process and self.process.poll() is not None:
                    print("\n⚠️  App process ended unexpectedly")
                    if self.running:
                        print("   Restarting...")
                        time.sleep(2)
                        self.start_app()
        except KeyboardInterrupt:
            signal_handler(None, None)

def main():
    """Main entry point"""
    # Check if watchdog is installed
    try:
        import watchdog
    except ImportError:
        print("❌ Error: 'watchdog' package is required for hot reloading")
        print("   Install it with: pip install watchdog")
        sys.exit(1)
    
    # Check if app.py exists
    if not APP_FILE.exists():
        print(f"❌ Error: {APP_FILE} not found")
        sys.exit(1)
    
    # Create and run the hot reload app
    app = HotReloadApp()
    app.run()

if __name__ == "__main__":
    main()

