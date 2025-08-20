#!/usr/bin/env python3
"""
Simple HTTP server for serving the locomotion data analysis dashboard
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    PORT = 8000
    
    # Check if port is already in use
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            print(f"🚀 Server starting on http://localhost:{PORT}")
            print(f"📁 Serving files from: {script_dir}")
            print(f"🌐 Open your browser and go to: http://localhost:{PORT}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Try a different port:")
            print(f"   python server.py --port 8001")
        else:
            print(f"❌ Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
