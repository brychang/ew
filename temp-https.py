import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Change to the directory containing the annotation files
os.chdir("data/annotations_ribbon_2026-03-23_11-46-27")


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


HTTPServer(("127.0.0.1", 9000), CORSRequestHandler).serve_forever()
