"""Servidor minimo del motor (stdlib http.server, sin dependencias extra). Solo escucha en 127.0.0.1.
  GET  /health                       -> {"status":"ok","model":...,"V":...}
  POST /generate  {"prompt": str, "max_tokens": int, "seed": int}
       -> {"text": generado, "tokens": n, "seconds": s, "tok_per_s": x, "prompt_chars": L}
Cada peticion crea su propia sesion (estado incremental); el modelo (mmap) es de solo lectura.
Uso: python -I server.py [puerto=8765]"""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from engine import IxchelEngine

MODEL = os.path.join(HERE, "model_v03")
MAX_TOKENS = 2000
MAX_PROMPT = 200_000
ENG = None


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "model": ENG.meta["name"], "V": ENG.V})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            prompt = req.get("prompt", "")
            max_tokens = int(req.get("max_tokens", 200))
            seed = int(req.get("seed", 0))
            if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT:
                return self._send(400, {"error": f"prompt debe ser texto de <= {MAX_PROMPT} caracteres"})
            if not (1 <= max_tokens <= MAX_TOKENS):
                return self._send(400, {"error": f"max_tokens en [1, {MAX_TOKENS}]"})
            bad = sorted({c for c in prompt if c not in ENG.stoi})
            if bad:
                return self._send(400, {"error": "caracteres fuera del vocabulario", "chars": bad})
        except Exception as ex:
            return self._send(400, {"error": f"peticion invalida: {ex}"})
        t0 = time.perf_counter()
        sess = ENG.new_session(prompt)
        text, _ = ENG.generate_fast(sess, max_tokens, seed=seed)
        dt = time.perf_counter() - t0
        self._send(200, {"text": text, "tokens": max_tokens, "seconds": dt, "tok_per_s": max_tokens / dt,
                         "prompt_chars": len(prompt)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ENG = IxchelEngine.load(MODEL)
    ENG.generate_fast(ENG.new_session(), 4, seed=0)   # JIT antes de atender
    srv = HTTPServer(("127.0.0.1", port), H)
    print(f"escuchando en http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
