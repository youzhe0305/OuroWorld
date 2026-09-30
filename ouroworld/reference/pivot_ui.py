"""A small browser page to pick the pivot by clicking the import camera's render.

The server shows the render, answers each click with the pivot it would store
and stores the last one on "Save", then stops.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

from ouroworld.io.scene_package import PivotAnnotation

logger = logging.getLogger(__name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pivot picker</title>
<style>
body{margin:0;background:#111;color:#eee;font:16px system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:16px}
.stage{position:relative;display:inline-block;max-width:100%}
#image{display:block;max-width:100%;height:auto;cursor:crosshair}
#marker{display:none;position:absolute;width:22px;height:22px;border:3px solid #ff3b30;
border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}
#status{white-space:pre-wrap;background:#222;padding:12px;border-radius:8px;margin-top:12px}
button{margin-top:12px;padding:10px 18px;font-size:16px;border:0;border-radius:7px;
background:#2878ff;color:#fff}button:disabled{background:#555;color:#aaa}
</style></head><body><main>
<h1>Pick the pivot: __TITLE__</h1>
<p>Click the surface the camera orbits should turn around, check the depth, then save.</p>
<div class="stage"><img id="image" src="/image.png"><div id="marker"></div></div>
<div id="status">No pivot picked yet.</div><button id="save" disabled>Save pivot</button>
<script>
const image=document.querySelector('#image'),marker=document.querySelector('#marker');
const status=document.querySelector('#status'),save=document.querySelector('#save');
image.addEventListener('click',async e=>{
 const r=image.getBoundingClientRect();
 const x=(e.clientX-r.left)*image.naturalWidth/r.width;
 const y=(e.clientY-r.top)*image.naturalHeight/r.height;
 marker.style.display='block';marker.style.left=(100*x/image.naturalWidth)+'%';
 marker.style.top=(100*y/image.naturalHeight)+'%';save.disabled=true;
 const response=await fetch('/pick',{method:'POST',body:JSON.stringify({x,y})});
 const data=await response.json();
 if(!response.ok){status.textContent='Cannot use this pixel: '+data.error;return;}
 status.textContent=`pixel (${x.toFixed(1)}, ${y.toFixed(1)})\\ndepth ${data.depth.toFixed(5)}\\n`+
  `world [${data.world.map(v=>v.toFixed(5)).join(', ')}]`;save.disabled=false;});
save.addEventListener('click',async()=>{save.disabled=true;
 const response=await fetch('/save',{method:'POST',body:'{}'});const data=await response.json();
 status.textContent=response.ok?'Saved to '+data.path+'. You can close this page.'
  :'Error: '+data.error;});
</script></main></body></html>"""


def serve_pivot_picker(
    image: np.ndarray,
    pick: Callable[[float, float], PivotAnnotation],
    save: Callable[[PivotAnnotation], str],
    host: str,
    port: int,
    title: str,
) -> PivotAnnotation | None:
    """Serve the page until a pivot is saved (returned) or the server is interrupted (None)."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    png = buffer.getvalue()
    page = PAGE.replace("__TITLE__", title).encode("utf-8")
    state: dict[str, PivotAnnotation | None] = {"picked": None, "saved": None}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, document: dict[str, Any]) -> None:
            self._send(status, "application/json", json.dumps(document).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", page)
            elif self.path == "/image.png":
                self._send(200, "image/png", png)
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if self.path == "/pick":
                    pivot = pick(float(body["x"]), float(body["y"]))
                    state["picked"] = pivot
                    self._json(200, {"depth": pivot.depth, "world": pivot.world.tolist()})
                elif self.path == "/save":
                    picked = state["picked"]
                    if picked is None:
                        raise ValueError("pick a pivot first")
                    self._json(200, {"path": save(picked)})
                    state["saved"] = picked
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, KeyError) as error:
                self._json(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug(format, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("open http://%s:%d/ to pick the pivot (Ctrl-C cancels)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("cancelled")
    finally:
        server.server_close()
    return state["saved"]
