# Deploying The EYE publicly (Batch A notes)

## Public mode
Set `M2_PUBLIC=1` (the service file does). It turns on:
- upload limit 8 MB, images only (JPEG, PNG, WebP), processed in memory, never written to disk;
- per-visitor limits: 40 detections, 8 COMPOUND runs and 20 uploads per minute; 2 stress tests per 10 minutes;
- stress tests capped at 20 grids (`M2_STRESS_MAX` overrides), one at a time for the whole server;
- internal error details hidden from visitors;
- every model preloaded at start, so no visitor waits for a lazy load.

Without `M2_PUBLIC`, the EYE behaves exactly as before on your laptop.

## Health and auto-restart
- `GET /health` returns 200 with `{"ok": true, "warm": "ready", "uptime_s": ...}` (503 if the model failed to load).
- Point a free uptime monitor (for example UptimeRobot) at `https://<your-address>/health` to get an email if it goes down.
- `eye.service` restarts the EYE 5 s after any crash and starts it on boot.

## Link preview
The page carries Open Graph and Twitter tags with `/og.jpg` (1200 x 630), built from the address visitors use,
so LinkedIn and other sites show a proper card. After deploying, check the card with LinkedIn's Post Inspector.

## Keep the server behind a tunnel
Keep `M2_HOST=127.0.0.1` and publish through a Cloudflare tunnel (free), so the Python server itself is never exposed.
The EYE reads the visitor's address from the tunnel's `CF-Connecting-IP` header for the per-visitor limits.

## Batch C: speed, caching, the counter

### Faster CPU inference with ONNX (optional, measure before relying on it)
The EYE uses an exported `<weights>.onnx` instead of the `.pt` file when `M2_ONNX=1` is set and the `.onnx` file
sits next to the weights. Each model must be exported at the image size the EYE uses for it (`model_imgsz` in its settings).
To export and verify (change nothing else while doing it):

1. Run `pip install onnx onnxruntime` in the EYE's environment.
2. For each detector in the EYE's settings (models + model_imgsz), run Ultralytics export with format=onnx, imgsz=<its model_imgsz>,
   dynamic=False, simplify=True, writing <weights>.onnx next to each .pt file.
3. Verify: start the EYE twice on port 7869 (once normally, once with M2_ONNX=1), make the 3x3 grid with seed 7 and detect with
   each detector; report, per detector, whether the boxes (object IDs and results) are identical and the detection time in each run.
4. Report the file sizes and any detector whose results differ. STOP.

Only switch `M2_ONNX=1` on for the server if every detector's results are identical (or differ only in tiny box decimals).

### Result caching
A seeded picture (same layout, size, mix, sections and seed) is always the same picture, so the EYE caches its detection
(120 most recent) and its COMPOUND view (60). Repeated demos and shared links answer instantly; the status line says
"instant (seen this picture before)". Uploads are never cached. The cache lives in memory and empties on restart.

### The live counter
Counts objects the EYE has looked at. In public mode it is saved every 30 s to `eye_counter.json`
(set `M2_COUNTER` to move it), so it survives restarts.
