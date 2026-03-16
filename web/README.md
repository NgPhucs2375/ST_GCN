# MediaPipe Capture UI

## Run

Serve the folder with any static server:

```bat
python -m http.server 8000
```

Open http://localhost:8000/web/

## Output format

Each saved file is a JSON with:

- label: gesture name
- frames: list of frames, each frame has 21 points (x, y, z)
