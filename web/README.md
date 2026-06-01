# MediaPipe Capture UI

## Run

Serve the project root with any static server:

```bat
python -m http.server 8000
```

Open `http://localhost:8000/web/`.

## Output format

Each saved file is a JSON with:

- `label`: tên cử chỉ.
- `frames`: list frame, mỗi frame có 21 điểm `(x, y, z)`.
