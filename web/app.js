const video = document.getElementById("inputVideo");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

const labelInput = document.getElementById("labelInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const recordBtn = document.getElementById("recordBtn");
const saveBtn = document.getElementById("saveBtn");
const statusEl = document.getElementById("status");

let camera = null;
let isRecording = false;
let recordedFrames = [];

const QUALITY = {
  maxHands: 1,
  minDetConf: 0.7,
  minTrackConf: 0.7,
  landmarkEmaAlpha: 0.65,
  maxMeanJump: 0.12,
  maxConsecutiveMisses: 6,
};

let smoothedLandmarks = null;
let missingCount = 0;
let droppedFrames = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function drawLandmarks(landmarksList) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (!landmarksList || landmarksList.length === 0) return;

  const colors = ["#f0c96b", "#6bb7f0"];
  ctx.lineWidth = 2;

  landmarksList.forEach((landmarks, idx) => {
    ctx.fillStyle = colors[idx % colors.length];
    ctx.strokeStyle = "#2c2a27";

    for (const point of landmarks) {
      const x = (1 - point.x) * canvas.width;
      const y = point.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  });
}

function pushFrame(landmarks) {
  const frame = landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z }));
  recordedFrames.push(frame);
}

function smoothLandmarks(raw) {
  if (!smoothedLandmarks) {
    return raw.map((p) => ({ ...p }));
  }
  const alpha = Math.min(Math.max(QUALITY.landmarkEmaAlpha, 0), 0.99);
  return raw.map((p, i) => ({
    x: alpha * smoothedLandmarks[i].x + (1 - alpha) * p.x,
    y: alpha * smoothedLandmarks[i].y + (1 - alpha) * p.y,
    z: alpha * smoothedLandmarks[i].z + (1 - alpha) * p.z,
  }));
}

function meanJump(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const dx = a[i].x - b[i].x;
    const dy = a[i].y - b[i].y;
    sum += Math.hypot(dx, dy);
  }
  return sum / a.length;
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: QUALITY.maxHands,
  modelComplexity: 1,
  minDetectionConfidence: QUALITY.minDetConf,
  minTrackingConfidence: QUALITY.minTrackConf,
});

hands.onResults((results) => {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const landmarksList = results.multiHandLandmarks || [];
  const primaryHand = landmarksList[0] || null;

  if (primaryHand) {
    missingCount = 0;
    const smoothed = smoothLandmarks(primaryHand);
    if (smoothedLandmarks) {
      // Drop unstable frames to reduce jump noise in saved sequences.
      const jump = meanJump(smoothed, smoothedLandmarks);
      if (jump > QUALITY.maxMeanJump) {
        droppedFrames += 1;
        drawLandmarks([smoothedLandmarks]);
        return;
      }
    }
    smoothedLandmarks = smoothed;
    drawLandmarks([smoothed]);

    if (isRecording) {
      // Keep single-hand data format for ST-GCN training.
      pushFrame(smoothed);
      setStatus(
        `Recording... frames: ${recordedFrames.length} | miss: ${missingCount} | drop: ${droppedFrames}`
      );
    }
  } else {
    missingCount += 1;
    if (missingCount >= QUALITY.maxConsecutiveMisses) {
      smoothedLandmarks = null;
    }
    drawLandmarks(landmarksList);
  }
});

startBtn.addEventListener("click", async () => {
  if (camera) return;
  camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480,
  });
  await camera.start();
  setStatus("Camera started");
  startBtn.disabled = true;
  stopBtn.disabled = false;
  recordBtn.disabled = false;
});

stopBtn.addEventListener("click", () => {
  if (camera) {
    camera.stop();
    camera = null;
  }
  setStatus("Camera stopped");
  startBtn.disabled = false;
  stopBtn.disabled = true;
  recordBtn.disabled = true;
});

recordBtn.addEventListener("click", () => {
  if (!isRecording) {
    recordedFrames = [];
    droppedFrames = 0;
    missingCount = 0;
    isRecording = true;
    recordBtn.textContent = "Stop";
    saveBtn.disabled = true;
    setStatus("Recording... frames: 0");
  } else {
    isRecording = false;
    recordBtn.textContent = "Record";
    saveBtn.disabled = recordedFrames.length === 0;
    setStatus(`Recorded ${recordedFrames.length} frames`);
  }
});

saveBtn.addEventListener("click", () => {
  const label = labelInput.value.trim() || "unknown";
  const payload = {
    label,
    frames: recordedFrames,
    createdAt: new Date().toISOString(),
  };
  const filename = `${label}_${Date.now()}.json`;
  downloadJson(payload, filename);
  setStatus(`Saved ${recordedFrames.length} frames`);
});

setStatus("Idle");
