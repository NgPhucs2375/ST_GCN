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
let lastLandmarks = null;

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
  lastLandmarks = frame;
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
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

hands.onResults((results) => {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const landmarksList = results.multiHandLandmarks || [];
  drawLandmarks(landmarksList);

  const primaryHand = landmarksList[0] || null;
  if (isRecording && primaryHand) {
    // Keep single-hand data format for ST-GCN training.
    pushFrame(primaryHand);
    setStatus(`Recording... frames: ${recordedFrames.length}`);
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
