import {
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
} from "@mediapipe/tasks-vision";
import "./styles.css";

const MEDIAPIPE_VERSION = "0.10.34";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const SELFIE_SEGMENTER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";
const HAND_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

const CONFIG = {
  background: [15, 14, 12],
  baseColor: [210, 210, 200],
  disturbedColor: [240, 235, 220],
  hotColor: [255, 244, 225],
  rimColor: [255, 250, 240],
  gravity: 0.08,
  friction: 0.87,
  returnForce: 0.015,
  settleReturnForce: 0.05,
  pushStrength: 6.0,
  handGatherRadius: 110,
  handGatherStrength: 1.8,
  handBlastRadius: 140,
  handBlastStrength: 11.0,
  handEnergyBoost: 0.9,
  handChargeRate: 0.026,
  maxHandCharge: 2.4,
  maxBlastCharge: 1.8,
  gatherRadiusMultiplier: 1.18,
  gatherRadiusExponent: 1.45,
  gatherSofteningGain: 1.15,
  gatherFieldFloor: 0.18,
  burstDuration: 12,
  burstRadiusGain: 520,
  burstForceGain: 5.5,
  particleRadius: 2,
};

const elements = {
  video: document.querySelector("#camera"),
  canvas: document.querySelector("#scene"),
  cameraButton: document.querySelector("#cameraButton"),
  pauseButton: document.querySelector("#pauseButton"),
  underlayToggle: document.querySelector("#underlayToggle"),
  particleCount: document.querySelector("#particleCount"),
  cameraSelect: document.querySelector("#cameraSelect"),
  status: document.querySelector("#status"),
};

const ctx = elements.canvas.getContext("2d", { alpha: false });

let stream = null;
let vision = null;
let imageSegmenter = null;
let handLandmarker = null;
let modelsPromise = null;
let latestMask = null;
let currentHands = [];
let particleSystem = null;
let running = false;
let paused = false;
let frameIndex = 0;
let lastVideoTime = -1;
let viewRect = { x: 0, y: 0, width: 1, height: 1 };

class ParticleSystem {
  constructor(width, height, count) {
    this.reset(width, height, count);
  }

  reset(width, height, count) {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.count = count;
    this.radius = Math.max(
      1,
      Math.round(CONFIG.particleRadius * Math.min(this.width / 1280, this.height / 720, 1.6)),
    );
    this.homeX = new Float32Array(count);
    this.homeY = new Float32Array(count);
    this.x = new Float32Array(count);
    this.y = new Float32Array(count);
    this.vx = new Float32Array(count);
    this.vy = new Float32Array(count);
    this.energy = new Float32Array(count);
    this.boundaryGlow = new Uint8Array(count);
    this.previousMask = null;
    this.noPersonFrames = 0;
    this.handCharge = new Map();
    this.burstWaves = [];

    const cols = Math.ceil(Math.sqrt((count * this.width) / this.height));
    const rows = Math.ceil(count / cols);
    const usableWidth = Math.max(1, this.width - this.radius * 2 - 1);
    const usableHeight = Math.max(1, this.height - this.radius * 2 - 1);

    for (let index = 0; index < count; index += 1) {
      const col = index % cols;
      const row = Math.floor(index / cols);
      const homeX =
        this.radius + (cols > 1 ? (col / (cols - 1)) * usableWidth : usableWidth / 2);
      const homeY =
        this.radius + (rows > 1 ? (row / (rows - 1)) * usableHeight : usableHeight / 2);
      this.homeX[index] = homeX;
      this.homeY[index] = homeY;
      this.x[index] = homeX;
      this.y[index] = homeY;
    }
  }

  resize(width, height) {
    if (Math.abs(width - this.width) < 2 && Math.abs(height - this.height) < 2) {
      return;
    }

    this.reset(width, height, this.count);
  }

  setCount(count) {
    if (count === this.count) {
      return;
    }

    this.reset(this.width, this.height, count);
  }

  sampleMask(mask, x, y, rect) {
    if (!mask || !mask.data.length) {
      return 0;
    }

    const displayX = (x - rect.x) / rect.width;
    const displayY = (y - rect.y) / rect.height;

    if (displayX < 0 || displayX > 1 || displayY < 0 || displayY > 1) {
      return 0;
    }

    const videoX = 1 - displayX;
    const maskX = clamp(Math.floor(videoX * mask.width), 0, mask.width - 1);
    const maskY = clamp(Math.floor(displayY * mask.height), 0, mask.height - 1);
    return mask.data[maskY * mask.width + maskX] > 0 ? 1 : 0;
  }

  update(mask, hands, rect, maskStats) {
    const hasPerson = Boolean(maskStats);
    this.noPersonFrames = hasPerson ? 0 : this.noPersonFrames + 1;

    const activeHands = this.updateHandState(hands);
    const activeWaves = this.updateWaves();
    const settling = this.noPersonFrames > 30;
    const returnForce = settling ? CONFIG.settleReturnForce : CONFIG.returnForce;
    const edgeOffset = Math.max(4, this.radius * 3);

    for (let index = 0; index < this.count; index += 1) {
      let px = this.x[index];
      let py = this.y[index];
      let vx = this.vx[index];
      let vy = this.vy[index];
      let energy = this.energy[index] * 0.9;
      let glow = 0;

      if (hasPerson) {
        const inside = this.sampleMask(mask, px, py, rect);
        const wasInside = this.sampleMask(this.previousMask, px, py, rect);

        if (inside) {
          const left = this.sampleMask(mask, px - edgeOffset, py, rect);
          const right = this.sampleMask(mask, px + edgeOffset, py, rect);
          const up = this.sampleMask(mask, px, py - edgeOffset, rect);
          const down = this.sampleMask(mask, px, py + edgeOffset, rect);
          const edgeStrength = Math.abs(left - right) + Math.abs(up - down);
          let pushX = left - right;
          let pushY = up - down;

          if (edgeStrength === 0 && maskStats) {
            pushX = px - maskStats.centerX;
            pushY = py - maskStats.centerY;
          }

          const magnitude = Math.max(Math.hypot(pushX, pushY), 1e-6);
          const force = CONFIG.pushStrength * (edgeStrength > 0 ? 1 : 0.36);
          vx += (pushX / magnitude) * force;
          vy += (pushY / magnitude) * force;
          energy += edgeStrength > 0 ? 0.72 : 0.28;
          glow = edgeStrength > 0 ? 1 : 0;
        }

        if (inside !== wasInside && maskStats) {
          const motionX = px - maskStats.centerX;
          const motionY = py - maskStats.centerY;
          const magnitude = Math.max(Math.hypot(motionX, motionY), 1e-6);
          vx += (motionX / magnitude) * CONFIG.pushStrength * 0.55;
          vy += (motionY / magnitude) * CONFIG.pushStrength * 0.55;
          energy += 0.48;
        }
      }

      for (const hand of activeHands) {
        const dx = hand.x - px;
        const dy = hand.y - py;
        const distance = Math.hypot(dx, dy);
        const safeDistance = Math.max(distance, 1e-6);

        if (hand.isClosed) {
          const chargeRatio = clamp(hand.charge / CONFIG.maxHandCharge, 0, 1);
          const maxGatherRadius = Math.hypot(this.width, this.height) * CONFIG.gatherRadiusMultiplier;
          const gatherRadius =
            CONFIG.handGatherRadius +
            (maxGatherRadius - CONFIG.handGatherRadius) *
              chargeRatio ** CONFIG.gatherRadiusExponent;
          const frontWidth = Math.max(CONFIG.handGatherRadius * 0.5, gatherRadius * 0.18);
          const fieldActivation = smoothstep(clamp((gatherRadius - distance) / frontWidth, 0, 1));
          const softening = CONFIG.handGatherRadius * (1 + hand.charge * CONFIG.gatherSofteningGain);
          const nearWeight = (softening * softening) / (distance * distance + softening * softening);
          const minimumPull = CONFIG.gatherFieldFloor * chargeRatio * chargeRatio;
          const gatherScale = fieldActivation * Math.max(nearWeight, minimumPull);
          const gatherStrength =
            CONFIG.handGatherStrength *
            (0.7 + hand.charge * 1.15 + hand.charge * hand.charge * 0.38);
          vx += (dx / safeDistance) * gatherStrength * gatherScale;
          vy += (dy / safeDistance) * gatherStrength * gatherScale;

          // Held fists expand into a field-wide pull; nearby particles still feel it first.
          const damping = clamp(
            fieldActivation * nearWeight * (0.08 + hand.charge * 0.18),
            0,
            0.44,
          );
          vx *= 1 - damping;
          vy *= 1 - damping;
          energy += gatherScale * CONFIG.handEnergyBoost * (0.5 + hand.charge * 0.55);
        }

        if (hand.justOpened) {
          const blastRadius = CONFIG.handBlastRadius + hand.blastCharge * CONFIG.burstRadiusGain;
          const blastScale = Math.sqrt(Math.max(0, (blastRadius - distance) / blastRadius));
          const blastStrength =
            CONFIG.handBlastStrength * (2.4 + hand.blastCharge * CONFIG.burstForceGain);
          vx += (-dx / safeDistance) * blastStrength * blastScale;
          vy += (-dy / safeDistance) * blastStrength * blastScale;
          energy += blastScale * (1.35 + hand.blastCharge * 0.85);
        }
      }

      for (const wave of activeWaves) {
        const dx = px - wave.x;
        const dy = py - wave.y;
        const distance = Math.hypot(dx, dy);
        const safeDistance = Math.max(distance, 1e-6);
        const waveRadius = Math.min(
          wave.radius * (0.22 + wave.progress * 0.9),
          Math.hypot(this.width, this.height),
        );
        const waveBand = 34 + wave.charge * 28;
        const waveScale = Math.max(0, 1 - Math.abs(distance - waveRadius) / waveBand) ** 1.6;
        const waveStrength =
          CONFIG.handBlastStrength * (1.4 + wave.charge * 2.8) * (1 - wave.progress);
        vx += (dx / safeDistance) * waveStrength * waveScale;
        vy += (dy / safeDistance) * waveStrength * waveScale;
        energy += waveScale * (0.3 + wave.charge * 0.35);
      }

      vx += (this.homeX[index] - px) * returnForce;
      vy += (this.homeY[index] - py) * returnForce;

      if (!settling) {
        vx += (Math.random() - 0.5) * 0.04;
        vy += CONFIG.gravity;
      } else {
        energy *= 0.86;
      }

      vx *= CONFIG.friction;
      vy *= CONFIG.friction;

      px = clamp(px + vx, this.radius, this.width - 1 - this.radius);
      py = clamp(py + vy, this.radius, this.height - 1 - this.radius);

      if (px === this.radius || px === this.width - 1 - this.radius) {
        vx = 0;
      }
      if (py === this.radius || py === this.height - 1 - this.radius) {
        vy = 0;
      }

      if (settling) {
        const settled =
          Math.abs(px - this.homeX[index]) < 0.5 &&
          Math.abs(py - this.homeY[index]) < 0.5 &&
          Math.abs(vx) < 0.05 &&
          Math.abs(vy) < 0.05;
        if (settled) {
          px = this.homeX[index];
          py = this.homeY[index];
          vx = 0;
          vy = 0;
        }
      }

      this.x[index] = px;
      this.y[index] = py;
      this.vx[index] = vx;
      this.vy[index] = vy;
      this.energy[index] = clamp(energy, 0, 1.45);
      this.boundaryGlow[index] = glow;
    }

    this.previousMask = mask;
  }

  updateHandState(hands) {
    if (!hands.length) {
      this.handCharge.clear();
      return [];
    }

    const seenLabels = new Set();
    const activeHands = [];

    for (const hand of hands) {
      seenLabels.add(hand.label);
      const previousCharge = this.handCharge.get(hand.label) ?? 0;
      let charge = previousCharge;
      let blastCharge = previousCharge;

      if (hand.isClosed) {
        charge = Math.min(previousCharge + CONFIG.handChargeRate, CONFIG.maxHandCharge);
        this.handCharge.set(hand.label, charge);
      } else if (hand.justOpened) {
        blastCharge = Math.max(Math.min(previousCharge, CONFIG.maxBlastCharge), 0.4);
        this.handCharge.delete(hand.label);
        this.burstWaves.push({
          x: hand.x,
          y: hand.y,
          charge: blastCharge,
          radius: CONFIG.handBlastRadius + blastCharge * CONFIG.burstRadiusGain,
          age: 0,
        });
      } else {
        const fadedCharge = Math.max(previousCharge - 0.18, 0);
        if (fadedCharge > 0) {
          this.handCharge.set(hand.label, fadedCharge);
        } else {
          this.handCharge.delete(hand.label);
        }
        charge = fadedCharge;
      }

      activeHands.push({ ...hand, charge, blastCharge });
    }

    for (const label of this.handCharge.keys()) {
      if (!seenLabels.has(label)) {
        this.handCharge.delete(label);
      }
    }

    return activeHands;
  }

  updateWaves() {
    const activeWaves = [];

    for (const wave of this.burstWaves) {
      wave.age += 1;
      const progress = wave.age / CONFIG.burstDuration;
      if (progress < 1) {
        wave.progress = progress;
        activeWaves.push(wave);
      }
    }

    this.burstWaves = activeWaves;
    return activeWaves;
  }

  draw(context) {
    for (let index = 0; index < this.count; index += 1) {
      const displaced = Math.hypot(
        this.x[index] - this.homeX[index],
        this.y[index] - this.homeY[index],
      ) > 5;
      const baseColor = displaced ? CONFIG.disturbedColor : CONFIG.baseColor;
      const energyMix = clamp(this.energy[index], 0, 1);
      const color = this.boundaryGlow[index]
        ? CONFIG.rimColor
        : mixColor(baseColor, CONFIG.hotColor, energyMix);
      const size = this.boundaryGlow[index] ? this.radius * 3 : this.radius * 2;

      context.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      context.fillRect(
        Math.round(this.x[index] - size / 2),
        Math.round(this.y[index] - size / 2),
        size,
        size,
      );
    }
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function smoothstep(value) {
  return value * value * (3 - 2 * value);
}

function mixColor(from, to, amount) {
  return [
    Math.round(from[0] + (to[0] - from[0]) * amount),
    Math.round(from[1] + (to[1] - from[1]) * amount),
    Math.round(from[2] + (to[2] - from[2]) * amount),
  ];
}

function setStatus(message, level = "info") {
  elements.status.value = message;
  elements.status.dataset.level = level;
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(window.innerWidth * dpr));
  const height = Math.max(1, Math.round(window.innerHeight * dpr));

  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
    particleSystem?.resize(width, height);
  }
}

function getVideoRect() {
  const videoWidth = elements.video.videoWidth || 16;
  const videoHeight = elements.video.videoHeight || 9;
  const canvasWidth = elements.canvas.width;
  const canvasHeight = elements.canvas.height;
  const scale = Math.max(canvasWidth / videoWidth, canvasHeight / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;

  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
}

function videoPointToCanvas(x, y) {
  return {
    x: viewRect.x + (1 - x) * viewRect.width,
    y: viewRect.y + y * viewRect.height,
  };
}

function drawMirroredVideo() {
  ctx.save();
  ctx.translate(viewRect.x + viewRect.width, viewRect.y);
  ctx.scale(-1, 1);
  ctx.drawImage(elements.video, 0, 0, viewRect.width, viewRect.height);
  ctx.restore();
}

async function initModels() {
  if (modelsPromise) {
    return modelsPromise;
  }

  modelsPromise = (async () => {
    setStatus("Models");
    vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: SELFIE_SEGMENTER_MODEL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_LANDMARKER_MODEL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  })();

  return modelsPromise;
}

async function startCamera() {
  try {
    elements.cameraButton.disabled = true;
    setStatus("Loading");
    await initModels();

    if (stream) {
      stopCamera();
    }

    const deviceId = elements.cameraSelect.value;
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: deviceId ? undefined : "user",
      },
      audio: false,
    });

    elements.video.srcObject = stream;
    await elements.video.play();
    await populateCameraList();

    running = true;
    paused = false;
    latestMask = null;
    currentHands = [];
    lastVideoTime = -1;
    elements.pauseButton.disabled = false;
    elements.pauseButton.textContent = "Pause";
    elements.cameraButton.textContent = "Stop";
    elements.cameraButton.disabled = false;
    setStatus("Live");
  } catch (error) {
    console.error(error);
    elements.cameraButton.disabled = false;
    elements.cameraButton.textContent = "Start";
    setStatus("Camera blocked", "error");
  }
}

function stopCamera() {
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }

  stream = null;
  running = false;
  paused = false;
  latestMask = null;
  currentHands = [];
  elements.video.srcObject = null;
  elements.pauseButton.disabled = true;
  elements.pauseButton.textContent = "Pause";
  elements.cameraButton.textContent = "Start";
  setStatus("Ready");
}

async function populateCameraList() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return;
  }

  const selectedDeviceId = elements.cameraSelect.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoInputs = devices.filter((device) => device.kind === "videoinput");
  elements.cameraSelect.replaceChildren();

  for (const [index, device] of videoInputs.entries()) {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    elements.cameraSelect.append(option);
  }

  if (selectedDeviceId) {
    elements.cameraSelect.value = selectedDeviceId;
  }
}

function updateDetectors(now) {
  if (
    !running ||
    paused ||
    !imageSegmenter ||
    !handLandmarker ||
    elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    elements.video.currentTime === lastVideoTime
  ) {
    return;
  }

  lastVideoTime = elements.video.currentTime;
  frameIndex += 1;

  if (frameIndex % 2 === 0) {
    imageSegmenter.segmentForVideo(elements.video, now, (result) => {
      const categoryMask = result.categoryMask;
      if (!categoryMask) {
        latestMask = null;
        return;
      }

      latestMask = {
        data: new Uint8Array(categoryMask.getAsUint8Array()),
        width: categoryMask.width,
        height: categoryMask.height,
      };
      categoryMask.close();
    });
  }

  const handResult = handLandmarker.detectForVideo(elements.video, now);
  currentHands = mapHands(handResult);
}

function mapHands(handResult) {
  if (!handResult?.landmarks?.length) {
    previousClosedHands.clear();
    return [];
  }

  const hands = [];
  const seenLabels = new Set();

  for (let index = 0; index < handResult.landmarks.length; index += 1) {
    const landmarks = handResult.landmarks[index];
    const label =
      handResult.handednesses?.[index]?.[0]?.categoryName ||
      handResult.handedness?.[index]?.[0]?.categoryName ||
      `Hand ${index + 1}`;
    const center = averageLandmarks(landmarks, [0, 5, 9, 13, 17]);
    const canvasPoint = videoPointToCanvas(center.x, center.y);
    const { isClosed, openness } = isClosedFist(landmarks, label);
    const previousClosed = previousClosedHands.get(label) ?? false;

    seenLabels.add(label);
    hands.push({
      label,
      x: clamp(canvasPoint.x, 0, elements.canvas.width - 1),
      y: clamp(canvasPoint.y, 0, elements.canvas.height - 1),
      isClosed,
      justOpened: previousClosed && !isClosed,
      justClosed: !previousClosed && isClosed,
      openness,
    });
    previousClosedHands.set(label, isClosed);
  }

  for (const label of previousClosedHands.keys()) {
    if (!seenLabels.has(label)) {
      previousClosedHands.delete(label);
    }
  }

  return hands;
}

const previousClosedHands = new Map();

function averageLandmarks(landmarks, indices) {
  let x = 0;
  let y = 0;

  for (const index of indices) {
    x += landmarks[index].x;
    y += landmarks[index].y;
  }

  return {
    x: x / indices.length,
    y: y / indices.length,
  };
}

function isClosedFist(landmarks, label) {
  const wrist = landmarks[0];
  const palmAnchors = [5, 9, 13, 17];
  let palmSize = 0;

  for (const index of palmAnchors) {
    palmSize += distance2d(landmarks[index], wrist);
  }

  palmSize = Math.max(palmSize / palmAnchors.length, 1e-6);

  const tipPoints = [8, 12, 16, 20];
  let extension = 0;
  for (const index of tipPoints) {
    extension += distance2d(landmarks[index], wrist) / palmSize;
  }
  extension /= tipPoints.length;

  const thumbExtension = distance2d(landmarks[4], landmarks[2]) / palmSize;
  const openness = extension + thumbExtension * 0.35;
  const previousClosed = previousClosedHands.get(label) ?? false;
  const threshold = previousClosed ? 1.9 : 1.55;

  return {
    isClosed: openness < threshold,
    openness,
  };
}

function distance2d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function analyzeMask(mask) {
  if (!mask || !mask.data.length) {
    return null;
  }

  const step = mask.width * mask.height > 90000 ? 3 : 2;
  let count = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < mask.height; y += step) {
    for (let x = 0; x < mask.width; x += step) {
      if (mask.data[y * mask.width + x] === 0) {
        continue;
      }

      count += 1;
      sumX += x;
      sumY += y;
    }
  }

  if (count === 0) {
    return null;
  }

  const videoX = sumX / count / Math.max(1, mask.width - 1);
  const videoY = sumY / count / Math.max(1, mask.height - 1);
  const center = videoPointToCanvas(videoX, videoY);

  return {
    centerX: center.x,
    centerY: center.y,
  };
}

function drawFrame() {
  ctx.fillStyle = `rgb(${CONFIG.background[0]}, ${CONFIG.background[1]}, ${CONFIG.background[2]})`;
  ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);

  if (
    running &&
    !paused &&
    elements.underlayToggle.checked &&
    elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    ctx.save();
    ctx.globalAlpha = 0.16;
    drawMirroredVideo();
    ctx.restore();
  }

  particleSystem.draw(ctx);
}

function render(now) {
  resizeCanvas();
  viewRect = getVideoRect();
  updateDetectors(now);

  const maskStats = paused ? null : analyzeMask(latestMask);
  particleSystem.update(paused ? null : latestMask, paused ? [] : currentHands, viewRect, maskStats);
  drawFrame();

  requestAnimationFrame(render);
}

elements.cameraButton.addEventListener("click", () => {
  if (running) {
    stopCamera();
  } else {
    void startCamera();
  }
});

elements.pauseButton.addEventListener("click", () => {
  paused = !paused;
  elements.pauseButton.textContent = paused ? "Resume" : "Pause";
  setStatus(paused ? "Paused" : "Live");
});

elements.cameraSelect.addEventListener("change", () => {
  if (running) {
    void startCamera();
  }
});

elements.particleCount.addEventListener("input", () => {
  particleSystem.setCount(Number(elements.particleCount.value));
});

window.addEventListener("resize", resizeCanvas);

if (!navigator.mediaDevices?.getUserMedia) {
  elements.cameraButton.disabled = true;
  setStatus("No camera API", "error");
}

resizeCanvas();
particleSystem = new ParticleSystem(
  elements.canvas.width,
  elements.canvas.height,
  Number(elements.particleCount.value),
);
void populateCameraList();
requestAnimationFrame(render);
