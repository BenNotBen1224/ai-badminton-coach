import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, openSync, writeSync, closeSync, readFileSync, unlinkSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const PY = process.platform === "win32" ? join(PROJECT, ".venv", "Scripts", "python.exe")
  : join(PROJECT, ".venv", "bin", "python");
const SERVE = join(PROJECT, "coach", "web", "serve.py");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const LOCK = join(PROJECT, "work", "cpu.lock");
const SHOTS = join(PROJECT, "work", "C2", "shots");
const LAYOUT = join(PROJECT, "work", "C2", "layout.json");
const FRAME_LOG = join(PROJECT, "work", "C2", "logs", "framecode.json");
const MARK_KINDS = ["dot", "label", "line", "bracket", "band", "angle", "trail"];
const SLOW = 0.2822265625;
const SIZES = [[1280, 720], [1366, 768], [1920, 1080]];

const summary = {
  paused_match: null,
  playback_within_one: null,
  layout_failures: null,
  freeze_frame: null,
  compare_frame: null,
};
const failures = [];
let owned = false;
let server = null;
let edge = null;
let profile = null;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (...args) => console.error(...args);

function fail(message) {
  failures.push(message);
  log("FAIL", message);
}

function seekFrames(count) {
  const rand = mulberry32(0xc2f1);
  const frames = [0, 1, 58, 59];
  while (frames.length < count) frames.push((rand() * 60) | 0);
  return frames.slice(0, count);
}

function mulberry32(seed) {
  let a = seed;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function acquireLock() {
  const deadline = Date.now() + 30 * 60 * 1000;
  const body = `C2 ${process.pid} browser`;
  for (;;) {
    try {
      const fd = openSync(LOCK, "wx");
      writeSync(fd, body);
      closeSync(fd);
      owned = true;
      log("lock acquired", body);
      return;
    } catch (error) {
      if (!existsSync(LOCK)) throw error;
      if (Date.now() >= deadline) throw new Error("cpu.lock still held after 30 minutes");
      let holder = "";
      try { holder = readFileSync(LOCK, "utf8"); } catch { holder = error.code || "held"; }
      log("lock held, retry in 30s:", holder);
      await sleep(30000);
    }
  }
}

function releaseLock() {
  if (!owned) return;
  const body = `C2 ${process.pid} browser`;
  try {
    const text = readFileSync(LOCK, "utf8");
    if (text === body) {
      unlinkSync(LOCK);
      owned = false;
      log("lock released");
    } else {
      log("lock text changed, left in place");
    }
  } catch (error) {
    log("lock release failed", error.message);
  }
}

function startServer() {
  return new Promise((resolve, reject) => {
    const child = spawn(PY, [SERVE, "--port", "0", PROJECT], { windowsHide: true });
    let buf = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error(`server silent: ${buf}`));
    }, 15000);
    const take = (text) => {
      buf += text;
      const match = buf.match(/port (\d+)/);
      if (!match || settled) return;
      const port = Number(match[1]);
      if (port === 8765) {
        settled = true;
        clearTimeout(timer);
        child.kill();
        reject(new Error("server bound 8765"));
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve({ child, port });
    };
    child.stdout.on("data", (chunk) => take(chunk.toString()));
    child.stderr.on("data", (chunk) => log("server", chunk.toString().trim()));
    child.on("exit", (code) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`server exited ${code}: ${buf}`));
      }
    });
  });
}

function debugPort() {
  for (;;) {
    const port = 9300 + Math.floor(Math.random() * 500);
    if (port !== 8765) return port;
  }
}

function startEdge(port, width, height) {
  profile = join(tmpdir(), `c2-viewer-${port}`);
  mkdirSync(profile, { recursive: true });
  const child = spawn(EDGE, [
    "--headless=new",
    "--disable-gpu",
    "--mute-audio",
    "--autoplay-policy=no-user-gesture-required",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    `--window-size=${width},${height}`,
    "about:blank",
  ], { stdio: "ignore", windowsHide: true });
  return child;
}

async function waitFor(fn, ms, label) {
  const end = Date.now() + ms;
  let last = "";
  for (;;) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (error) {
      last = error.message || String(error);
    }
    if (Date.now() > end) throw new Error(`timeout waiting for ${label}: ${last}`);
    await sleep(200);
  }
}

function attach(ws) {
  let next = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject, timer } = pending.get(msg.id);
      clearTimeout(timer);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result);
    }
  });
  const send = (method, params = {}, timeout = 30000) => new Promise((resolve, reject) => {
    const id = ++next;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`cdp timeout ${method}`));
    }, timeout);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params }));
  });
  return send;
}

async function evaluate(send, expression, timeout = 30000) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  }, timeout);
  if (result.exceptionDetails) {
    const text = result.exceptionDetails.text || "evaluate failed";
    const extra = result.exceptionDetails.exception && result.exceptionDetails.exception.description;
    throw new Error(extra || text);
  }
  return result.result.value;
}

async function setSize(send, width, height) {
  await send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false,
  });
}

async function openViewer(send, port, bundlePath) {
  const url = `http://127.0.0.1:${port}/badminton-coach/coach/web/viewer.html?bundle=${bundlePath}`;
  log("open", url);
  await send("Page.navigate", { url });
  await waitFor(async () => {
    const value = await evaluate(send, "typeof window.viewerReady === 'undefined' ? false : true");
    return value === true;
  }, 20000, "viewerReady");
  await evaluate(send, "viewerReady.then(() => true)", 30000);
  const errors = await evaluate(send, "window.__viewerErrors || []");
  if (errors && errors.length) throw new Error(errors.join("\n"));
}

async function shot(send, name) {
  const path = join(SHOTS, name);
  mkdirSync(join(path, ".."), { recursive: true });
  const result = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(path, Buffer.from(result.data, "base64"));
  log("shot", path);
  return path;
}

async function videoBox(send) {
  return evaluate(send, `(() => {
    const video = document.querySelector("video");
    const rect = video.getBoundingClientRect();
    return {
      x: rect.x,
      y: rect.y,
      w: rect.width,
      h: rect.height,
      vw: window.innerWidth,
      vh: window.innerHeight,
    };
  })()`);
}

function coveredFraction(pngPath, box) {
  const script = [
    "import json, sys",
    "import cv2",
    "import numpy as np",
    "path, x, y, w, h, vw, vh = sys.argv[1:]",
    "im = cv2.imread(path)",
    "ih, iw = im.shape[:2]",
    "sx = float(iw) / float(vw)",
    "sy = float(ih) / float(vh)",
    "left = max(0, min(iw - 1, int(float(x) * sx)))",
    "top = max(0, min(ih - 1, int(float(y) * sy)))",
    "right = max(left + 1, min(iw, int((float(x) + float(w)) * sx)))",
    "bottom = max(top + 1, min(ih, int((float(y) + float(h)) * sy)))",
    "flat = im[top:bottom, left:right].reshape(-1, 3).astype(int)",
    "near = (np.abs(flat - np.array([17, 17, 17]))).max(axis=1) <= 8",
    "print(json.dumps({'near111': round(float(near.mean()), 4), 'n': int(flat.shape[0])}))",
  ].join("\n");
  const result = spawnSync(PY, ["-c", script, pngPath, String(box.x), String(box.y), String(box.w), String(box.h), String(box.vw), String(box.vh)], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || "cover sample failed");
  return JSON.parse(result.stdout);
}

async function assertIdlePaint(send) {
  const sample = await evaluate(send, `(() => new Promise((resolve) => {
    setTimeout(() => {
      const before = viewer.state();
      let draws = 0;
      const orig = viewer.draw.bind(viewer);
      viewer.draw = function () { draws += 1; return orig(); };
      setTimeout(() => {
        viewer.draw = orig;
        const canvas = document.querySelector("canvas.overlay");
        const ctx = canvas.getContext("2d");
        const hash = () => {
          const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
          let n = 0;
          for (let i = 0; i < data.length; i += 64) n = (n + data[i]) | 0;
          return n;
        };
        const left = hash();
        viewer.skeletonButton.click();
        const right = hash();
        viewer.skeletonButton.click();
        resolve({ draws, before, after: viewer.state(), skeletonDiff: left !== right });
      }, 1000);
    }, 600);
  }))()`, 8000);
  log("idle paint", JSON.stringify(sample));
  if (sample.draws > 2) fail(`idle paint ${sample.draws} draws in 1s`);
  if (!sample.before.paused || sample.before.mode !== "idle") fail(`idle paint not idle ${JSON.stringify(sample.before)}`);
  if (sample.after.frame !== sample.before.frame || sample.after.overlayFrame !== sample.before.overlayFrame) {
    fail(`idle paint moved frame ${sample.before.frame} to ${sample.after.frame}`);
  }
  if (!sample.skeletonDiff) fail("skeleton toggle did not change the overlay");
  return sample;
}

async function assertVideoShown(send, label) {
  const path = await shot(send, label);
  const cover = coveredFraction(path, await videoBox(send));
  log("cover", label, JSON.stringify(cover));
  if (!(cover.n > 1000)) fail(`${label} video sample too small ${cover.n}`);
  if (cover.near111 > 0.5) fail(`${label} composited video covered ${cover.near111}`);
  return cover;
}

function fileHash(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

const PAGE = String.raw`
function readCodedFrame() {
  const video = viewer.video;
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return -1;
  if (!readCodedFrame.canvas) readCodedFrame.canvas = document.createElement("canvas");
  const canvas = readCodedFrame.canvas;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  const origin = 32;
  const size = 40;
  const gap = 8;
  const cols = 6;
  const samples = [];
  for (let slot = 0; slot < 12; slot++) {
    const col = slot % cols;
    const row = (slot / cols) | 0;
    const x = origin + col * (size + gap) + (size >> 1);
    const y = origin + row * (size + gap) + (size >> 1);
    if (x >= w || y >= h) return -1;
    samples.push(ctx.getImageData(x, y, 1, 1).data[0]);
  }
  if (samples.some((value) => value >= 40 && value <= 200)) return -1;
  const bits = samples.map((value) => (value > 128 ? 1 : 0));
  let low = 0;
  let high = 0;
  for (let i = 0; i < 6; i++) {
    low |= bits[i] << i;
    high |= bits[6 + i] << i;
  }
  if (low !== high) return -1;
  return low;
}

function frameFromMedia(media) {
  const stamps = viewer.bundle.timestamps;
  let best = 0;
  let bestAbs = Infinity;
  for (let i = 0; i < stamps.length; i++) {
    const delta = Math.abs(stamps[i] - media);
    if (delta < bestAbs) {
      bestAbs = delta;
      best = i;
    }
  }
  return best;
}

globalThis.__c2 = {
  async seeks(frames) {
    const rows = [];
    for (const want of frames) {
      await viewer.seekFrame(want);
      const decoded = readCodedFrame();
      const st = viewer.state();
      rows.push({
        want,
        decoded,
        frame: st.frame,
        overlay: st.overlayFrame,
        paused: st.paused,
        agree: decoded === st.overlayFrame && st.overlayFrame === st.frame,
        hit: decoded === want && st.overlayFrame === want && st.frame === want,
      });
    }
    return rows;
  },
  async playback(rate, count) {
    viewer.leaveMoment();
    viewer.setRate(rate);
    await viewer.seekFrame(2);
    await viewer.video.play();
    const rows = [];
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        viewer.video.pause();
        resolve();
      }, 8000);
      const tick = () => {
        const decoded = readCodedFrame();
        const st = viewer.state();
        rows.push({
          decoded,
          frame: st.frame,
          overlay: st.overlayFrame,
          paused: st.paused,
          rate: st.playbackRate,
        });
        if (rows.length >= count) {
          clearTimeout(timer);
          viewer.video.pause();
          resolve();
          return;
        }
        viewer.video.requestVideoFrameCallback(tick);
      };
      viewer.video.requestVideoFrameCallback(tick);
    });
    const live = rows.filter((row) => row.decoded >= 0 && !row.paused);
    const within = live.filter((row) => Math.abs(row.decoded - row.frame) <= 1 && Math.abs(row.decoded - row.overlay) <= 1);
    return {
      n: rows.length,
      live: live.length,
      within: within.length,
      misses: live.filter((row) => Math.abs(row.decoded - row.frame) > 1 || Math.abs(row.decoded - row.overlay) > 1).slice(0, 6),
      paused: rows.filter((row) => row.paused).length,
      negative: rows.filter((row) => row.decoded < 0).length,
    };
  },
  async moment(id, index, key, slow) {
    viewer.leaveMoment();
    const rates = [];
    const trace = [];
    const timer = setInterval(() => {
      const st = viewer.state();
      const row = {
        rate: st.playbackRate,
        element: viewer.video.playbackRate,
        paused: st.paused,
        frame: st.frame,
        ready: viewer.segmentReady,
        speed: viewer.segment && viewer.segment.speed,
        kind: viewer.segment && viewer.segment.kind,
        show: st.segmentShow,
        time: Math.round(viewer.video.currentTime * 1000) / 1000,
      };
      rates.push(row);
      if (trace.length < 8 || rates.length % 20 === 0) trace.push(row);
    }, 5);
    viewer.playMoment(id, "near", index);
    const deadline = performance.now() + 20000;
    let freeze = null;
    while (performance.now() < deadline) {
      const st = viewer.state();
      if (st.mode === "moment" && st.frame === key && st.overlayFrame === key && st.playbackRate === 0 && st.paused && st.segmentShow === "issue") {
        freeze = st;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    clearInterval(timer);
    if (freeze) await new Promise((resolve) => setTimeout(resolve, 400));
    const after = viewer.state();
    const slowHits = rates.filter((row) => !row.paused && row.ready && Math.abs(row.element - slow) <= slow * 0.1 && Math.abs(row.rate - slow) <= slow * 0.1);
    return {
      freeze: after,
      reached: !!freeze,
      still: after.frame === key && after.overlayFrame === key && after.playbackRate === 0 && after.paused,
      slowHits: slowHits.length,
      slowSample: slowHits[0] || null,
      minElement: rates.reduce((min, row) => Math.min(min, row.element), 99),
      trace,
      layout: measureLayout(),
      errors: window.__viewerErrors || [],
    };
  },
  async compare() {
    const button = document.querySelector("button.compare");
    if (!button || button.disabled) return { error: "compare disabled", state: viewer.state() };
    const samples = [];
    const started = performance.now();
    const note = (media) => {
      const st = viewer.state();
      samples.push({
        t: Math.round(performance.now() - started),
        media,
        presented: frameFromMedia(media),
        frame: st.frame,
        overlay: st.overlayFrame,
      });
    };
    const watch = (_now, meta) => {
      note(meta.mediaTime);
      setTimeout(() => note(meta.mediaTime), 50);
      if (samples.length < 24) viewer.video.requestVideoFrameCallback(watch);
    };
    viewer.video.requestVideoFrameCallback(watch);
    button.click();
    const deadline = performance.now() + 8000;
    while (performance.now() < deadline) {
      const st = viewer.state();
      if (st.frame === 48 && st.overlayFrame === 48 && st.playbackRate === 0 && st.paused && st.segmentShow === "best") {
        await new Promise((resolve) => setTimeout(resolve, 400));
        return { state: viewer.state(), layout: measureLayout(), text: document.querySelector(".panel").innerText, samples };
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    return { error: "compare freeze missed", state: viewer.state(), samples };
  },
  async cancel() {
    await viewer.seekFrame(0);
    const pending = viewer.seekFrame(40);
    await new Promise((resolve) => setTimeout(resolve, 20));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await pending;
    await new Promise((resolve) => setTimeout(resolve, 200));
    const decoded = readCodedFrame();
    const st = viewer.state();
    return { decoded, frame: st.frame, overlay: st.overlayFrame, mode: st.mode };
  },
  async showMark(id, player, frame) {
    viewer.leaveMoment();
    await viewer.seekFrame(frame);
    viewer.focusFinding(id, player);
    await new Promise((resolve) => setTimeout(resolve, 200));
    const st = viewer.state();
    const drawn = viewer.drawn();
    return {
      frame: st.frame,
      overlay: st.overlayFrame,
      kinds: drawn.pills.map((pill) => pill.kind),
    };
  },
  async keepFrame() {
    await viewer.seekFrame(17);
    viewer.focusFinding("technique.stay_lower", "near");
    const before = viewer.state();
    await viewer.update(new URLSearchParams(location.search).get("bundle"));
    const paused = viewer.state();
    viewer.setRate(1);
    await viewer.video.play();
    await new Promise((resolve) => setTimeout(resolve, 80));
    const runningBefore = viewer.state();
    await viewer.update(new URLSearchParams(location.search).get("bundle"));
    const running = viewer.state();
    viewer.video.pause();
    return { before, paused, runningBefore, running };
  },
  callbacks() {
    let refine = null;
    let exercise = null;
    viewer.onRefine = (window) => { refine = window; };
    viewer.onExercise = (id) => { exercise = id; };
    const refineButton = document.querySelector("button.warn");
    const chip = document.querySelector("button.chip");
    if (refineButton) refineButton.click();
    if (chip) chip.click();
    const panel = document.querySelector(".panel");
    const style = getComputedStyle(panel);
    return {
      refine,
      exercise,
      position: style.position,
      stripInside: !!document.querySelector(".video-col .strip"),
      text: panel.innerText,
    };
  },
};

function measureLayout() {
  const rect = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height };
  };
  const clip = (a, b) => {
    if (!a || !b) return null;
    const left = Math.max(a.left, b.left);
    const top = Math.max(a.top, b.top);
    const right = Math.min(a.right, b.right);
    const bottom = Math.min(a.bottom, b.bottom);
    if (right - left <= 0 || bottom - top <= 0) return null;
    return { left, top, right, bottom, width: right - left, height: bottom - top };
  };
  const stage = rect(document.querySelector(".stage"));
  const video = rect(document.querySelector("video"));
  const panel = rect(document.querySelector(".panel"));
  const canvas = document.querySelector("canvas.overlay");
  const canvasRect = rect(canvas);
  const drawn = viewer.drawn();
  const map = (r) => {
    if (!canvasRect || !drawn.canvas.width || !drawn.canvas.height) return null;
    const sx = canvasRect.width / drawn.canvas.width;
    const sy = canvasRect.height / drawn.canvas.height;
    return {
      kind: r.kind || r.role || "",
      left: canvasRect.left + r.x * sx,
      top: canvasRect.top + r.y * sy,
      right: canvasRect.left + (r.x + r.w) * sx,
      bottom: canvasRect.top + (r.y + r.h) * sy,
      width: r.w * sx,
      height: r.h * sy,
    };
  };
  return {
    stage,
    videoRaw: video,
    videoVisible: clip(video, stage),
    panel,
    canvas: canvasRect,
    minimap: rect(document.querySelector("canvas.minimap")),
    pills: drawn.pills.map(map),
    boxes: drawn.boxes.map(map),
    captions: drawn.captions,
    kinds: drawn.pills.map((pill) => pill.kind),
  };
}
`;

function overlaps(a, b) {
  if (!a || !b || a.width <= 0 || b.width <= 0) return false;
  return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
}

function checkLayout(size, finding, layout, expectedKinds) {
  const problems = [];
  if (!layout || !layout.panel || !layout.stage || !layout.videoVisible) {
    problems.push("missing rect");
  } else {
    if (overlaps(layout.panel, layout.stage)) problems.push("panel intersects stage");
    if (overlaps(layout.panel, layout.videoVisible)) problems.push("panel intersects visible video");
  }
  const kinds = new Set(layout && layout.kinds || []);
  for (const kind of expectedKinds) {
    if (!kinds.has(kind)) problems.push(`missing pill ${kind}`);
  }
  const pills = (layout && layout.pills) || [];
  const boxes = (layout && layout.boxes) || [];
  for (const pill of pills) {
    for (const box of boxes) {
      if (overlaps(pill, box)) problems.push(`pill ${pill.kind} intersects ${box.kind}`);
    }
  }
  if (layout && layout.minimap && layout.panel && overlaps(layout.minimap, layout.videoVisible)) {
    problems.push("minimap intersects visible video");
  }
  if ((finding === "footwork.recovery" || finding === "tactics.base_discipline") && layout && !layout.minimap) {
    problems.push("missing minimap");
  }
  return problems.map((problem) => `${size[0]}x${size[1]} ${finding}: ${problem}`);
}

async function main() {
  await acquireLock();
  mkdirSync(SHOTS, { recursive: true });
  server = await startServer();
  log("server", server.port);
  const port = debugPort();
  log("edge port", port);
  edge = startEdge(port, 1280, 720);
  const target = await waitFor(async () => {
    const res = await fetch(`http://127.0.0.1:${port}/json`);
    return (await res.json()).find((item) => item.type === "page");
  }, 90000, "page target");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve);
    ws.addEventListener("error", reject);
  });
  const send = attach(ws);
  await send("Page.enable");
  await send("Runtime.enable");
  await setSize(send, 1280, 720);

  await openViewer(send, server.port, "/work/C2/fixture/sync.json");
  await evaluate(send, PAGE, 10000);
  await evaluate(send, "viewer.seekFrame(1)", 15000);
  summary.cover_sync = await assertVideoShown(send, "cover-sync.png");
  const frames = seekFrames(50);
  log("paused seeks", frames.join(","));
  const seeks = await evaluate(send, `__c2.seeks(${JSON.stringify(frames)})`, 180000);
  const agreed = seeks.filter((row) => row.agree).length;
  const hit = seeks.filter((row) => row.hit).length;
  summary.paused_match = agreed / seeks.length;
  log("paused", summary.paused_match, "hit", hit, "/", seeks.length);
  if (summary.paused_match !== 1) {
    fail(`paused agreement ${agreed}/${seeks.length}`);
    log(JSON.stringify(seeks.filter((row) => !row.agree).slice(0, 8)));
  }
  if (hit !== seeks.length) {
    fail(`paused seek missed ${seeks.length - hit}`);
    log(JSON.stringify(seeks.filter((row) => !row.hit).slice(0, 8)));
  }

  const warm = await evaluate(send, `__c2.seeks(${JSON.stringify(frames)})`, 180000);
  const warmHit = warm.filter((row) => row.hit).length;
  const warmAgree = warm.filter((row) => row.agree).length;
  summary.warm_match = warmAgree / warm.length;
  log("warm", summary.warm_match, "hit", warmHit, "/", warm.length);
  if (warmAgree !== warm.length) {
    fail(`warm agreement ${warmAgree}/${warm.length}`);
    log(JSON.stringify(warm.filter((row) => !row.agree).slice(0, 8)));
  }
  if (warmHit !== warm.length) {
    fail(`warm seek missed ${warm.length - warmHit}`);
    log(JSON.stringify(warm.filter((row) => !row.hit).slice(0, 8)));
  }

  const cancelled = await evaluate(send, "__c2.cancel()", 30000);
  log("cancel", JSON.stringify(cancelled));
  summary.cancel = cancelled;
  if (cancelled.mode !== "idle") fail(`cancel mode ${cancelled.mode}`);
  if (cancelled.decoded < 0) fail("cancel pixels unreadable");
  if (cancelled.overlay !== cancelled.decoded || cancelled.frame !== cancelled.decoded) {
    fail(`cancel overlay ${cancelled.overlay} pixels ${cancelled.decoded}`);
  }
  if (cancelled.decoded === 40 && cancelled.overlay === 40) {
    log("cancel landed on 40 before Escape; overlay still matches pixels");
  }

  const playRates = [];
  for (const rate of [1, 0.25]) {
    const sample = await evaluate(send, `__c2.playback(${rate}, ${rate === 1 ? 30 : 24})`, 30000);
    log("playback", rate, JSON.stringify(sample));
    if (sample.live < 20) fail(`playback ${rate}x only ${sample.live} live samples`);
    const ratio = sample.live ? sample.within / sample.live : 0;
    playRates.push(ratio);
    if (ratio < 0.99) fail(`playback ${rate}x within one ${sample.within}/${sample.live}`);
  }
  summary.playback_within_one = Math.min(...playRates);

  await setSize(send, 1280, 720);
  await openViewer(send, server.port, "/work/C2/fixture/bundle.json");
  await evaluate(send, PAGE, 10000);
  const kept = await evaluate(send, "__c2.keepFrame()", 30000);
  log("update", JSON.stringify(kept));
  if (kept.paused.frame !== 17 || kept.paused.overlayFrame !== 17 || kept.paused.findingId !== "technique.stay_lower" || kept.paused.player !== "near" || kept.paused.paused !== true) {
    fail(`update paused kept ${JSON.stringify(kept.paused)}`);
  }
  if (kept.running.paused !== false || kept.running.findingId !== "technique.stay_lower" || kept.running.frame !== kept.runningBefore.frame) {
    fail(`update running kept ${JSON.stringify({ before: kept.runningBefore, after: kept.running })}`);
  }

  const hooks = await evaluate(send, "__c2.callbacks()");
  log("callbacks", JSON.stringify(hooks));
  if (!hooks.refine || hooks.refine.start_frame !== 0 || hooks.refine.end_frame !== 12 || hooks.refine.reason !== "fixture") {
    fail(`onRefine ${JSON.stringify(hooks.refine)}`);
  }
  if (hooks.exercise !== "wall_sit") fail(`onExercise ${hooks.exercise}`);
  if (hooks.position === "absolute") fail("panel is position absolute");
  if (!hooks.stripInside) fail("strip is outside the video column");
  for (const label of [
    "Fixture",
    "Your number",
    "Practice guide",
    "Your best",
    "A coaching guide, not a tested personal target.",
    "projected",
    "rough estimate from the 2D picture",
    "Preview estimate",
    "Only a few arm swings were found, so averages are rough.",
    "Compare with your best",
    "Wall sit",
    "high",
  ]) {
    if (!hooks.text.includes(label)) fail(`panel missing ${label}`);
  }
  if (hooks.text.includes("few_swings") || hooks.text.includes("wall_sit")) fail("panel still prints raw ids");
  if (hooks.text.includes("The video repeats some frames") || hooks.text.includes("Only a few shots were found")) {
    fail("panel still prints the old warning");
  }
  const previewCount = (hooks.text.match(/Preview estimate/g) || []).length;
  if (previewCount !== 1) fail(`preview labels ${previewCount}`);

  const layoutDump = [];
  let layoutFailures = 0;
  const note = (problems) => {
    for (const problem of problems) {
      layoutFailures += 1;
      fail(problem);
    }
  };

  const issue = await evaluate(send, `__c2.moment("technique.stay_lower", 0, 18, ${SLOW})`, 40000);
  log("freeze", JSON.stringify({
    reached: issue.reached,
    still: issue.still,
    freeze: issue.freeze,
    slowHits: issue.slowHits,
    slowSample: issue.slowSample,
    minElement: issue.minElement,
    kinds: issue.layout && issue.layout.kinds,
    captions: issue.layout && issue.layout.captions,
    errors: issue.errors,
  }));
  summary.freeze_frame = issue.freeze ? issue.freeze.frame : null;
  if (!issue.reached || !issue.still || issue.freeze.frame !== 18 || issue.freeze.overlayFrame !== 18 || issue.freeze.playbackRate !== 0) {
    fail(`freeze ${JSON.stringify(issue.freeze)}`);
  }
  if (issue.slowHits < 1) fail(`slow rate not observed, min element ${issue.minElement}`);
  note(checkLayout([1280, 720], "technique.stay_lower", issue.layout, MARK_KINDS));
  layoutDump.push({ size: [1280, 720], finding: "technique.stay_lower", layout: issue.layout });
  await shot(send, "fixture-technique.png");

  const compared = await evaluate(send, "__c2.compare()", 20000);
  const late = (compared.samples || []).find((row) => row.presented === 48);
  const caught = (compared.samples || []).find((row) => late && row.t >= late.t + 40 && row.overlay === 48 && row.frame === 48);
  const compareLag = late && caught ? caught.t - late.t : null;
  log("compare", JSON.stringify({
    error: compared.error,
    state: compared.state,
    kinds: compared.layout && compared.layout.kinds,
    lagMs: compareLag,
    firstAt48: late || null,
  }));
  summary.compare_frame = compared.state ? compared.state.frame : null;
  summary.compare_lag_ms = compareLag;
  if (!compared.state || compared.state.frame !== 48 || compared.state.overlayFrame !== 48 || compared.state.playbackRate !== 0 || compared.state.segmentShow !== "best") {
    fail(`compare ${JSON.stringify(compared.state || compared.error)}`);
  }
  if (compareLag == null || compareLag > 100) fail(`compare overlay lagged ${compareLag}ms after media frame 48`);
  if (compared.text && !compared.text.includes("Your best one")) fail("compare caption missing");
  if (compared.layout) {
    note(checkLayout([1280, 720], "technique.stay_lower best", compared.layout, ["dot", "label"]));
    layoutDump.push({ size: [1280, 720], finding: "technique.stay_lower best", layout: compared.layout });
  }
  await shot(send, "compare.png");

  for (const [id, key, kinds, file] of [
    ["footwork.recovery", 22, ["dot", "line"], "footwork.png"],
    ["tactics.base_discipline", 26, ["dot", "line"], "tactics.png"],
  ]) {
    const result = await evaluate(send, `__c2.moment(${JSON.stringify(id)}, 0, ${key}, ${SLOW})`, 40000);
    log(id, JSON.stringify({
      reached: result.reached,
      frame: result.freeze && result.freeze.frame,
      kinds: result.layout && result.layout.kinds,
      minimap: !!(result.layout && result.layout.minimap),
    }));
    if (!result.reached || result.freeze.frame !== key || result.freeze.overlayFrame !== key || result.freeze.playbackRate !== 0) {
      fail(`${id} freeze ${JSON.stringify(result.freeze)}`);
    }
    note(checkLayout([1280, 720], id, result.layout, kinds));
    layoutDump.push({ size: [1280, 720], finding: id, layout: result.layout });
    await shot(send, file);
    const disabled = await evaluate(send, "document.querySelector('button.compare').disabled");
    if (disabled !== true) fail(`${id} compare should be disabled`);
  }

  for (const size of SIZES.slice(1)) {
    await setSize(send, size[0], size[1]);
    await evaluate(send, "new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))");
    for (const [id, key, kinds] of [
      ["technique.stay_lower", 18, MARK_KINDS],
      ["footwork.recovery", 22, ["dot", "line"]],
      ["tactics.base_discipline", 26, ["dot", "line"]],
    ]) {
      const result = await evaluate(send, `__c2.moment(${JSON.stringify(id)}, 0, ${key}, ${SLOW})`, 40000);
      log("layout", size.join("x"), id, result.reached, result.layout && result.layout.kinds, JSON.stringify(result.trace || []).slice(0, 700));
      if (!result.reached) fail(`${size.join("x")} ${id} did not freeze`);
      note(checkLayout(size, id, result.layout, kinds));
      layoutDump.push({ size, finding: id, layout: result.layout });
    }
  }

  summary.layout_failures = layoutFailures;
  writeFileSync(LAYOUT, JSON.stringify({ failures: layoutFailures, checks: layoutDump }, null, 2));
  log("layout failures", layoutFailures);

  await openViewer(send, server.port, "/work/C2/real/bundle.json");
  await evaluate(send, PAGE, 10000);
  const realText = await evaluate(send, `document.querySelector(".panel").innerText`);
  if (!realText.includes("Some pose estimates repeat, so speed estimates skip those repeats.")) {
    fail("real panel missing repeated pose warning");
  }
  if (realText.includes("The video repeats some frames") || realText.includes("Only a few shots were found")) {
    fail("real panel still prints the old warning");
  }
  summary.idle_paint = await assertIdlePaint(send);
  const realMarks = [
    ["dot", "footwork.recovery", "near", 14],
    ["label", "footwork.split_step", "near", 111],
    ["line", "technique.trunk_lean", "far", 9],
    ["bracket", "technique.lunge_depth", "far", 580],
    ["band", "technique.stay_lower", "far", 680],
    ["angle", "technique.knee_bend", "far", 668],
    ["trail", "contact.hand_speed", "far", 787],
  ];
  const realRows = [];
  for (const [kind, id, player, frame] of realMarks) {
    const shown = await evaluate(send, `__c2.showMark(${JSON.stringify(id)}, ${JSON.stringify(player)}, ${frame})`, 30000);
    const path = await shot(send, `real/${kind}.png`);
    const cover = coveredFraction(path, await videoBox(send));
    log("cover", kind, JSON.stringify(cover));
    if (!(cover.n > 1000)) fail(`real ${kind} video sample too small ${cover.n}`);
    if (cover.near111 > 0.5) fail(`real ${kind} composited video covered ${cover.near111}`);
    realRows.push({ kind, id, player, frame, shown, sha256: fileHash(path), near111: cover.near111 });
    log("real mark", kind, JSON.stringify(shown));
    if (!shown.kinds.includes(kind)) fail(`real ${kind} not drawn ${JSON.stringify(shown)}`);
    if (shown.overlay !== frame && Math.abs(shown.overlay - frame) > 1) {
      fail(`real ${kind} overlay ${shown.overlay} wanted ${frame}`);
    }
  }
  const realHashes = realRows.map((row) => row.sha256);
  if (new Set(realHashes).size !== realHashes.length) fail("real mark screenshots are copies");
  summary.real_marks = realRows.map((row) => ({ kind: row.kind, overlay: row.shown.overlay, kinds: row.shown.kinds }));

  mkdirSync(join(FRAME_LOG, ".."), { recursive: true });
  writeFileSync(FRAME_LOG, JSON.stringify({
    cold: seeks,
    warm,
    cancel: cancelled,
    compare_lag_ms: summary.compare_lag_ms,
    playback_within_one: summary.playback_within_one,
    real_marks: summary.real_marks,
  }, null, 2));
  log("frame log", FRAME_LOG);
  ws.close();
}

try {
  await main();
} catch (error) {
  fail(error && error.stack || String(error));
} finally {
  if (edge) edge.kill();
  if (server) server.child.kill();
  releaseLock();
  await sleep(300);
}

console.log(JSON.stringify(summary));
if (failures.length) {
  log("failed checks", failures.length);
  process.exit(1);
}
