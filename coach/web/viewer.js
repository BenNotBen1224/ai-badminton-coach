const JOINTS = 17;
const NOSE = 0;
const LEFT_EYE = 1;
const RIGHT_EYE = 2;
const LEFT_ANKLE = 15;
const RIGHT_ANKLE = 16;
const CONFIDENCE = [
  ["high", 0.75],
  ["medium", 0.5],
  ["low", 0],
];

const WARNING_LABELS = {
  doubles: "This looks like a doubles game. Coaching follows two players only, so some numbers may be off.",
  high_fps: "This video has a high frame rate, so analysis can take longer than usual.",
  no_court: "The court lines were not found, so court positions and distances are skipped.",
  repeated_frames: "Some pose estimates repeat, so speed estimates skip those repeats.",
  few_swings: "Only a few arm swings were found, so averages are rough.",
  partial_track: "A player was visible for only part of the clip, so some numbers use fewer frames.",
  unresolved_recovery: "A hip drop had no measured return to the ready hips from before the drop, so that unfinished move was not scored.",
};

const EXERCISE_NAMES = {
  wall_sit: "Wall sit",
  sit_low_hold: "Sit-low hold",
  goblet_squat: "Goblet squat",
  forward_lunge_reach: "Forward lunge with reach",
  pause_lunge_recover: "Pause lunge and recover",
  lateral_lunge: "Lateral lunge",
  split_step_hops: "Split-step hops",
  jump_rope: "Jump rope",
  six_corner_shadow: "Six-corner shadow",
  skater_hops: "Skater hops",
  calf_raises: "Calf raises",
  single_leg_calf_raise: "Single-leg calf raise",
  squat_jumps: "Squat jumps",
  front_plank: "Front plank",
  side_plank: "Side plank",
  standing_trunk_turn: "Standing trunk turn",
  band_trunk_rotation: "Band trunk rotation",
  wall_angels: "Wall angels",
};

function warningLabel(code) {
  return WARNING_LABELS[code] || code;
}

function exerciseName(id) {
  return EXERCISE_NAMES[id] || String(id).replaceAll("_", " ");
}

const MARK_DRAW = {
  dot(ctx, pts, look) {
    ctx.save();
    ctx.fillStyle = look.fill;
    ctx.globalAlpha *= 0.35;
    ctx.beginPath();
    ctx.arc(pts[0].x, pts[0].y, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = look.fill;
    ctx.beginPath();
    ctx.arc(pts[0].x, pts[0].y, 5, 0, Math.PI * 2);
    ctx.fill();
  },
  label() {},
  line(ctx, pts, look) {
    stroke(ctx, look, () => {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.stroke();
    });
  },
  bracket(ctx, pts, look) {
    stroke(ctx, look, () => {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      cap(ctx, pts[0], pts[1]);
      cap(ctx, pts[1], pts[0]);
      ctx.stroke();
    });
  },
  band(ctx, pts, look) {
    const x = Math.min(pts[0].x, pts[1].x);
    const y = Math.min(pts[0].y, pts[1].y);
    const w = Math.abs(pts[1].x - pts[0].x);
    const h = Math.abs(pts[1].y - pts[0].y);
    ctx.save();
    ctx.globalAlpha *= 0.28;
    ctx.fillStyle = look.fill;
    ctx.fillRect(x, y, w, h);
    ctx.restore();
    stroke(ctx, look, () => ctx.strokeRect(x, y, w, h));
  },
  angle(ctx, pts, look) {
    const [a, vertex, b] = pts;
    stroke(ctx, look, () => {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(vertex.x, vertex.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });
    const a1 = Math.atan2(a.y - vertex.y, a.x - vertex.x);
    const a2 = Math.atan2(b.y - vertex.y, b.x - vertex.x);
    let sweep = a2 - a1;
    while (sweep > Math.PI) sweep -= Math.PI * 2;
    while (sweep < -Math.PI) sweep += Math.PI * 2;
    const reach = Math.min(
      36,
      Math.hypot(a.x - vertex.x, a.y - vertex.y) * 0.4,
      Math.hypot(b.x - vertex.x, b.y - vertex.y) * 0.4,
    );
    if (reach < 4) return;
    stroke(ctx, look, () => {
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, reach, a1, a1 + sweep, sweep < 0);
      ctx.stroke();
    });
  },
  trail(ctx, pts, look) {
    if (pts.length < 2) return;
    const saved = ctx.globalAlpha;
    for (let i = 1; i < pts.length; i++) {
      const fade = 0.15 + 0.85 * (i / (pts.length - 1));
      ctx.globalAlpha = saved * fade;
      stroke(ctx, look, () => {
        ctx.beginPath();
        ctx.moveTo(pts[i - 1].x, pts[i - 1].y);
        ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      });
    }
    ctx.globalAlpha = saved;
    ctx.fillStyle = look.fill;
    ctx.beginPath();
    ctx.arc(pts[pts.length - 1].x, pts[pts.length - 1].y, 5, 0, Math.PI * 2);
    ctx.fill();
  },
};

function stroke(ctx, look, draw) {
  ctx.save();
  ctx.strokeStyle = look.stroke;
  ctx.lineWidth = look.width;
  ctx.lineCap = "round";
  if (look.dash) ctx.setLineDash([7, 5]);
  draw();
  ctx.restore();
}

function cap(ctx, end, other) {
  const dx = end.x - other.x;
  const dy = end.y - other.y;
  const len = Math.hypot(dx, dy) || 1;
  const px = (-dy / len) * 8;
  const py = (dx / len) * 8;
  ctx.moveTo(end.x - px, end.y - py);
  ctx.lineTo(end.x + px, end.y + py);
}

function frameAt(stamps, fps, n, mediaTime) {
  if (!n) return 0;
  // Edge reports the presented time a fraction of a microsecond before the frame stamp.
  const time = mediaTime + Math.min(0.001, 0.25 / fps);
  if (!(time > stamps[0])) return 0;
  const lastEnd = stamps[n - 1] + 1 / fps;
  if (time >= lastEnd) return n - 1;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (stamps[mid] <= time) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function seekTime(stamps, fps, n, frame, frac) {
  const start = stamps[frame];
  const end = frame + 1 < n ? stamps[frame + 1] : start + 1 / fps;
  return start + Math.max(end - start, 1e-6) * frac;
}

function confidenceWord(value) {
  for (const [word, min] of CONFIDENCE) {
    if (value >= min) return word;
  }
  return "low";
}

function formatNum(value) {
  if (value == null || Number.isNaN(Number(value))) return "";
  return String(Math.round(Number(value) * 1000) / 1000);
}

export function normalizeBundle(bundle) {
  if (bundle == null || typeof bundle !== "object" || Array.isArray(bundle)) {
    throw new Error("bundle is missing schema_version");
  }
  const version = bundle.schema_version;
  if (version !== 2 && version !== 3) {
    throw new Error(`schema_version must be 2 or 3, got ${version}`);
  }
  const next = { ...bundle };
  if (Object.prototype.hasOwnProperty.call(bundle, "findings")) {
    next.findings = normalizeFindings(bundle.findings, version);
  }
  if (Object.prototype.hasOwnProperty.call(bundle, "metrics")) {
    next.metrics = normalizeMetricGroup(bundle.metrics, version);
  }
  return next;
}

function normalizeFindings(findings, version) {
  if (!Array.isArray(findings)) throw new Error("findings must be a list");
  return findings.map((finding) => {
    if (finding == null || typeof finding !== "object" || Array.isArray(finding)) {
      throw new Error("finding must be an object");
    }
    const next = { ...finding };
    if (Object.prototype.hasOwnProperty.call(finding, "metrics")) {
      if (!Array.isArray(finding.metrics)) throw new Error("finding metrics must be a list");
      next.metrics = finding.metrics.map((metric) => normalizeMetric(metric, version));
      requireUnavailableFinding(next);
    }
    return next;
  });
}

function normalizeMetricGroup(metrics, version) {
  if (Array.isArray(metrics)) return metrics.map((metric) => normalizeMetric(metric, version));
  if (metrics == null || typeof metrics !== "object") throw new Error("metrics must be a list");
  const next = {};
  for (const key of Object.keys(metrics)) {
    const rows = metrics[key];
    if (!Array.isArray(rows)) throw new Error("metrics must be a list");
    next[key] = rows.map((metric) => normalizeMetric(metric, version));
  }
  return next;
}

function normalizeMetric(metric, version) {
  if (metric == null || typeof metric !== "object" || Array.isArray(metric)) {
    throw new Error("metric must be an object");
  }
  const name = metric.name == null ? "?" : metric.name;
  const nObs = Object.prototype.hasOwnProperty.call(metric, "n_obs") ? metric.n_obs : 1;
  let next;
  if (version === 3) {
    next = { ...metric };
  } else {
    if (!Object.prototype.hasOwnProperty.call(metric, "value")) {
      throw new Error(`metric ${name} is missing value`);
    }
    if (!isZeroInt(nObs)) {
      if (metric.value == null) throw new Error(`metric ${name} value is null but n_obs is not 0`);
      next = { ...metric };
    } else {
      const confidencePresent = Object.prototype.hasOwnProperty.call(metric, "confidence");
      if (!confidencePresent || !isZeroReal(metric.confidence) || metric.best != null) {
        throw new Error(`metric ${name} has n_obs 0 but confidence or best is set; refusing to guess`);
      }
      if (metric.value != null && !isFiniteReal(metric.value)) {
        throw new Error(`metric ${name} has a contradictory legacy value ${metric.value}`);
      }
      next = { ...metric, value: null };
    }
  }
  requireReading(next);
  return next;
}

function requireReading(metric) {
  const name = metric.name == null ? "?" : metric.name;
  if (!Object.prototype.hasOwnProperty.call(metric, "value")) {
    throw new Error(`metric ${name} is missing value`);
  }
  const nObs = Object.prototype.hasOwnProperty.call(metric, "n_obs") ? metric.n_obs : 1;
  if (!isInteger(nObs) || nObs < 0) {
    throw new Error(`metric ${name} n_obs must be an integer >= 0, got ${nObs}`);
  }
  if (!isFiniteReal(metric.confidence) || metric.confidence < 0 || metric.confidence > 1) {
    throw new Error(`metric ${name} confidence must be in [0, 1]`);
  }
  if (metric.best != null && !isFiniteReal(metric.best)) {
    throw new Error(`metric ${name} best must be a finite number`);
  }
  if (metric.value == null) {
    if (nObs !== 0) throw new Error(`metric ${name} value is missing but n_obs is ${nObs}`);
    if (metric.confidence !== 0) throw new Error(`metric ${name} value is missing but confidence is ${metric.confidence}`);
    if (metric.best != null) throw new Error(`metric ${name} value is missing but best is set`);
    return;
  }
  if (!isFiniteReal(metric.value)) throw new Error(`metric ${name} value must be a finite number`);
  if (nObs <= 0) throw new Error(`metric ${name} has a value but n_obs is ${nObs}`);
}

function requireUnavailableFinding(finding) {
  const metrics = finding.metrics;
  if (!Array.isArray(metrics) || metrics.length === 0) return;
  if (!metrics.every((item) => item.value == null)) return;
  const id = finding.id == null ? "?" : finding.id;
  if (finding.severity !== "info") {
    throw new Error(`finding ${id} is unavailable, so it must be info`);
  }
  const score = Object.prototype.hasOwnProperty.call(finding, "score") ? finding.score : 0;
  if (typeof score !== "number" || !Number.isFinite(score) || score !== 0) {
    throw new Error(`finding ${id} is unavailable, so its score must be 0`);
  }
  if (!Array.isArray(finding.moments)) {
    throw new Error(`finding ${id} is unavailable, so moments must be a list`);
  }
  if (finding.moments.length > 0) {
    throw new Error(`finding ${id} is unavailable, so it must not have a moment`);
  }
}

function isInteger(value) {
  return typeof value === "number" && Number.isInteger(value);
}

function isZeroInt(value) {
  return isInteger(value) && value === 0;
}

function isFiniteReal(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isZeroReal(value) {
  return isFiniteReal(value) && value === 0;
}

function bytesOf(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function decodeFloats(packed) {
  const bytes = bytesOf(packed.b64);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = packed.shape.reduce((a, b) => a * b, 1);
  const scale = packed.scale == null ? 1 : packed.scale;
  const unknown = packed.unknown;
  const out = new Float32Array(count);
  if (packed.dtype === "int16") {
    for (let i = 0; i < count; i++) {
      const v = view.getInt16(i * 2, true);
      out[i] = unknown != null && v === unknown ? NaN : v * scale;
    }
    return out;
  }
  for (let i = 0; i < count; i++) {
    const v = view.getUint8(i);
    out[i] = unknown != null && v === unknown ? NaN : v * scale;
  }
  return out;
}

function overlap(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function inflate(rect, pad) {
  return { x: rect.x - pad, y: rect.y - pad, w: rect.w + pad * 2, h: rect.h + pad * 2 };
}

function gapTo(rect, point) {
  const dx = point.x < rect.x ? rect.x - point.x : point.x > rect.x + rect.w ? point.x - (rect.x + rect.w) : 0;
  const dy = point.y < rect.y ? rect.y - point.y : point.y > rect.y + rect.h ? point.y - (rect.y + rect.h) : 0;
  return Math.hypot(dx, dy);
}

function blocked(rect, boxes, faces, taken) {
  if (boxes.some((box) => overlap(rect, box))) return true;
  if (taken.some((box) => overlap(inflate(rect, 3), box))) return true;
  return faces.some((face) => overlap(rect, { x: face.x - 5, y: face.y - 5, w: 10, h: 10 }));
}

function placeLabel(anchor, size, boxes, faces, taken, bounds) {
  const { w, h } = size;
  const found = [];
  const push = (x, y) => {
    if (x < 1 || y < 1 || x + w > bounds.w - 1 || y + h > bounds.h - 1) return;
    found.push({ x, y, w, h });
  };
  push(anchor.x + 8, anchor.y - h / 2);
  push(anchor.x - w - 8, anchor.y - h / 2);
  push(anchor.x - w / 2, anchor.y - h - 8);
  push(anchor.x - w / 2, anchor.y + 8);
  const dirs = [
    [1, 0], [-1, 0], [0, -1], [0, 1],
    [0.7, -0.7], [-0.7, -0.7], [0.7, 0.7], [-0.7, 0.7],
    [1, -0.35], [-1, -0.35], [0.35, -1], [-0.35, -1],
  ];
  for (const gap of [16, 28, 44, 64, 88, 116, 148]) {
    for (const [dx, dy] of dirs) {
      const len = Math.hypot(dx, dy) || 1;
      push(anchor.x + (dx / len) * (gap + w / 2) - w / 2, anchor.y + (dy / len) * (gap + h / 2) - h / 2);
    }
  }
  for (const rect of found) {
    if (!blocked(rect, boxes, faces, taken)) return rect;
  }
  return null;
}

function anchorOf(kind, pts) {
  if (kind === "dot" || kind === "label") return pts[0];
  if (kind === "line" || kind === "bracket") {
    return { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
  }
  if (kind === "band") return pts[0];
  if (kind === "angle") return pts[1];
  return pts[pts.length - 1];
}

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function button(className, text) {
  const el = node("button", className, text);
  el.type = "button";
  return el;
}

export class Viewer {
  onRefine = () => {};
  onExercise = () => {};
  onSelect = () => {};

  constructor() {
    this.token = 0;
    this.seekSeq = 0;
    this.seekChain = Promise.resolve();
    this.mode = "idle";
    this.frame = 0;
    this.overlayFrame = 0;
    this.paused = true;
    this.playbackRate = 1;
    this.selectedRate = 1;
    this.findingId = null;
    this.player = null;
    this.momentIndex = 0;
    this.segment = null;
    this.segmentShow = null;
    this.segmentIndex = -1;
    this.segmentReady = false;
    this.locking = false;
    this.lastMediaTime = 0;
    this.showPlayer = { near: true, far: true };
    this.showSkeleton = true;
    this.zoomForced = false;
    this.alive = false;
    this.pills = [];
    this.boxes = [];
    this.captions = [];
    this.playbackError = "";
    this.shownAt = new Map();
    this.poses = null;
    this.bundle = null;
    this.reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  async mount(element, bundleUrl) {
    this.root = element;
    this.root.classList.add("c2-viewer");
    this.root.replaceChildren();
    this.build(element);
    this.bind();
    this.alive = true;
    await this.load(bundleUrl, { selectFirst: true });
    this.armRvfc();
    this.fit();
    await this.land(0);
    this.renderPanel();
    this.draw();
  }

  async update(bundleUrl) {
    const keepFrame = this.frame;
    const keepId = this.findingId;
    const keepPlayer = this.player;
    const keepIndex = this.momentIndex;
    const running = !!(this.video && !this.video.paused);
    const rate = this.selectedRate;
    this.token += 1;
    this.mode = "idle";
    this.segment = null;
    this.segmentShow = null;
    this.segmentReady = false;
    this.playbackError = "";
    await this.load(bundleUrl, { selectFirst: false });
    if (this.find(keepId, keepPlayer)) {
      this.findingId = keepId;
      this.player = keepPlayer;
      this.momentIndex = keepIndex;
    } else {
      this.findingId = null;
      this.player = null;
      this.momentIndex = 0;
    }
    this.selectedRate = rate;
    await this.land(Math.max(0, Math.min(this.bundle.n_frames - 1, keepFrame)));
    if (running) {
      try { this.video.playbackRate = rate; } catch { /* keep the landed pause if the element rejects it */ }
      this.playbackRate = this.video.playbackRate;
      try {
        await this.video.play();
        this.paused = false;
      } catch (error) {
        this.paused = true;
        this.notePlaybackError(error);
        throw error;
      }
    }
    this.renderPanel();
    this.draw();
  }

  playMoment(id, player, index) {
    const finding = this.find(id, player);
    if (!finding || !finding.timelines[index]) return Promise.resolve(false);
    this.token += 1;
    const token = this.token;
    if (this.video) this.video.pause();
    this.findingId = id;
    this.player = player;
    this.momentIndex = index;
    this.mode = "moment";
    this.playbackError = "";
    this.notifySelect();
    this.renderPanel();
    return this.runFrom(0, token);
  }

  async seekFrame(frame) {
    this.leaveMoment();
    const clamped = Math.max(0, Math.min(this.bundle.n_frames - 1, frame | 0));
    const shown = await this.land(clamped);
    this.renderPanel();
    return shown;
  }

  focusFinding(id, player) {
    if (!this.find(id, player)) return;
    if (this.mode === "moment") this.leaveMoment();
    this.findingId = id;
    this.player = player;
    this.momentIndex = 0;
    this.notifySelect();
    this.renderPanel();
    this.draw();
  }

  state() {
    const freeze = this.mode === "moment" && this.segment && this.segment.kind === "freeze";
    return {
      frame: this.frame,
      overlayFrame: this.overlayFrame,
      paused: freeze ? true : !!(this.video && this.video.paused),
      playbackRate: this.mode === "moment" ? this.playbackRate : (this.video ? this.video.playbackRate : this.selectedRate),
      findingId: this.findingId,
      player: this.player,
      mode: this.mode,
      segmentShow: this.mode === "moment" ? this.segmentShow : null,
    };
  }

  drawn() {
    return {
      pills: this.pills.map((pill) => ({ ...pill })),
      boxes: this.boxes.map((box) => ({ ...box })),
      captions: this.captions.slice(),
      canvas: {
        width: this.canvas ? this.canvas.width : 0,
        height: this.canvas ? this.canvas.height : 0,
      },
    };
  }

  build(element) {
    const app = node("div", "app");
    const col = node("div", "video-col");
    this.stage = node("div", "stage");
    this.wrap = node("div", "zoom-wrap");
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.preload = "auto";
    this.canvas = node("canvas", "overlay");
    this.wrap.append(this.video, this.canvas);
    this.stage.append(this.wrap);
    this.caption = node("div", "caption");
    this.strip = node("canvas", "strip");
    this.transport = node("div", "transport");
    this.playButton = button("", "Play");
    this.transport.append(this.playButton);
    this.speedButtons = [0.25, 0.5, 1].map((rate) => {
      const el = button("", String(rate));
      el.dataset.rate = String(rate);
      el.setAttribute("aria-pressed", rate === 1 ? "true" : "false");
      this.transport.append(el);
      return el;
    });
    this.playerButtons = ["near", "far"].map((role) => {
      const el = button("", role);
      el.dataset.role = role;
      el.setAttribute("aria-pressed", "true");
      this.transport.append(el);
      return el;
    });
    this.skeletonButton = button("", "Skeleton");
    this.skeletonButton.setAttribute("aria-pressed", "true");
    this.zoomButton = button("", "Zoom");
    this.zoomButton.setAttribute("aria-pressed", "false");
    this.transport.append(this.skeletonButton, this.zoomButton);
    col.append(this.stage, this.caption, this.strip, this.transport);
    this.panel = node("aside", "panel");
    app.append(col, this.panel);
    element.append(app);
  }

  bind() {
    this.playButton.addEventListener("click", () => {
      if (this.mode === "moment") this.leaveMoment();
      else this.togglePlay();
    });
    for (const el of this.speedButtons) {
      el.addEventListener("click", () => this.setRate(Number(el.dataset.rate)));
    }
    for (const el of this.playerButtons) {
      el.addEventListener("click", () => {
        const role = el.dataset.role;
        this.showPlayer[role] = !this.showPlayer[role];
        el.setAttribute("aria-pressed", this.showPlayer[role] ? "true" : "false");
        this.draw();
      });
    }
    this.skeletonButton.addEventListener("click", () => {
      this.showSkeleton = !this.showSkeleton;
      this.skeletonButton.setAttribute("aria-pressed", this.showSkeleton ? "true" : "false");
      this.draw();
    });
    this.zoomButton.addEventListener("click", () => {
      this.zoomForced = !this.zoomForced;
      this.zoomButton.setAttribute("aria-pressed", this.zoomForced ? "true" : "false");
      this.draw();
    });
    this.video.addEventListener("click", () => {
      if (this.mode === "moment") this.leaveMoment();
      else this.togglePlay();
    });
    this.strip.addEventListener("click", (event) => {
      const rect = this.strip.getBoundingClientRect();
      const t = rect.width ? (event.clientX - rect.left) / rect.width : 0;
      const frame = Math.round(t * (this.bundle.n_frames - 1));
      this.seekFrame(frame);
    });
    window.addEventListener("resize", () => {
      this.fit();
      this.draw();
    });
    window.addEventListener("keydown", (event) => {
      if (this.ignoresKeys(event)) return;
      if (event.target && event.target.closest && event.target.closest("button") && event.key === " ") return;
      if (event.key === " " || event.key === "Escape") {
        event.preventDefault();
        if (this.mode === "moment" || event.key === "Escape") this.leaveMoment();
        else this.togglePlay();
        return;
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
        event.preventDefault();
        this.seekFrame(this.frame - 1);
      } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
        event.preventDefault();
        this.seekFrame(this.frame + 1);
      }
    });
  }

  async load(bundleUrl, options) {
    this.bundleUrl = new URL(bundleUrl, location.href).href;
    const response = await fetch(this.bundleUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`bundle ${response.status}`);
    const parsed = await response.json();
    this.bundle = normalizeBundle(parsed);
    this.poses = decodePoses(this.bundle);
    this.shownAt.clear();
    const videoUrl = new URL(this.bundle.video_url, this.bundleUrl).href;
    await waitForVideo(this.video, videoUrl);
    if (options.selectFirst && this.bundle.findings[0]) {
      this.findingId = this.bundle.findings[0].id;
      this.player = this.bundle.findings[0].player;
      this.momentIndex = 0;
    }
    this.fit();
  }

  armRvfc() {
    if (!this.video.requestVideoFrameCallback || this.rvfcArmed) return;
    this.rvfcArmed = true;
    const tick = (_now, meta) => {
      this.onPresented(meta.mediaTime);
      if (this.alive) this.video.requestVideoFrameCallback(tick);
    };
    this.video.requestVideoFrameCallback(tick);
  }

  onPresented(mediaTime) {
    if (!Number.isFinite(mediaTime) || !this.bundle) return;
    this.lastMediaTime = mediaTime;
    const frame = frameAt(this.bundle.timestamps, this.bundle.fps, this.bundle.n_frames, mediaTime);
    this.frame = frame;
    this.overlayFrame = frame;
    this.paused = !!(this.video && this.video.paused);
    if (this.mode === "moment" && this.segment && this.segment.kind === "freeze") {
      this.playbackRate = 0;
      this.paused = true;
    } else if (this.mode === "moment" && this.segment && this.segment.kind === "play") {
      if (this.video && Math.abs(this.video.playbackRate - this.segment.speed) > 0.001) {
        try { this.video.playbackRate = this.segment.speed; } catch { /* report whatever the element kept */ }
      }
      this.playbackRate = this.video ? this.video.playbackRate : this.segment.speed;
    } else if (this.mode !== "moment" && this.video) {
      this.playbackRate = this.video.playbackRate;
    }
    this.draw();
  }

  presentedFrame() {
    if (!this.bundle) return this.frame;
    return frameAt(this.bundle.timestamps, this.bundle.fps, this.bundle.n_frames, this.lastMediaTime || 0);
  }

  async runFrom(segmentIndex, token) {
    const finding = this.selected();
    const timeline = finding && finding.timelines[this.momentIndex];
    if (!timeline) return;
    try {
      for (let i = segmentIndex; i < timeline.segments.length; i++) {
        if (token !== this.token) return;
        const seg = timeline.segments[i];
        this.segmentIndex = i;
        this.segment = seg;
        const showChanged = this.segmentShow !== seg.show;
        this.segmentShow = seg.show;
        this.mode = "moment";
        if (showChanged) this.renderPanel();
        if (seg.kind === "freeze") {
          this.playbackRate = 0;
          this.paused = true;
          this.video.pause();
        } else {
          try { this.video.playbackRate = seg.speed; } catch { /* applied again after the seek */ }
          this.playbackRate = readRate(this.video, seg.speed);
        }
        this.applyZoom();
        if (seg.kind === "freeze") await this.runFreeze(seg, token);
        else await this.runPlay(seg, token);
        if (token !== this.token) return;
      }
      if (token !== this.token) return;
      this.video.pause();
      this.paused = true;
      this.mode = "idle";
      this.segment = null;
      this.segmentShow = null;
      this.playbackRate = this.selectedRate;
      try { this.video.playbackRate = this.selectedRate; } catch { /* idle rate is advisory */ }
      this.renderPanel();
      this.draw();
    } catch (error) {
      if (token !== this.token) return;
      this.failPlayback(error);
      throw error;
    }
  }

  async runFreeze(seg, token) {
    this.playbackRate = 0;
    this.paused = true;
    let got = await this.land(seg.start_frame);
    if (token !== this.token) return;
    if (got !== seg.start_frame) got = await this.land(seg.start_frame);
    if (token !== this.token) return;
    if (got === seg.start_frame) {
      this.frame = got;
      this.overlayFrame = got;
    }
    this.playbackRate = 0;
    this.paused = true;
    this.video.pause();
    this.draw();
    await wait(seg.hold_s * 1000, () => token !== this.token);
  }

  async runPlay(seg, token) {
    this.segmentReady = false;
    try { this.video.playbackRate = seg.speed; } catch { /* read back below */ }
    this.playbackRate = readRate(this.video, seg.speed);
    await this.land(seg.start_frame);
    if (token !== this.token) return;
    try { this.video.playbackRate = seg.speed; } catch { /* the element keeps the previous rate */ }
    this.playbackRate = readRate(this.video, seg.speed);
    this.segmentReady = true;
    const status = await this.startPlay(token);
    if (token !== this.token || status === "interrupted") return;
    if (this.video.paused) throw new Error("playback did not start");
    try { this.video.playbackRate = seg.speed; } catch { /* play can reset the element rate */ }
    this.playbackRate = readRate(this.video, seg.speed);
    this.paused = false;
    await this.waitForEnd(seg, token);
    this.segmentReady = false;
  }

  startPlay(token) {
    const video = this.video;
    let timer = 0;
    let finishWait = () => {};
    const timeout = new Promise((resolve, reject) => {
      const startedAt = performance.now();
      finishWait = () => resolve("interrupted");
      const step = () => {
        if (token !== this.token) {
          resolve("interrupted");
          return;
        }
        if (!video.paused) {
          resolve("started");
          return;
        }
        if (performance.now() - startedAt >= 800) {
          reject(new Error("playback timed out"));
          return;
        }
        timer = setTimeout(step, 20);
      };
      timer = setTimeout(step, 20);
    });
    const started = video.play().then(
      () => (token !== this.token ? "interrupted" : "started"),
      (error) => {
        if (token !== this.token) return "interrupted";
        throw error;
      },
    );
    started.catch(() => {});
    timeout.catch(() => {});
    return new Promise((resolve, reject) => {
      let done = false;
      const win = (status) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        finishWait();
        resolve(status);
      };
      const lose = (error) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        finishWait();
        reject(error);
      };
      started.then(win, lose);
      timeout.then(win, lose);
    });
  }

  waitForEnd(seg, token) {
    const end = this.bundle.timestamps[seg.end_frame];
    const span = end - this.bundle.timestamps[seg.start_frame];
    const budget = (span / Math.max(seg.speed, 0.05)) * 1000 + 1200;
    const started = performance.now();
    return new Promise((resolve, reject) => {
      const check = () => {
        if (token !== this.token) {
          resolve();
          return;
        }
        const media = Math.max(this.lastMediaTime || 0, this.video.currentTime || 0);
        const duration = this.video.duration;
        if (media >= end - 1e-4 || this.video.ended || (Number.isFinite(duration) && media >= duration - 1e-3)) {
          resolve();
          return;
        }
        if (this.video.paused) {
          reject(new Error("playback stopped"));
          return;
        }
        if (performance.now() - started > budget) {
          resolve();
          return;
        }
        setTimeout(check, 12);
      };
      check();
    });
  }

  failPlayback(error) {
    if (this.video) {
      this.video.pause();
      try { this.video.playbackRate = this.selectedRate; } catch { /* already stopped */ }
    }
    this.paused = true;
    this.mode = "idle";
    this.segment = null;
    this.segmentShow = null;
    this.segmentIndex = -1;
    this.segmentReady = false;
    this.playbackRate = this.selectedRate;
    this.locking = false;
    this.notePlaybackError(error);
    this.renderPanel();
    this.draw();
  }

  notePlaybackError(error) {
    const message = error && error.message ? error.message : String(error || "playback failed");
    this.playbackError = message;
    if (window.__viewerErrors) window.__viewerErrors.push(message);
  }

  leaveMoment() {
    this.token += 1;
    this.seekSeq += 1;
    this.mode = "idle";
    this.segment = null;
    this.segmentShow = null;
    this.segmentIndex = -1;
    this.segmentReady = false;
    this.playbackError = "";
    if (this.video) {
      this.video.pause();
      try { this.video.playbackRate = this.selectedRate; } catch { /* pause is the interrupt */ }
    }
    this.paused = true;
    this.playbackRate = this.selectedRate;
    this.locking = false;
    this.renderPanel();
    this.applyZoom();
    this.draw();
  }

  compareBest() {
    const finding = this.selected();
    if (!finding) return Promise.resolve(false);
    const timeline = finding.timelines[this.momentIndex] || finding.timelines[0];
    if (!timeline) return Promise.resolve(false);
    const index = timeline.segments.findIndex((seg) => seg.show === "best" && seg.kind === "freeze");
    if (index < 0) return Promise.resolve(false);
    this.token += 1;
    const token = this.token;
    this.playbackError = "";
    this.mode = "moment";
    this.momentIndex = finding.timelines[this.momentIndex] ? this.momentIndex : 0;
    this.notifySelect();
    return this.runFrom(index, token);
  }

  togglePlay() {
    if (!this.video) return Promise.resolve();
    if (!this.video.paused) {
      this.video.pause();
      this.paused = true;
      this.playbackRate = this.video.playbackRate;
      this.draw();
      return Promise.resolve();
    }
    try { this.video.playbackRate = this.selectedRate; } catch { /* play at the element default */ }
    return this.video.play().then(() => {
      this.paused = false;
      this.playbackError = "";
      this.playbackRate = this.video.playbackRate;
      this.draw();
    }, (error) => {
      this.paused = true;
      this.playbackRate = this.video.playbackRate;
      this.notePlaybackError(error);
      this.draw();
      throw error;
    });
  }

  setRate(rate) {
    this.selectedRate = rate;
    if (this.mode === "moment") return;
    try { this.video.playbackRate = rate; } catch { /* keep the previous element rate */ }
    this.playbackRate = this.video.playbackRate;
    this.draw();
  }

  async land(frame) {
    const token = this.token;
    const bundle = this.bundle;
    frame = Math.max(0, Math.min(bundle.n_frames - 1, frame));
    this.locking = true;
    this.video.pause();
    this.paused = true;
    const keepRate = this.mode === "moment" && this.segment && this.segment.kind === "play" ? this.segment.speed : null;
    if (keepRate != null) {
      try { this.video.playbackRate = keepRate; } catch { /* reported from playbackRate below */ }
      this.playbackRate = readRate(this.video, keepRate);
    }
    const fracs = [0.5, 0.38, 0.62, 0.28, 0.72, 0.18, 0.82, 0.1, 0.9];
    for (const frac of fracs) {
      if (token !== this.token) {
        this.locking = false;
        return this.presentedFrame();
      }
      const time = this.clampedTime(seekTime(bundle.timestamps, bundle.fps, bundle.n_frames, frame, frac));
      const media = await this.enqueueSeek(time);
      if (token !== this.token) {
        this.locking = false;
        return this.presentedFrame();
      }
      if (keepRate != null) {
        try { this.video.playbackRate = keepRate; } catch { /* seek can reset the element */ }
        this.playbackRate = readRate(this.video, keepRate);
      }
      if (frameAt(bundle.timestamps, bundle.fps, bundle.n_frames, media) === frame) break;
    }
    if (token !== this.token) {
      this.locking = false;
      return this.presentedFrame();
    }
    const shown = this.presentedFrame();
    this.frame = shown;
    this.overlayFrame = shown;
    this.paused = true;
    this.locking = false;
    this.fit();
    this.draw();
    return shown;
  }

  clampedTime(time) {
    const dur = this.video.duration;
    if (Number.isFinite(dur) && dur > 0) return Math.min(Math.max(0, time), Math.max(0, dur - 0.0005));
    return Math.max(0, time);
  }

  enqueueSeek(time) {
    const run = this.seekChain.then(() => this.seekVideo(time));
    this.seekChain = run.then(() => {}, () => {});
    return run;
  }

  seekVideo(time) {
    const video = this.video;
    const seq = ++this.seekSeq;
    const before = video.currentTime;
    return new Promise((resolve) => {
      let settled = false;
      const finish = (mediaTime) => {
        if (settled) return;
        settled = true;
        video.removeEventListener("seeked", onSeeked);
        const value = Number.isFinite(mediaTime) ? mediaTime : this.lastMediaTime;
        if (seq === this.seekSeq && Number.isFinite(value)) this.lastMediaTime = value;
        resolve(Number.isFinite(value) ? value : 0);
      };
      const watch = (_now, meta) => {
        if (settled || seq !== this.seekSeq) return;
        const media = meta && Number.isFinite(meta.mediaTime) ? meta.mediaTime : NaN;
        if (!Number.isFinite(media)) return;
        const bundle = this.bundle;
        if (bundle && frameAt(bundle.timestamps, bundle.fps, bundle.n_frames, media) === frameAt(bundle.timestamps, bundle.fps, bundle.n_frames, time)) {
          finish(media);
          return;
        }
        const samePlace = Number.isFinite(before) && Math.abs(time - before) < 0.0004;
        const stillBefore = !samePlace && Number.isFinite(before) && Math.abs(media - before) < 0.0004;
        if (stillBefore) {
          video.requestVideoFrameCallback(watch);
          return;
        }
        finish(media);
      };
      const onSeeked = () => {
        if (settled || typeof video.requestVideoFrameCallback === "function") return;
        finish(video.currentTime);
      };
      if (typeof video.requestVideoFrameCallback === "function") video.requestVideoFrameCallback(watch);
      video.addEventListener("seeked", onSeeked);
      if (Math.abs(video.currentTime - time) < 0.0004) {
        try { video.currentTime = Math.max(0, time - 0.002); } catch { /* the following assignment still seeks */ }
      }
      try { video.currentTime = time; } catch { finish(this.lastMediaTime); }
      setTimeout(() => finish(this.lastMediaTime), 700);
    });
  }

  fit() {
    if (!this.stage || !this.bundle) return;
    const bounds = this.stage.getBoundingClientRect();
    if (bounds.width < 8 || bounds.height < 8) return;
    const sw = this.sourceWidth();
    const sh = this.sourceHeight();
    let w = bounds.width;
    let h = w * sh / sw;
    if (h > bounds.height) {
      h = bounds.height;
      w = h * sw / sh;
    }
    w = Math.max(2, Math.round(w));
    h = Math.max(2, Math.round(h));
    if (this.canvas.width === w && this.canvas.height === h && this.wrap.style.width === `${w}px`) {
      this.applyZoom();
      return;
    }
    this.wrap.style.width = `${w}px`;
    this.wrap.style.height = `${h}px`;
    this.canvas.width = w;
    this.canvas.height = h;
    this.applyZoom();
  }

  applyZoom() {
    if (!this.wrap) return;
    const scale = this.zoomForced ? 1 : this.currentZoom();
    const origin = this.zoomOrigin();
    this.wrap.style.transformOrigin = `${origin.x}px ${origin.y}px`;
    this.wrap.style.transform = scale === 1 ? "none" : `scale(${scale})`;
  }

  currentZoom() {
    if (this.zoomForced) return 1;
    if (this.mode === "moment" && this.segment) return this.segment.zoom || 1;
    return 1;
  }

  zoomOrigin() {
    const w = this.canvas ? this.canvas.width : 1;
    const h = this.canvas ? this.canvas.height : 1;
    const finding = this.selected();
    const frame = this.originFrame();
    const box = finding ? this.boxOf(finding.player, frame) : null;
    if (!box) return { x: w / 2, y: h / 2 };
    const cx = ((box[0] + box[2]) / 2) / this.sourceWidth() * w;
    const cy = ((box[1] + box[3]) / 2) / this.sourceHeight() * h;
    return { x: cx, y: cy };
  }

  originFrame() {
    const finding = this.selected();
    if (!finding) return this.frame;
    if (this.segmentShow === "best") {
      const best = finding.moments.find((moment) => moment.kind === "best");
      if (best) return best.key_frame;
    }
    const moment = finding.moments[this.momentIndex] || finding.moments[0];
    return moment ? moment.key_frame : this.frame;
  }

  loop() {
    if (!this.alive) {
      this.raf = 0;
      return;
    }
    this.raf = -1;
    if (this.needsPaint()) this.draw();
    if (!this.alive) {
      this.raf = 0;
      return;
    }
    if (this.needsPaint()) this.raf = requestAnimationFrame(() => this.loop());
    else this.raf = 0;
  }

  needsPaint() {
    if (!this.video || this.locking) return true;
    if (!this.video.paused) return false;
    return this.marksFading();
  }

  marksFading() {
    if (this.reduceMotion) return false;
    const now = performance.now();
    for (const mark of this.visibleMarks()) {
      const started = this.shownAt.get(markKey(mark));
      if (started == null || now - started < 350) return true;
    }
    return false;
  }

  armLoop() {
    if (!this.alive || this.raf) return;
    if (!this.needsPaint()) return;
    this.raf = requestAnimationFrame(() => this.loop());
  }

  draw() {
    if (!this.bundle || !this.canvas || !this.canvas.width) return;
    this.fit();
    this.applyZoom();
    this.drawOverlay();
    this.drawCaption();
    this.drawStrip();
    this.drawMinimap();
    if (this.playButton) this.playButton.textContent = this.video.paused ? "Play" : "Pause";
    for (const el of this.speedButtons || []) {
      const on = this.mode !== "moment" && Number(el.dataset.rate) === this.selectedRate;
      el.setAttribute("aria-pressed", on ? "true" : "false");
    }
    this.armLoop();
  }

  drawOverlay() {
    const ctx = this.canvas.getContext("2d");
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.boxes = [];
    this.pills = [];
    this.captions = [];
    if (this.showSkeleton) this.drawSkeleton(ctx);
    else this.recordBoxes();
    this.drawMarks(ctx);
  }

  recordBoxes() {
    for (const role of this.bundle.players) {
      const box = this.boxCanvas(role, this.overlayFrame);
      if (box) this.boxes.push(box);
    }
  }

  drawSkeleton(ctx) {
    const style = this.bundle.style;
    const finding = this.selected();
    const focus = new Set((finding ? finding.focus_joints : []).map((name) => this.jointIndex(name)));
    for (const role of this.bundle.players) {
      const box = this.boxCanvas(role, this.overlayFrame);
      if (box) this.boxes.push(box);
      if (!this.showPlayer[role]) continue;
      const color = style.roles[role] || style.joint;
      const measured = this.measured(role, this.overlayFrame);
      const limb = Math.max(style.min_limb_px, (box ? box.h : 48) / style.limb_per_box_height);
      const dashed = measured === 0;
      for (const [a, b] of style.bones) {
        if (!this.jointOn(role, this.overlayFrame, a) || !this.jointOn(role, this.overlayFrame, b)) continue;
        const pa = this.canvasJoint(role, this.overlayFrame, a);
        const pb = this.canvasJoint(role, this.overlayFrame, b);
        const focused = focus.has(a) && focus.has(b);
        ctx.save();
        ctx.globalAlpha = this.jointAlpha(style, measured, finding, focused);
        ctx.lineCap = "round";
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.strokeStyle = style.outline;
        ctx.lineWidth = limb + style.outline_extra_px;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = limb;
        ctx.stroke();
        ctx.restore();
      }
      for (let joint = 0; joint < JOINTS; joint++) {
        if (!this.jointOn(role, this.overlayFrame, joint)) continue;
        const point = this.canvasJoint(role, this.overlayFrame, joint);
        const focused = focus.has(joint);
        const radius = Math.max(3, limb * 0.5) * (focused ? style.focus_scale : 1);
        ctx.save();
        ctx.globalAlpha = this.jointAlpha(style, measured, finding, focused);
        if (focused) {
          ctx.shadowColor = color;
          ctx.shadowBlur = 12;
        }
        ctx.fillStyle = style.outline;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius + 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = style.joint;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  }

  jointAlpha(style, measured, finding, focused) {
    let alpha = measured === 0 ? style.interpolated.alpha : 1;
    if (finding && !focused) alpha *= style.dim;
    return alpha;
  }

  drawMarks(ctx) {
    const marks = this.visibleMarks();
    const live = new Set(marks.map(markKey));
    for (const key of this.shownAt.keys()) {
      if (!live.has(key)) this.shownAt.delete(key);
    }
    const boxes = this.boxes.slice();
    const faces = this.facePoints(this.overlayFrame);
    const taken = [];
    const font = Math.max(14, Math.round(this.canvas.height / 28));
    ctx.font = `${font}px "Segoe UI", sans-serif`;
    ctx.textBaseline = "middle";
    for (const mark of marks) {
      const pts = this.markPoints(mark);
      if (!pts) continue;
      const look = (this.bundle.style.marks && this.bundle.style.marks[mark.style]) || {
        stroke: "#fff", fill: "#fff", dash: false, width: 2,
      };
      const opacity = this.markOpacity(mark);
      ctx.save();
      ctx.globalAlpha = opacity;
      const draw = MARK_DRAW[mark.kind];
      if (draw) draw(ctx, pts, look);
      ctx.restore();
      if (!mark.label) continue;
      const size = { w: Math.ceil(ctx.measureText(mark.label).width) + 16, h: font + 10 };
      const placed = placeLabel(anchorOf(mark.kind, pts), size, boxes, faces, taken, {
        w: this.canvas.width,
        h: this.canvas.height,
      });
      if (!placed) {
        this.captions.push(mark.label);
        continue;
      }
      ctx.save();
      ctx.globalAlpha = opacity;
      if (gapTo(placed, anchorOf(mark.kind, pts)) > 8) {
        const anchor = anchorOf(mark.kind, pts);
        const endX = Math.min(Math.max(anchor.x, placed.x), placed.x + placed.w);
        const endY = Math.min(Math.max(anchor.y, placed.y), placed.y + placed.h);
        ctx.strokeStyle = look.stroke;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(anchor.x, anchor.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }
      drawPill(ctx, placed, mark.label, look);
      ctx.restore();
      taken.push(placed);
      this.pills.push({ kind: mark.kind, text: mark.label, x: placed.x, y: placed.y, w: placed.w, h: placed.h });
    }
  }

  markOpacity(mark) {
    if (this.reduceMotion) return 1;
    const key = markKey(mark);
    if (!this.shownAt.has(key)) this.shownAt.set(key, performance.now());
    return Math.max(0, Math.min(1, (performance.now() - this.shownAt.get(key)) / 350));
  }

  visibleMarks() {
    const finding = this.selected();
    if (!finding) return [];
    const moment = this.markMoment(finding);
    if (!moment) return [];
    if (moment.kind === "best" && this.segmentShow !== "best") return [];
    return moment.marks.filter((mark) => Math.abs(this.overlayFrame - mark.frame) <= 1);
  }

  markMoment(finding) {
    if (this.segmentShow === "best") return finding.moments.find((moment) => moment.kind === "best") || null;
    return finding.moments[this.momentIndex] || finding.moments[0] || null;
  }

  momentScopeLabel(finding) {
    const moment = this.markMoment(finding);
    if (!moment || !moment.marks) return "";
    const valued = moment.marks.find((mark) => mark.value != null && String(mark.label || "").trim());
    if (!valued) return "";
    const onFrame = Math.abs(this.overlayFrame - valued.frame) <= 1;
    if (!onFrame && this.mode !== "moment") return "";
    return String(valued.label).trim();
  }

  notifySelect() {
    this.onSelect(this.findingId, this.player, this.momentIndex);
  }

  ignoresKeys(event) {
    const el = event.target;
    if (!el || el === document.body || el === document.documentElement) return false;
    if (el.isContentEditable) return true;
    const tag = el.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (el.closest && el.closest("dialog")) return true;
    if (this.root && !this.root.contains(el)) return true;
    return false;
  }

  markPoints(mark) {
    const pts = mark.points.map(([x, y]) => {
      const point = this.toCanvas(x, y);
      return { x: point[0], y: point[1] };
    });
    if (pts.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) return null;
    return pts;
  }

  drawCaption() {
    const parts = [];
    if (this.playbackError) parts.push(this.playbackError);
    else if (this.segmentShow === "best") parts.push("Your best one");
    parts.push(...this.captions);
    const text = parts.join(" ");
    const on = text.length > 0;
    if (this.caption.textContent !== text) this.caption.textContent = text;
    this.caption.classList.toggle("on", on);
  }

  drawStrip() {
    const canvas = this.strip;
    const width = Math.max(2, Math.round(canvas.clientWidth));
    const height = Math.max(2, Math.round(canvas.clientHeight));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    const n = this.bundle.n_frames;
    const xOf = (frame) => (n <= 1 ? 0 : (frame / (n - 1)) * width);
    for (const finding of this.bundle.findings) {
      ctx.fillStyle = (this.bundle.style.severity && this.bundle.style.severity[finding.severity]) || "#57534e";
      for (const moment of finding.moments) {
        const x = xOf(moment.key_frame);
        ctx.fillRect(x - 1.5, 10, 3, height - 16);
      }
    }
    ctx.fillStyle = "#f6f4f1";
    for (const event of this.bundle.events || []) {
      if (event.kind !== "swing") continue;
      ctx.fillRect(xOf(event.frame), 2, 1, 8);
    }
    ctx.fillStyle = "#ff00aa";
    ctx.fillRect(xOf(this.frame) - 1, 0, 2, height);
  }

  drawMinimap() {
    const canvas = this.panel.querySelector("canvas.minimap");
    if (!canvas || !this.bundle.court) return;
    const finding = this.selected();
    const moment = finding && this.markMoment(finding);
    if (!finding || !moment || (finding.category !== "footwork" && finding.category !== "tactics")) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);
    const court = this.bundle.court;
    const pad = 10;
    const toMini = (x, y) => [
      pad + (x / court.width_m) * (w - pad * 2),
      pad + (y / court.length_m) * (h - pad * 2),
    ];
    ctx.strokeStyle = "#1c1c1c";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - pad * 2, h - pad * 2);
    ctx.beginPath();
    const midX = toMini(court.width_m / 2, 0)[0];
    ctx.moveTo(midX, pad);
    ctx.lineTo(midX, h - pad);
    const netY = toMini(0, court.length_m / 2)[1];
    ctx.moveTo(pad, netY);
    ctx.lineTo(w - pad, netY);
    ctx.stroke();
    for (const role of this.bundle.players) {
      ctx.strokeStyle = this.bundle.style.roles[role] || "#1c1c1c";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (let frame = moment.start_frame; frame <= moment.end_frame; frame++) {
        const feet = this.feet(role, frame);
        if (!feet) continue;
        const courtPt = project(court.to_court, feet[0], feet[1]);
        if (!courtPt) continue;
        const mini = toMini(courtPt[0], courtPt[1]);
        if (!started) {
          ctx.moveTo(mini[0], mini[1]);
          started = true;
        } else ctx.lineTo(mini[0], mini[1]);
      }
      ctx.stroke();
      const now = this.feet(role, this.overlayFrame);
      if (!now) continue;
      const courtNow = project(court.to_court, now[0], now[1]);
      if (!courtNow) continue;
      const dot = toMini(courtNow[0], courtNow[1]);
      ctx.fillStyle = this.bundle.style.roles[role];
      ctx.beginPath();
      ctx.arc(dot[0], dot[1], 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    for (const mark of moment.marks) {
      if (mark.kind !== "dot" && mark.kind !== "line") continue;
      const look = this.bundle.style.marks[mark.style] || { stroke: "#1c1c1c" };
      ctx.strokeStyle = look.stroke;
      ctx.fillStyle = look.fill || look.stroke;
      const projected = mark.points.map(([x, y]) => project(court.to_court, x, y)).filter(Boolean);
      if (mark.kind === "dot" && projected[0]) {
        const mini = toMini(projected[0][0], projected[0][1]);
        ctx.beginPath();
        ctx.arc(mini[0], mini[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
      if (mark.kind === "line" && projected.length === 2) {
        const a = toMini(projected[0][0], projected[0][1]);
        const b = toMini(projected[1][0], projected[1][1]);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
    }
  }

  renderPanel() {
    const panel = this.panel;
    if (!panel) return;
    panel.replaceChildren();
    const bundle = this.bundle;
    if (!bundle) return;
    const body = node("div", "panel-body");
    panel.append(body);
    if (bundle.source && bundle.source.fixture) body.append(node("p", "fixture", "Fixture"));
    const finding = this.selected();
    if (!finding) return;
    body.append(node("h1", "headline", finding.headline));
    body.append(node("p", "explain", finding.explain));
    const metric = finding.metrics[0];
    if (metric && metric.value != null && Number.isFinite(metric.value)) {
      const block = node("div", "block");
      block.append(node("span", "k", "Your number, clip summary"));
      block.append(node("div", "v", `${metric.label} ${formatNum(metric.value)} ${metric.unit}`.trim()));
      block.append(node("span", "tag", confidenceWord(metric.confidence)));
      if (metric.basis === "projected_2d") {
        block.append(node("span", "tag", "projected"));
        block.append(node("span", "note", "rough estimate from the 2D picture"));
      }
      const momentLabel = this.momentScopeLabel(finding);
      if (momentLabel) block.append(node("span", "note", `This moment. ${momentLabel}`));
      body.append(block);
    }
    const guided = finding.metrics.filter((item) => item.target);
    if (guided.length) {
      const block = node("div", "block");
      block.append(node("span", "k", "Practice guide"));
      for (const item of guided) {
        block.append(node("div", "v", `${formatNum(item.target[0])} to ${formatNum(item.target[1])} ${item.unit}`.trim()));
        block.append(node("span", "note", "A coaching guide, not a tested personal target."));
      }
      body.append(block);
    }
    const best = finding.metrics.filter((item) => item.best != null);
    if (best.length) {
      const block = node("div", "block");
      block.append(node("span", "k", "Your best"));
      for (const item of best) block.append(node("div", "v", `${formatNum(item.best)} ${item.unit}`.trim()));
      body.append(block);
    }
    if (this.segmentShow === "best") body.append(node("p", "note", "Your best one"));
    const warns = node("div", "warn-row");
    for (const code of bundle.warnings || []) {
      if (code === "preview_estimate") {
        const refine = button("warn", "Preview estimate");
        refine.addEventListener("click", () => {
          const first = bundle.windows && bundle.windows.length ? bundle.windows[0] : null;
          this.onRefine(first);
        });
        warns.append(refine);
      } else {
        warns.append(node("span", "warn", warningLabel(code)));
      }
    }
    if (warns.childNodes.length) body.append(warns);
    const chips = node("div", "chip-row");
    for (const id of finding.exercise_ids || []) {
      const chip = button("chip", exerciseName(id));
      chip.dataset.exercise = id;
      chip.addEventListener("click", () => this.onExercise(id));
      chips.append(chip);
    }
    if (chips.childNodes.length) body.append(chips);
    if (metric && finding.overlay_series && finding.overlay_series.length) {
      const spark = node("canvas", "spark");
      spark.width = 288;
      spark.height = 64;
      body.append(spark);
      const moment = this.markMoment(finding);
      const series = bundle.series && bundle.series[finding.player] && bundle.series[finding.player][finding.overlay_series[0]];
      if (moment && series) {
        drawSpark(spark, series.slice(moment.start_frame, moment.end_frame + 1), metric.target);
      }
    }
    if (bundle.court && (finding.category === "footwork" || finding.category === "tactics")) {
      const mini = node("canvas", "minimap");
      mini.width = 288;
      mini.height = 160;
      body.append(mini);
    }
    const compare = button("compare", "Compare with your best");
    compare.disabled = !this.hasBestSegment(finding);
    compare.addEventListener("click", () => this.compareBest());
    panel.append(compare);
    const nav = node("div", "finding-nav");
    const prev = button("", "Previous");
    const next = button("", "Next");
    prev.addEventListener("click", () => this.stepFinding(-1));
    next.addEventListener("click", () => this.stepFinding(1));
    nav.append(prev, next);
    panel.append(nav);
  }

  hasBestSegment(finding) {
    return finding.timelines.some((timeline) => timeline.segments.some((seg) => seg.show === "best"));
  }

  stepFinding(dir) {
    const list = this.bundle.findings;
    if (!list.length) return;
    let index = list.findIndex((item) => item.id === this.findingId && item.player === this.player);
    index = (index + dir + list.length) % list.length;
    this.focusFinding(list[index].id, list[index].player);
  }

  selected() {
    return this.find(this.findingId, this.player);
  }

  find(id, player) {
    if (!this.bundle || id == null) return null;
    return this.bundle.findings.find((item) => item.id === id && item.player === player) || null;
  }

  sourceWidth() {
    return (this.bundle.source && this.bundle.source.width) || this.video.videoWidth || 1;
  }

  sourceHeight() {
    return (this.bundle.source && this.bundle.source.height) || this.video.videoHeight || 1;
  }

  toCanvas(x, y) {
    return [x / this.sourceWidth() * this.canvas.width, y / this.sourceHeight() * this.canvas.height];
  }

  jointIndex(name) {
    const joints = this.bundle.joints || [];
    const index = joints.indexOf(name);
    return index >= 0 ? index : -1;
  }

  pose(role) {
    return this.poses && this.poses[role];
  }

  xy(role, frame, joint) {
    const pose = this.pose(role);
    if (!pose) return [NaN, NaN];
    const base = (frame * JOINTS + joint) * 2;
    return [pose.xy[base], pose.xy[base + 1]];
  }

  conf(role, frame, joint) {
    const pose = this.pose(role);
    if (!pose) return NaN;
    return pose.conf[frame * JOINTS + joint];
  }

  measured(role, frame) {
    const pose = this.pose(role);
    if (!pose) return 1;
    return pose.measured[frame];
  }

  boxOf(role, frame) {
    const pose = this.pose(role);
    if (!pose) return null;
    const base = frame * 4;
    const box = [pose.bbox[base], pose.bbox[base + 1], pose.bbox[base + 2], pose.bbox[base + 3]];
    if (box.some((value) => !Number.isFinite(value))) return null;
    return box;
  }

  boxCanvas(role, frame) {
    const box = this.boxOf(role, frame);
    if (!box) return null;
    const [x0, y0] = this.toCanvas(box[0], box[1]);
    const [x1, y1] = this.toCanvas(box[2], box[3]);
    return {
      role,
      x: Math.min(x0, x1),
      y: Math.min(y0, y1),
      w: Math.abs(x1 - x0),
      h: Math.abs(y1 - y0),
    };
  }

  jointOn(role, frame, joint) {
    const conf = this.conf(role, frame, joint);
    if (!(conf >= this.bundle.style.confidence_gate)) return false;
    const [x, y] = this.xy(role, frame, joint);
    return Number.isFinite(x) && Number.isFinite(y);
  }

  canvasJoint(role, frame, joint) {
    const [x, y] = this.xy(role, frame, joint);
    const point = this.toCanvas(x, y);
    return { x: point[0], y: point[1] };
  }

  facePoints(frame) {
    const points = [];
    for (const role of this.bundle.players) {
      for (const joint of [NOSE, LEFT_EYE, RIGHT_EYE]) {
        if (!this.jointOn(role, frame, joint)) continue;
        points.push(this.canvasJoint(role, frame, joint));
      }
    }
    return points;
  }

  feet(role, frame) {
    const left = this.jointOn(role, frame, LEFT_ANKLE) ? this.xy(role, frame, LEFT_ANKLE) : null;
    const right = this.jointOn(role, frame, RIGHT_ANKLE) ? this.xy(role, frame, RIGHT_ANKLE) : null;
    if (left && right) return [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2];
    return left || right;
  }
}

function markKey(mark) {
  return `${mark.kind}|${mark.frame}|${mark.label}`;
}

function drawPill(ctx, rect, text, look) {
  ctx.save();
  ctx.fillStyle = "rgba(246, 244, 241, 0.94)";
  ctx.strokeStyle = look.stroke || "#1c1c1c";
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(rect.x, rect.y, rect.w, rect.h, 11);
  else ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#1c1c1c";
  ctx.textBaseline = "middle";
  ctx.fillText(text, rect.x + 8, rect.y + rect.h / 2);
  ctx.restore();
}

function drawSpark(canvas, values, target) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  const nums = values.map((value) => (value == null ? NaN : Number(value)));
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of nums) {
    if (!Number.isFinite(value)) continue;
    lo = Math.min(lo, value);
    hi = Math.max(hi, value);
  }
  if (target) {
    lo = Math.min(lo, target[0]);
    hi = Math.max(hi, target[1]);
  }
  if (!Number.isFinite(lo) || lo === hi) {
    lo = (Number.isFinite(lo) ? lo : 0) - 1;
    hi = (Number.isFinite(hi) ? hi : 0) + 1;
  }
  const xOf = (index) => (values.length <= 1 ? 4 : 4 + (index / (values.length - 1)) * (w - 8));
  const yOf = (value) => 4 + (1 - (value - lo) / (hi - lo)) * (h - 8);
  if (target) {
    const y0 = yOf(target[0]);
    const y1 = yOf(target[1]);
    ctx.fillStyle = "rgba(0, 230, 255, 0.2)";
    ctx.fillRect(4, Math.min(y0, y1), w - 8, Math.abs(y1 - y0));
  }
  ctx.beginPath();
  let started = false;
  nums.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      started = false;
      return;
    }
    const x = xOf(index);
    const y = yOf(value);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#1c1c1c";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function project(matrix, x, y) {
  const X = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
  const Y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
  const W = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  if (!Number.isFinite(W) || Math.abs(W) < 1e-8) return null;
  return [X / W, Y / W];
}

function decodePoses(bundle) {
  const poses = {};
  for (const role of bundle.players) {
    const packed = bundle.poses[role];
    poses[role] = {
      xy: decodeFloats(packed.xy),
      conf: decodeFloats(packed.conf),
      measured: bytesOf(packed.measured.b64),
      bbox: decodeFloats(packed.bbox),
    };
  }
  return poses;
}

function readRate(video, fallback) {
  const applied = video.playbackRate;
  return Number.isFinite(applied) ? applied : fallback;
}

function waitForVideo(video, url) {
  if (video.src === url && video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const same = video.src === url;
    const events = same ? ["seeked", "canplay"] : ["loadeddata"];
    const ok = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("video"));
    };
    const cleanup = () => {
      for (const name of events) video.removeEventListener(name, ok);
      video.removeEventListener("error", fail);
    };
    for (const name of events) video.addEventListener(name, ok);
    video.addEventListener("error", fail);
    if (!same) video.src = url;
    if (video.readyState >= 2) ok();
  });
}

function wait(ms, cancelled) {
  return new Promise((resolve) => {
    const started = performance.now();
    const step = () => {
      if (cancelled() || performance.now() - started >= ms) {
        resolve();
        return;
      }
      setTimeout(step, 20);
    };
    step();
  });
}

window.__viewerErrors = [];
window.addEventListener("error", (event) => {
  window.__viewerErrors.push(String(event.message));
});
window.addEventListener("unhandledrejection", (event) => {
  window.__viewerErrors.push(String(event.reason && event.reason.stack || event.reason));
});

const params = new URLSearchParams(location.search);
const mountPoint = document.querySelector("#app");
if (mountPoint && params.has("bundle")) {
  const viewer = new Viewer();
  window.viewer = viewer;
  window.viewerReady = viewer.mount(mountPoint, params.get("bundle")).catch((error) => {
    window.__viewerErrors.push(String(error && error.stack || error));
    throw error;
  });
}
