// The app in a real browser (headless Edge over the DevTools protocol), with real mouse clicks and typing.
//
//   node coach\web\tests\e2e.mjs
//       Starts the app on a free port with the Replay backend (stored pose files, test-only) and an empty data folder,
//       then runs the whole flow: sample, names, slow motion, chat, exercise GIFs, "too hard", equipment, sharpening,
//       plan, a real upload, progress, the three bad files, and the layout at three window sizes.
//   node coach\web\tests\e2e.mjs --url http://127.0.0.1:8767/ --session <id> [--prefix live-turbo-]
//       Looks at one finished session of an app that is already running: slow motion, chat, GIFs, plan and layout.
//       It uploads nothing, renames nobody and sharpens nothing.
//   node coach\web\tests\e2e.mjs --url http://127.0.0.1:8767/ --upload <video> [--expect no_court] [--prefix live-black-]
//       Uploads one video that the real analyser should refuse (a black or non-badminton clip) through the file picker
//       of an app that is already running. The page and the server must give the same plain error, and the app must
//       still answer afterwards.
//
// Screenshots and <prefix>summary.json go to work\C6\shots. The exit code is 1 when any step fails or the page
// logs an error.
import { spawn, spawnSync } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PROJECT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const PY = process.platform === "win32" ? join(PROJECT, ".venv", "Scripts", "python.exe")
  : join(PROJECT, ".venv", "bin", "python");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const SHOTS = join(PROJECT, "work", "C6", "shots");
const SIZES = [[1280, 800], [1440, 900], [1920, 1080]];
const PLAYER = "E2E Player";

const args = parseArgs(process.argv.slice(2));
const PREFIX = args.prefix ?? (args.upload ? "live-upload-" : args.url ? "live-" : "e2e-");
const summary = {
  mode: args.upload ? "upload, to an app that is already running, of a clip it should refuse"
    : args.url ? "live" : "replay (test-only stored poses, not a speed or analysis claim)",
  started: new Date().toISOString(), url: null, backend: null, steps: [], timings: {}, page_errors: [], shots: [],
};
const tmp = mkdtempSync(join(tmpdir(), "c6-e2e-"));
let base = null;
let send = null;
let app = null;
let edge = null;
let current = "start";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (...parts) => console.error(...parts);

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i].startsWith("--") || argv[i + 1] === undefined) throw new Error(`bad argument ${argv[i]}`);
    out[argv[i].slice(2)] = argv[i + 1];
  }
  if (out.session && out.upload) throw new Error("give --session or --upload, not both");
  if (Boolean(out.url) !== Boolean(out.session || out.upload)) throw new Error("--url goes with --session or --upload");
  return out;
}

class Critical extends Error {}

async function step(name, fn, { critical = false } = {}) {
  current = name;
  const started = Date.now();
  try {
    const detail = (await fn()) ?? {};
    summary.steps.push({ name, ok: true, ms: Date.now() - started, ...detail });
    log("ok  ", name, JSON.stringify(detail).slice(0, 400));
    return detail;
  } catch (error) {
    summary.steps.push({ name, ok: false, ms: Date.now() - started, error: error.message });
    log("FAIL", name, error.message);
    if (critical) throw new Critical(`${name}: ${error.message}`);
    return null;
  } finally {
    await collectCaught(name);
  }
}

// The viewer module and app.js keep the errors they catch in window arrays; a passing step must leave them empty.
async function collectCaught(name) {
  if (!send) return;
  try {
    const caught = await js(`(() => {
      const out = [...(window.__viewerErrors || []), ...(window.__appErrors || [])];
      if (window.__viewerErrors) window.__viewerErrors.length = 0;
      if (window.__appErrors) window.__appErrors.length = 0;
      return out;
    })()`, 5000);
    for (const text of caught) summary.page_errors.push({ step: name, text: `caught by the page: ${text}` });
  } catch { /* the page is between two loads */ }
}

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitFor(fn, ms, label) {
  const end = Date.now() + ms;
  let last = "";
  for (;;) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (error) {
      if (error instanceof Critical) throw error;
      last = error.message;
    }
    if (Date.now() > end) throw new Error(`timed out after ${ms} ms waiting for ${label}${last ? `: ${last}` : ""}`);
    await sleep(150);
  }
}

// --- processes -------------------------------------------------------------------------------------------------------

function startApp() {
  const data = join(tmp, "data");
  // The app's own log is kept next to the summary, so a failed step can be read against what the server did.
  summary.app_log = join(SHOTS, `${PREFIX}app.log`);
  const appLog = openSync(summary.app_log, "w");
  return new Promise((resolve, reject) => {
    const child = spawn(PY, ["-u", "-m", "coach.app", "--backend", "replay", "--no-browser", "--port", "0",
      "--data-dir", data], { cwd: PROJECT, windowsHide: true });
    child.once("exit", () => closeSync(appLog));
    let out = "";
    const timer = setTimeout(() => reject(new Error(`the app printed no address within 60 s: ${out}`)), 60000);
    const take = (chunk) => {
      writeSync(appLog, chunk);
      out = (out + chunk).slice(-20000);
      const match = out.match(/Badminton coach: (http:\/\/127\.0\.0\.1:\d+)\//);
      if (match) {
        clearTimeout(timer);
        resolve({ child, url: match[1] });
      }
    };
    child.stdout.on("data", take);
    child.stderr.on("data", (chunk) => {
      writeSync(appLog, chunk);
      out = (out + chunk).slice(-20000);
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`the app exited with ${code}: ${out}`));
    });
  });
}

async function startEdge(width, height) {
  const profile = join(tmp, "edge");
  mkdirSync(profile, { recursive: true });
  const child = spawn(EDGE, ["--headless=new", "--disable-gpu", "--mute-audio", "--no-first-run",
    "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
    `--window-size=${width},${height}`, "about:blank"], { stdio: "ignore", windowsHide: true });
  // Port 0 plus DevToolsActivePort: never attach to another session's browser on a guessed port.
  const port = await waitFor(() => Number(readFileSync(join(profile, "DevToolsActivePort"), "utf8").split("\n")[0]),
    60000, "Edge DevToolsActivePort");
  const target = await waitFor(async () => {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    return list.find((item) => item.type === "page");
  }, 30000, "an Edge page");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve);
    ws.addEventListener("error", reject);
  });
  return { child, port, ws };
}

async function stopEdge() {
  if (!edge) return;
  try {
    const { webSocketDebuggerUrl } = await (await fetch(`http://127.0.0.1:${edge.port}/json/version`)).json();
    const ws = new WebSocket(webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve);
      ws.addEventListener("error", reject);
    });
    ws.send(JSON.stringify({ id: 1, method: "Browser.close" }));
  } catch {
    edge.child.kill();
  }
  await Promise.race([new Promise((resolve) => edge.child.once("exit", resolve)), sleep(5000)]);
  if (edge.child.exitCode === null) edge.child.kill();
}

async function stopApp() {
  if (!app) return;
  app.child.kill();
  await Promise.race([new Promise((resolve) => app.child.once("exit", resolve)), sleep(5000)]);
}

// --- DevTools helpers ------------------------------------------------------------------------------------------------

function attach(ws) {
  let next = 0;
  const pending = new Map();
  const handlers = new Map();
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject, timer } = pending.get(msg.id);
      clearTimeout(timer);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(`${msg.error.message} ${msg.error.data || ""}`.trim()));
      else resolve(msg.result);
    } else if (msg.method && handlers.has(msg.method)) {
      for (const fn of handlers.get(msg.method)) fn(msg.params);
    }
  });
  const call = (method, params = {}, timeout = 30000) => new Promise((resolve, reject) => {
    const id = ++next;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`DevTools ${method} timed out`));
    }, timeout);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params }));
  });
  call.on = (method, fn) => handlers.set(method, [...(handlers.get(method) || []), fn]);
  return call;
}

async function js(expression, timeout = 30000) {
  const result = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, timeout + 2000);
  if (result.exceptionDetails) {
    const detail = result.exceptionDetails;
    throw new Error((detail.exception && detail.exception.description) || detail.text);
  }
  return result.result.value;
}

const until = (expression, ms, label) => waitFor(() => js(expression), ms, label || expression);

async function setSize(width, height) {
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
}

async function open(path) {
  await js("window.__e2eOld = true");
  await send("Page.navigate", { url: base + path });
  await until("!window.__e2eOld && document.readyState === 'complete'", 30000, `load ${path}`);
}

// Clicks like a person: the middle of the element, after checking that nothing covers that point.
async function click(selector, timeout = 15000) {
  const box = await until(`(() => {
    const node = document.querySelector(${JSON.stringify(selector)});
    if (!node || node.disabled) return null;
    node.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' });
    const r = node.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const hit = document.elementFromPoint(x, y);
    const covered = !(hit && (hit === node || node.contains(hit)));
    return { x, y, covered, hit: hit ? hit.tagName.toLowerCase() + (hit.id ? '#' + hit.id : '') : null };
  })()`, timeout, `clickable ${selector}`);
  check(!box.covered, `${selector} is covered by ${box.hit}`);
  const at = { x: box.x, y: box.y };
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", ...at });
  await send("Input.dispatchMouseEvent", { type: "mousePressed", ...at, button: "left", clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", ...at, button: "left", clickCount: 1 });
}

async function type(selector, text) {
  await click(selector);
  await js(`document.querySelector(${JSON.stringify(selector)}).select()`);
  await send("Input.insertText", { text });
}

async function setFiles(selector, files) {
  const { root } = await send("DOM.getDocument", { depth: 1 });
  const { nodeId } = await send("DOM.querySelector", { nodeId: root.nodeId, selector });
  check(nodeId, `no ${selector}`);
  await send("DOM.setFileInputFiles", { nodeId, files });
}

async function shot(name) {
  const path = join(SHOTS, `${PREFIX}${name}.png`);
  const { data } = await send("Page.captureScreenshot", { format: "png" }, 60000);
  writeFileSync(path, Buffer.from(data, "base64"));
  summary.shots.push(path);
  return path;
}

// --- page facts ------------------------------------------------------------------------------------------------------

const FAILURE = `(() => {
  const start = document.querySelector('body > p.failure');
  if (start) return start.textContent.trim();
  const failure = document.getElementById('failure');
  if (!failure || failure.hidden) return null;
  return document.getElementById('failure-title').textContent.trim() + ': ' + document.getElementById('failure-text').textContent.trim();
})()`;

const NOTE = `(() => {
  const note = document.getElementById('notice');
  if (!note) return 'the page has no #notice element';
  return note.checkVisibility() ? note.textContent.trim() : null;
})()`;

const VIDEO_READY = "!!(coachApp.viewer && coachApp.viewer.video && coachApp.viewer.video.readyState >= 2)";

const SESSION = `(() => {
  const v = window.coachApp && coachApp.view;
  if (!v) return null;
  return {
    status: v.status && v.status.status, revision: v.revision, tier: v.bundle && v.bundle.tier,
    cards: document.querySelectorAll('#groups .card').length,
    video: ${VIDEO_READY},
    failure: ${FAILURE},
  };
})()`;

// Errors the page itself caught: the viewer module records its own, app.js records what it handled.
const CAUGHT = "[...(window.__viewerErrors || []), ...(window.__appErrors || [])]";

async function useful(ms) {
  try {
    return await waitFor(async () => {
      const s = await js(SESSION);
      if (s && s.failure) throw new Critical(`the page shows a failure: ${s.failure}`);
      return s && s.cards > 0 && s.video ? s : null;
    }, ms, "cards and a playable video");
  } catch (error) {
    if (error instanceof Critical) throw error;
    throw new Error(`${error.message}; ${JSON.stringify(await diagnose())}`);
  }
}

const SESSION_ID = "new URLSearchParams(location.search).get('s')";
const STATUS_TEXT = "document.getElementById('status').textContent";

// Injected into the session page before it loads: notes, at animation-frame precision, when the first cards and a
// playable video appear, so the first-useful-screen time is not blurred by the 150 ms polling of waitFor.
const MARK_USEFUL = `(() => {
  if (!new URLSearchParams(location.search).get('s')) return;
  const marks = window.__e2eMarks = { events: {} };
  let watched = null;
  const tick = () => {
    const element = document.querySelector('#viewer video');
    if (element && element !== watched) {
      watched = element;
      marks.video_element = performance.now();
      for (const name of ['loadstart', 'loadedmetadata', 'loadeddata', 'canplay', 'seeked']) {
        element.addEventListener(name, () => { if (marks.events[name] == null) marks.events[name] = performance.now(); });
      }
    }
    const video = window.coachApp && coachApp.viewer && coachApp.viewer.video;
    if (marks.cards == null && document.querySelectorAll('#groups .card').length > 0) marks.cards = performance.now();
    if (marks.video == null && video && video.readyState >= 2) marks.video = performance.now();
    if (marks.cards != null && marks.video != null) marks.useful = Math.max(marks.cards, marks.video);
    else requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
})()`;

// Where the time to the first useful screen went, in ms from the new page's start (resource start and end pairs).
const USEFUL_BREAKDOWN = `(() => {
  const marks = window.__e2eMarks || {};
  const spans = (re) => performance.getEntriesByType('resource').filter((e) => re.test(e.name))
    .map((e) => [Math.round(e.startTime), Math.round(e.responseEnd)]);
  const nav = performance.getEntriesByType('navigation')[0];
  const at = (value) => value == null ? null : Math.round(value);
  const events = {};
  for (const [name, value] of Object.entries(marks.events || {})) events[name] = at(value);
  return {
    origin: performance.timeOrigin, html: nav ? [Math.round(nav.requestStart), Math.round(nav.responseEnd)] : null,
    dom_content_loaded: nav ? Math.round(nav.domContentLoadedEventEnd) : null,
    cards: at(marks.cards), video: at(marks.video), useful: at(marks.useful), video_element: at(marks.video_element),
    video_events: events, static: Object.fromEntries(performance.getEntriesByType('resource')
      .filter((e) => /\\/static\\//.test(e.name)).map((e) => [e.name.replace(/^.*\\/static\\//, ''),
        [Math.round(e.startTime), Math.round(e.responseEnd)]])),
    exercises: spans(/\\/api\\/exercises/), status: spans(/\\/api\\/session\\/[^/?]+(\\?|$)/),
    bundle: spans(/\\/bundle\\?/),
  };
})()`;

const OVERFLOW = `(() => {
  const width = document.documentElement.clientWidth;
  const clips = (node) => {
    for (let a = node.parentElement; a && a !== document.body; a = a.parentElement) {
      if (getComputedStyle(a).overflowX !== 'visible') return true;
    }
    return false;
  };
  const bad = [];
  for (const node of document.body.querySelectorAll('*')) {
    if (node.closest('[hidden], dialog:not([open]), template')) continue;
    const r = node.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1) continue;
    const style = getComputedStyle(node);
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    if ((r.right > width + 1 || r.left < -1) && !clips(node)) {
      bad.push(node.tagName.toLowerCase() + (node.id ? '#' + node.id : '') + ' ' + Math.round(r.left) + '..' + Math.round(r.right));
    }
  }
  return { scroll: document.documentElement.scrollWidth, width, bad: bad.slice(0, 6), count: bad.length };
})()`;

const LAST_COACH = `(() => {
  const turn = coachApp.view.log.filter((t) => t.speaker === 'coach').at(-1);
  const drawer = document.getElementById('drawer');
  const reply = turn.reply || {};
  return {
    text: turn.text, ms: turn.ms, intent: reply.intent || null,
    actions: (reply.actions || []).map((a) => [a.kind, a.ref, a.arg].filter((x) => x != null).join(':')),
    drawer: drawer.hidden ? [] : [...document.querySelectorAll('#ex-list .ex-card')].map((n) => n.dataset.exercise),
    too_hard: coachApp.view.context.too_hard || [],
  };
})()`;

const COACH_TURNS = "coachApp.view.log.filter((t) => t.speaker === 'coach').length";

// --- flow pieces -----------------------------------------------------------------------------------------------------

async function nameNear() {
  await type("#name-near", PLAYER);
  await click("#names button[type=submit]");
  await until("document.getElementById('names-msg').textContent === 'Saved'", 10000, "names saved");
}

async function moment() {
  const pick = await js(`(() => {
    const button = document.querySelector('#groups .card [data-action=moment]');
    if (!button) return null;
    const key = button.closest('.card').dataset.key;
    const split = key.indexOf(':');
    const player = key.slice(0, split), id = key.slice(split + 1);
    const f = coachApp.view.bundle.findings.find((x) => x.player === player && x.id === id);
    const index = f.moments.findIndex((m) => m.kind !== 'best');
    const segments = f.timelines[index].segments;
    const freeze = segments.find((s) => s.kind === 'freeze');
    const slow = segments.find((s) => s.kind !== 'freeze');
    return { key, id, player, index, headline: f.headline, key_frame: f.moments[index].key_frame,
      freeze_start: freeze ? freeze.start_frame : null, hold_s: freeze ? freeze.hold_s : null,
      slow_speed: slow ? slow.speed : null, kinds: segments.map((s) => s.kind).join(',') };
  })()`);
  check(pick, "no card has a 'Show me in slow motion' button");
  check(pick.freeze_start !== null, `the timeline of ${pick.id} has no freeze segment (${pick.kinds})`);
  await js("window.__clicks = window.__clicks || (document.addEventListener('click', () => window.__clicks.push(performance.now()), true), []); true");
  await click(`.card[data-key="${pick.key}"] [data-action=moment]`);
  // The viewer reports a freeze as a moment that is paused at rate 0 on the freeze segment's first frame.
  const watch = await js(`(async () => {
    const t0 = window.__clicks[window.__clicks.length - 1];
    const viewer = coachApp.viewer;
    let slow = null, rate = null;
    const end = performance.now() + 20000;
    while (performance.now() < end) {
      const s = viewer.state();
      if (slow === null && s.mode === 'moment' && !s.paused && s.playbackRate > 0 && s.playbackRate < 1) {
        slow = performance.now() - t0;
        rate = viewer.video.playbackRate;
      }
      if (s.mode === 'moment' && s.paused && s.playbackRate === 0 && s.frame === ${pick.freeze_start}) {
        return { slow_ms: slow, freeze_ms: performance.now() - t0, rate, state: s, index: viewer.momentIndex,
          captions: viewer.drawn().captions, selected: coachApp.view.selected };
      }
      await new Promise((r) => requestAnimationFrame(r));
    }
    return { slow_ms: slow, freeze_ms: null, rate, state: viewer.state(), index: viewer.momentIndex,
      captions: viewer.drawn().captions, selected: coachApp.view.selected };
  })()`, 25000);
  await shot("moment-freeze");
  check(watch.freeze_ms !== null, `never froze on frame ${pick.freeze_start}; last state ${JSON.stringify(watch.state)}`);
  check(watch.state.findingId === pick.id && watch.state.player === pick.player && watch.index === pick.index,
    `the viewer plays a different moment: ${JSON.stringify(watch.state)} index ${watch.index}`);
  check(watch.selected === pick.key, `the selected card is ${watch.selected}, not ${pick.key}`);
  await js("coachApp.viewer.leaveMoment(), true");
  const after = await js("coachApp.viewer.state()");
  check(after.mode === "idle" && after.paused, `leaving the moment left ${JSON.stringify(after)}`);
  summary.timings.click_to_slow_ms = watch.slow_ms === null ? null : Math.round(watch.slow_ms);
  summary.timings.click_to_freeze_ms = Math.round(watch.freeze_ms);
  return {
    finding: pick.key, headline: pick.headline, frozen_on: watch.state.frame, key_frame: pick.key_frame,
    freeze_is_key_frame: pick.key_frame === pick.freeze_start, slow_rate: watch.rate, slow_speed: pick.slow_speed,
    kinds: pick.kinds, click_to_slow_ms: summary.timings.click_to_slow_ms, click_to_freeze_ms: summary.timings.click_to_freeze_ms,
    captions: watch.captions,
  };
}

// The viewer's own Next button changes its selection; the card list must follow it.
async function viewerNext() {
  const before = await js("({ selected: coachApp.view.selected, state: coachApp.viewer.state() })");
  await click("#viewer .finding-nav button:last-child");
  const after = await until(`(() => {
    const s = coachApp.viewer.state();
    const key = s.player + ':' + s.findingId;
    const card = document.querySelector('#groups .card.selected');
    if (key === ${JSON.stringify(before.selected)} || !card) return null;
    return { key, selected: coachApp.view.selected, card: card.dataset.key, player: coachApp.view.player };
  })()`, 5000, "the cards to follow the viewer's Next button");
  check(after.selected === after.key && after.card === after.key, `the viewer shows ${after.key} but the card list shows ${after.card}`);
  check(after.player === after.key.split(":")[0], `the player switch shows ${after.player} for ${after.key}`);
  const role = before.selected.split(":")[0];
  if (after.player !== role) await click(`.seg button[data-player="${role}"]`);
  await click(`.card[data-key="${before.selected}"] h3`);
  const back = await js("coachApp.viewer.state()");
  check(`${back.player}:${back.findingId}` === before.selected, `clicking the card left the viewer on ${back.player}:${back.findingId}`);
  return { from: before.selected, to: after.key };
}

// Keys typed into the chat stay in the chat; outside a text box the arrow keys still step the video.
async function keysStayInTheChat() {
  const key = async (keyName, code, keyCode, text) => {
    // Like Puppeteer: a key that types text is a keyDown, any other key a rawKeyDown.
    await send("Input.dispatchKeyEvent", { type: text ? "keyDown" : "rawKeyDown", key: keyName, code,
      windowsVirtualKeyCode: keyCode, text });
    await send("Input.dispatchKeyEvent", { type: "keyUp", key: keyName, code, windowsVirtualKeyCode: keyCode });
  };
  await js("coachApp.viewer.leaveMoment(), true");
  await click("#tab-coach");
  await type("#ask-text", "ab");
  const before = await js("coachApp.viewer.state()");
  await key(" ", "Space", 32, " ");
  await key("c", "KeyC", 67, "c");
  await key("ArrowLeft", "ArrowLeft", 37);
  await key("ArrowRight", "ArrowRight", 39);
  await key("ArrowLeft", "ArrowLeft", 37);
  await sleep(300);
  const typed = await js(`(() => { const i = document.getElementById('ask-text');
    return { value: i.value, caret: i.selectionStart, focused: document.activeElement === i, state: coachApp.viewer.state() }; })()`);
  check(typed.value === "ab c", `the chat box holds ${JSON.stringify(typed.value)}, not "ab c"`);
  check(typed.caret === 3, `the caret is at ${typed.caret}, not 3, so the arrow keys did not move it`);
  check(typed.focused, "the chat box lost focus");
  check(typed.state.frame === before.frame && typed.state.paused === before.paused,
    `typing moved the video: ${JSON.stringify(before)} -> ${JSON.stringify(typed.state)}`);
  await js("document.getElementById('ask-text').value = ''; document.getElementById('ask-text').blur(); document.body.focus(); true");
  await key("ArrowRight", "ArrowRight", 39);
  const stepped = await until(`coachApp.viewer.state().frame === ${before.frame + 1} ? coachApp.viewer.state() : null`, 3000,
    "ArrowRight outside the chat box to step one frame");
  await key("ArrowLeft", "ArrowLeft", 37);
  await until(`coachApp.viewer.state().frame === ${before.frame}`, 3000, "ArrowLeft to step back");
  return { typed: typed.value, caret: typed.caret, frame: before.frame, stepped_to: stepped.frame };
}

// Report warnings are shown in the words of coach.schema.WARNINGS, never as raw codes.
async function plainWarnings() {
  const seen = await js(`(() => {
    const section = [...document.querySelectorAll('#groups .group')].find((s) => s.querySelector('h2').textContent === 'Worth knowing');
    const items = section ? [...section.querySelectorAll('li')].map((li) => li.textContent.trim()) : [];
    return { codes: coachApp.view.bundle.warnings || [], items };
  })()`);
  const words = await (await fetch(`${base}/api/health`)).json();
  check(seen.items.length === seen.codes.length, `${seen.codes.length} warning(s) but ${seen.items.length} line(s)`);
  seen.codes.forEach((code, i) => {
    check(words.warnings[code], `the app has no words for the warning ${code}`);
    check(seen.items[i] === words.warnings[code], `${code} is shown as ${JSON.stringify(seen.items[i])}`);
  });
  return { codes: seen.codes, shown: seen.items };
}

async function ask(text) {
  const before = await js(COACH_TURNS);
  await click("#tab-coach");
  await type("#ask-text", text);
  await click("#ask button[type=submit]");
  await until(`${COACH_TURNS} > ${before} && !coachApp.view.chatBusy`, 20000, `a reply to "${text}"`);
  return js(LAST_COACH);
}

async function gifsLoaded() {
  return until(`(() => {
    const imgs = [...document.querySelectorAll('#drawer:not([hidden]) #ex-list .ex-card img')];
    if (!imgs.length || !imgs.every((i) => i.complete && i.naturalWidth > 0)) return null;
    return imgs.map((i) => ({ src: i.getAttribute('src'), w: i.naturalWidth, h: i.naturalHeight }));
  })()`, 15000, "exercise GIFs in the drawer");
}

async function howAndGifs(key) {
  const selector = key
    ? `.card[data-key="${key}"] [data-action=how]`
    : "#groups .card [data-action=how]";
  const before = await js(COACH_TURNS);
  await click(selector);
  await until(`${COACH_TURNS} > ${before} && !coachApp.view.chatBusy`, 20000, "a reply to 'How do I fix it?'");
  let reply = await js(LAST_COACH);
  const how = reply;
  if (!reply.drawer.length) reply = await ask("What exercises help?");
  check(reply.drawer.length, `no exercise was shown; the coach said: ${reply.text}`);
  const gifs = await gifsLoaded();
  await shot("drawer-gif");
  return { how_text: how.text, how_ms: how.ms, how_actions: how.actions, shown: reply.drawer, gifs };
}

async function tooHard(rejected) {
  const reply = await ask("That one is too hard for me");
  check(reply.too_hard.includes(rejected), `too_hard is ${JSON.stringify(reply.too_hard)}, not ${rejected}`);
  check(reply.drawer.length && reply.drawer[0] !== rejected,
    `no different exercise was offered after "too hard" (drawer ${JSON.stringify(reply.drawer)}); the coach said: ${reply.text}`);
  return { reply: reply.text, ms: reply.ms, offered: reply.drawer, too_hard: reply.too_hard };
}

async function bandOnly(rejected) {
  const reply = await ask("I only have a resistance band");
  check(reply.drawer.length, `no exercise was offered for a band; the coach said: ${reply.text}`);
  const gear = await js(`coachApp.view.exercises.get(${JSON.stringify(reply.drawer[0])}).equipment || []`);
  check(gear.every((item) => item === "band"), `${reply.drawer[0]} needs ${gear.join(", ")}`);
  // When nothing else fits a band, C4 offers the rejected exercise again at its easier dose, and says so.
  const eased = reply.drawer[0] === rejected;
  if (eased) {
    const dose = await js(`coachApp.view.exercises.get(${JSON.stringify(rejected)}).easier_dose`);
    check(dose, `the exercise list gives no easier dose for ${rejected}`);
    check(/too hard/i.test(reply.text) && reply.text.includes(dose),
      `the rejected exercise ${rejected} came back without its easier dose ${dose}; the coach said: ${reply.text}`);
    const card = await js(`(() => { const c = document.querySelector('#ex-list .ex-card[data-exercise=${JSON.stringify(rejected)}]');
      return c && { dose: c.querySelector('.dose').textContent.trim(), note: (c.querySelector('.eased') || {}).textContent || null }; })()`);
    check(card && card.dose === dose && card.note && card.note.includes(dose),
      `the drawer shows ${rejected} as ${JSON.stringify(card)}, not at the easier dose ${dose}`);
  }
  return { reply: reply.text, ms: reply.ms, offered: reply.drawer, equipment: gear, eased };
}

async function plan(rejected) {
  const rejectedName = await js(`(coachApp.view.exercises.get(${JSON.stringify(rejected)}) || {}).name || null`);
  // C4 plans only from findings marked improve or priority that name a fix; pick a player who has one.
  const who = await js(`(() => {
    const b = coachApp.view.bundle;
    const needs = (role) => b.findings.some((f) => f.player === role && (f.severity === 'improve' || f.severity === 'priority') && (f.fix_tags || []).length);
    return needs(coachApp.view.player) ? coachApp.view.player : (b.players.find(needs) || null);
  })()`);
  await click("#tab-findings");
  if (who && who !== await js("coachApp.view.player")) {
    // The exercise drawer covers the video column and the player switch under it: close it like a person would.
    if (await js("!document.getElementById('drawer').hidden")) await click("#drawer-close");
    await click(`.seg button[data-player=${who}]`);
    await until(`coachApp.view.player === ${JSON.stringify(who)}`, 5000, `the ${who} player`);
  }
  await click("#plan-open");
  await until("document.getElementById('plan-dialog').open", 5000, "the plan dialog");
  // Escape belongs to the dialog, not to the video behind it.
  await send("Input.dispatchKeyEvent", { type: "rawKeyDown", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 });
  await until("!document.getElementById('plan-dialog').open", 3000, "Escape to close the plan dialog");
  await click("#plan-open");
  await until("document.getElementById('plan-dialog').open", 5000, "the plan dialog again");
  const avoid = await js("(() => { const n = document.getElementById('plan-avoid'); return { shown: !n.hidden, text: n.textContent }; })()");
  await shot("plan-dialog");
  check(avoid.shown && avoid.text.includes(rejectedName), `the dialog does not mention ${rejectedName}: ${avoid.text}`);
  await js("window.__e2eOld = true");
  await click("#plan-form button[type=submit]");
  const outcome = await until(`(() => {
    if (!window.__e2eOld && location.pathname === '/plan') {
      const error = document.getElementById('plan-error');
      if (error && !error.hidden && error.textContent.trim()) return 'plan page error: ' + error.textContent.trim();
      return /practice plan for/i.test((document.getElementById('plan-heading') || {}).textContent || '') ? 'page' : null;
    }
    const message = document.getElementById('plan-msg');
    if (message && !message.hidden && message.textContent.trim()) return 'dialog: ' + message.textContent.trim();
    return null;
  })()`, 20000, "the plan page");
  if (!who) {
    // Nothing in this clip clearly needs work for either player: the honest answer is C4's NO_PLAN, not a page.
    check(/^dialog: There is nothing to plan yet/.test(outcome), `no player has an improve finding, yet: ${outcome}`);
    await shot("plan-nothing");
    return { player: null, message: outcome, dialog: avoid.text };
  }
  check(outcome === "page", `no plan page for ${who}: ${outcome}`);
  const page = await js(`(() => {
    const items = [...document.querySelectorAll('#plan-body ol.blocks li')].map((li) => li.textContent.replace(/\\s+/g, ' ').trim());
    const weeks = [...document.querySelectorAll('#plan-body h2, #plan-body h3')].map((h) => h.textContent.trim()).filter((t) => /^Week \\d/.test(t));
    return { url: location.pathname + location.search, heading: document.getElementById('plan-heading').textContent.trim(), weeks, items };
  })()`);
  await shot("plan-page");
  check(page.weeks.length && page.items.length, "the plan page shows no weeks or no exercises");
  const named = page.items.filter((item) => rejectedName && item.includes(rejectedName));
  check(named.every((item) => /easier dose/i.test(item)), `${rejectedName} is still in the plan at full dose: ${named[0]}`);
  return { player: who, url: page.url, heading: page.heading, weeks: page.weeks, items: page.items.length,
    rejected: rejectedName, rejected_lines: named, dialog: avoid.text };
}

async function sharpenKeepsPlace() {
  await until(`coachApp.view.revision >= 1 && coachApp.view.status.status === 'ready'`, 180000,
    "the key moments to be sharpened by themselves (revision 1)");
  const before = await js(`(async () => {
    const viewer = coachApp.viewer;
    viewer.leaveMoment();
    await viewer.seekFrame(45);
    const s = viewer.state();
    return { frame: s.frame, src: viewer.video.currentSrc, selected: coachApp.view.selected, shown: s.player + ':' + s.findingId,
      player: coachApp.view.player, revision: coachApp.view.revision, tier: coachApp.view.bundle.tier };
  })()`);
  check(before.frame === 45, `could not seek to frame 45 (at ${before.frame})`);
  const button = await js(`({ text: document.querySelector('#sharpen').textContent, quoted: coachApp.view.status.sharpen_all_s })`);
  const expected = button.quoted ? `Sharpen all (up to about ${button.quoted} s)` : "Sharpen all";
  check(button.text === expected, `the Sharpen all button says ${JSON.stringify(button.text)}, not ${JSON.stringify(expected)}`);
  await click("#sharpen");
  await until(`coachApp.view.revision > ${before.revision} && coachApp.view.bundle.tier === 'refined' && coachApp.view.status.status === 'ready'`,
    180000, "every frame measured");
  const after = await js(`(() => {
    const viewer = coachApp.viewer;
    const s = viewer.state();
    return { frame: s.frame, src: viewer.video.currentSrc, selected: coachApp.view.selected, shown: s.player + ':' + s.findingId,
      player: coachApp.view.player, revision: coachApp.view.revision, status_revision: coachApp.view.status.revision,
      same_bundle: viewer.bundle === coachApp.view.bundle, viewer_tier: viewer.bundle.tier, paused: viewer.video.paused,
      sharp: document.querySelectorAll('#groups .badge.sharp').length, status: ${STATUS_TEXT} };
  })()`);
  await shot("sharpened");
  check(after.frame === before.frame, `the video moved from frame ${before.frame} to ${after.frame}`);
  check(after.src === before.src, "the video was reloaded");
  check(after.selected === before.selected, `the selected card changed from ${before.selected} to ${after.selected}`);
  check(after.shown === before.shown, `the viewer changed from ${before.shown} to ${after.shown}`);
  check(after.player === before.player, "the player changed");
  check(after.revision === after.status_revision, "the cards and the status show different revisions");
  check(after.same_bundle && after.viewer_tier === "refined", "the viewer and the cards read different bundles");
  check(after.sharp > 0, "no card says Sharpened after every frame was measured");
  return { revisions: `${before.revision} -> ${after.revision}`, frame: after.frame, selected: after.selected,
    sharpened_badges: after.sharp, status: after.status, button: button.text };
}

// Opening the session again brings the chat back with its buttons, checked against the bundle now shown.
const CHAT_STATE = `({ turns: document.querySelectorAll('#chat-log .turn').length,
  buttons: document.querySelectorAll('#chat-log .turn.coach .row button').length })`;

async function reloadKeepsChat() {
  const before = await js(CHAT_STATE);
  check(before.turns > 0, "there is no chat to bring back");
  await open(`/?s=${await js(SESSION_ID)}`);
  await useful(30000);
  const after = await js(CHAT_STATE);
  check(after.turns === before.turns, `${before.turns} chat lines before opening the session again, ${after.turns} after`);
  check(!before.buttons || after.buttons, "the chat came back without its buttons");
  return { ...after, buttons_before: before.buttons };
}

async function uploadFile(path, ms) {
  await open("/");
  // A person can only pick a file once the drop zone shows; the input exists, hidden, before that.
  await until("!document.getElementById('home').hidden && !!document.getElementById('file')", 10000, "the file picker");
  await js("window.__e2eOld = true");
  const started = Date.now();
  await setFiles("#file", [path]);
  return { started };
}

async function failureText(ms) {
  return untilOrExplain(`(() => {
    const home = document.getElementById('home-error');
    if (home && !home.hidden && home.textContent.trim()) return 'home: ' + home.textContent.trim();
    const failure = ${FAILURE};
    return failure ? 'session: ' + failure : null;
  })()`, ms, "an error message");
}

// What a person would see, errors the page caught, and the server's answer over a separate connection.
const PAGE_STATE = `(() => {
  const $ = (id) => document.getElementById(id);
  const shown = (id) => ($(id) && !$(id).hidden ? $(id).textContent.replace(/\\s+/g, ' ').trim().slice(0, 160) : null);
  const v = window.coachApp && coachApp.viewer && coachApp.viewer.video;
  return { path: location.pathname + location.search, upload: shown('upload-text'), home_error: shown('home-error'),
    working: shown('working'), failure: ${FAILURE}, status: shown('status'),
    layout: !document.getElementById('layout').hidden, cards: document.querySelectorAll('#groups .card').length,
    view: window.coachApp ? { status: coachApp.view.status && coachApp.view.status.status, revision: coachApp.view.revision,
      viewer: !!coachApp.viewer } : null,
    caught: ${CAUGHT}.slice(-3).map((s) => String(s).slice(0, 300)),
    video: v ? { ready: v.readyState, network: v.networkState, error: v.error && v.error.code,
      src: v.currentSrc.replace(location.origin, '') } : null };
})()`;

async function diagnose() {
  const out = {};
  try {
    out.page = await js(PAGE_STATE, 5000);
  } catch (error) {
    out.page = `unreadable: ${error.message}`;
  }
  try {
    out.page_fetch = await js(`(async () => {
      const t = performance.now();
      try {
        const r = await fetch('/api/health', { cache: 'no-store', signal: AbortSignal.timeout(5000) });
        return { status: r.status, ms: Math.round(performance.now() - t) };
      } catch (e) { return { error: String(e), ms: Math.round(performance.now() - t) }; }
    })()`, 8000);
    out.page_resources = await js(`performance.getEntriesByType('resource').slice(-6)
      .map((e) => e.name.replace(location.origin, '') + ' ' + Math.round(e.duration) + ' ms')`, 5000);
  } catch (error) {
    out.page_fetch = `unreadable: ${error.message}`;
  }
  const sid = typeof out.page === "object" ? new URLSearchParams(out.page.path.split("?")[1] || "").get("s") : null;
  if (sid) {
    try {
      const s = await (await fetch(`${base}/api/session/${sid}`, { signal: AbortSignal.timeout(5000) })).json();
      out.server = { status: s.status, detail: s.detail, error: s.error };
    } catch (error) {
      out.server = `no answer: ${error.message}`;
    }
  }
  return out;
}

async function untilOrExplain(expression, ms, label) {
  try {
    return await until(expression, ms, label);
  } catch (error) {
    throw new Error(`${error.message}; ${JSON.stringify(await diagnose())}`);
  }
}

function writeLongVideo(path) {
  const script = [
    "import cv2, numpy as np, sys",
    "w = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (64, 36))",
    "f = np.zeros((36, 64, 3), np.uint8)",
    "[w.write(f) for _ in range(910)]",
    "w.release()",
  ].join("\n");
  const done = spawnSync(PY, ["-c", script, path], { cwd: PROJECT, encoding: "utf8" });
  check(done.status === 0, `could not write the 182 s test video: ${done.stderr}`);
}

async function layout(pages) {
  const problems = [];
  for (const [width, height] of SIZES) {
    await setSize(width, height);
    for (const [name, path, ready] of pages) {
      await open(path);
      await untilOrExplain(ready, 30000, `${name} at ${width}`);
      await sleep(300);
      const o = await js(OVERFLOW);
      if (o.scroll > o.width + 1) problems.push(`${name} ${width}: page is ${o.scroll} px wide in a ${o.width} px window`);
      if (o.count) problems.push(`${name} ${width}: ${o.count} element(s) outside the window: ${o.bad.join("; ")}`);
      await shot(`${name}-${width}`);
    }
  }
  await setSize(1440, 900);
  check(!problems.length, problems.join(" | "));
  return { sizes: SIZES.map(([w, h]) => `${w}x${h}`), pages: pages.map(([name]) => name) };
}

const READY = {
  home: "!!document.querySelector('button[data-sample]')",
  session: `document.querySelectorAll('#groups .card').length > 0 && ${VIDEO_READY}`,
  plan: "/practice plan for/i.test((document.getElementById('plan-heading') || {}).textContent || '')",
  progress: "document.querySelectorAll('.chart-card').length > 0",
};

// The viewer's stylesheet (unit C2) styles only the viewer: switching it off and on changes no computed style
// outside #viewer, and the app, not the viewer, sets the size of the viewer's box.
const STYLE_PROPS = ["display", "position", "box-sizing", "margin-top", "margin-right", "margin-bottom", "margin-left",
  "padding-top", "padding-right", "padding-bottom", "padding-left", "border-top-width", "border-top-style",
  "border-top-color", "border-top-left-radius", "font-family", "font-size", "font-weight", "font-style", "line-height",
  "letter-spacing", "text-transform", "white-space", "color", "background-color", "opacity", "cursor", "overflow-x",
  "overflow-y", "gap", "align-items", "justify-content"];
const STYLE_LEAKS = `(() => {
  const sheet = [...document.styleSheets].find((s) => s.href && /\\/static\\/viewer[^/]*\\.css$/.test(s.href));
  if (!sheet) return null;
  const root = document.getElementById('viewer');
  const nodes = [document.documentElement, document.body, ...document.body.querySelectorAll('*')]
    .filter((n) => !(root && root.contains(n)));
  const props = ${JSON.stringify(STYLE_PROPS)};
  const name = (n) => n.tagName.toLowerCase() + (n.id ? '#' + n.id : '') + [...n.classList].map((c) => '.' + c).join('');
  const snap = () => ({
    styles: nodes.map((n) => { const cs = getComputedStyle(n); return props.map((p) => cs.getPropertyValue(p)); }),
    box: root ? [root.offsetWidth, root.offsetHeight] : null,
  });
  const on = snap();
  sheet.disabled = true;
  const off = snap();
  sheet.disabled = false;
  const changed = [];
  nodes.forEach((n, i) => props.forEach((p, j) => {
    if (on.styles[i][j] !== off.styles[i][j]) changed.push(name(n) + ' ' + p + ': ' + off.styles[i][j] + ' -> ' + on.styles[i][j]);
  }));
  return { sheet: new URL(sheet.href).pathname, checked: nodes.length, changed, box_on: on.box, box_off: off.box };
})()`;

// app.html serves home and session and carries the viewer; plan.html and progress.html have no viewer.
const VIEWER_PAGES = new Set(["home", "session"]);

async function viewerStylesStayInTheBox(pages) {
  const seen = [];
  for (const [name, path, ready] of pages) {
    await open(path);
    await untilOrExplain(ready, 30000, name);
    const r = await js(STYLE_LEAKS);
    if (!r) {
      check(!VIEWER_PAGES.has(name), `${name}: the page has no viewer stylesheet`);
      seen.push({ page: name, sheet: null });
      continue;
    }
    check(!r.changed.length, `${name}: the viewer stylesheet changes ${r.changed.length} style(s) outside the viewer: ${r.changed.slice(0, 8).join("; ")}`);
    if (r.box_on) {
      check(Math.abs(r.box_on[0] - r.box_off[0]) <= 1 && Math.abs(r.box_on[1] - r.box_off[1]) <= 1,
        `${name}: the viewer stylesheet resizes the viewer's box from ${r.box_off} to ${r.box_on}`);
    }
    seen.push({ page: name, sheet: r.sheet, checked: r.checked, box: r.box_on });
  }
  return { pages: seen };
}

// --- the two flows ---------------------------------------------------------------------------------------------------

async function replayFlow() {
  await step("warm-up", () => waitFor(async () => {
    const health = await (await fetch(`${base}/api/health`)).json();
    return health.warm.state === "ready" ? { warm: health.warm } : null;
  }, 120000, "the analyser warm-up"), { critical: true });

  await step("home", async () => {
    await open("/");
    await until(READY.home, 15000, "sample buttons");
    await shot("home");
  }, { critical: true });

  let sid = null;
  await step("sample to first useful screen", async () => {
    const { identifier } = await send("Page.addScriptToEvaluateOnNewDocument", { source: MARK_USEFUL });
    await js("window.__e2eOld = true");
    const started = Date.now();
    try {
      await click('button[data-sample="01-singles-rally"]');
      await until(`!window.__e2eOld && !!${SESSION_ID}`, 15000, "the session page");
      const s = await useful(120000);
      summary.timings.sample_to_first_useful_ms = Date.now() - started;
      sid = await js(SESSION_ID);
      const b = await until(`(() => { const r = ${USEFUL_BREAKDOWN}; return r.useful == null ? null : r; })()`, 5000,
        "the in-page first-useful mark");
      const page = Math.round(b.origin - started);
      summary.timings.first_useful = { click_to_page_start_ms: page, click_to_useful_ms: page + b.useful,
        page: { html: b.html, dom_content_loaded: b.dom_content_loaded, static: b.static, exercises: b.exercises,
          status: b.status, bundle: b.bundle, video_element: b.video_element, video_events: b.video_events,
          cards: b.cards, video: b.video, useful: b.useful } };
      await shot("session-first");
      return { session: sid, ms: summary.timings.sample_to_first_useful_ms, first_useful: summary.timings.first_useful,
        cards: s.cards, tier: s.tier, status: await js(STATUS_TEXT) };
    } finally {
      await send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
    }
  }, { critical: true });

  await step("name the near player", nameNear);
  await step("worth knowing is in plain words", plainWarnings);
  const shown = await step("slow motion then freeze", moment);
  await step("the cards follow the viewer", viewerNext);
  const help = await step("how do I fix it, with GIFs", () => howAndGifs(shown && shown.finding));
  const rejected = help && help.shown[0];
  if (rejected) {
    await step("too hard", () => tooHard(rejected));
    await step("only a band", () => bandOnly(rejected));
  }
  await step("typing stays in the chat", keysStayInTheChat);
  await step("sharpen keeps the place", sharpenKeepsPlace);
  await step("reload keeps the chat", reloadKeepsChat);
  let planUrl = null;
  if (rejected) {
    const made = await step("plan avoids the rejected exercise", () => plan(rejected));
    planUrl = made && made.url;
  }

  let second = null;
  await step("real upload of 02-singles-dive.mp4", async () => {
    const { started } = await uploadFile(join(PROJECT, "sample-videos", "02-singles-dive.mp4"));
    await until(`!window.__e2eOld && !!${SESSION_ID}`, 60000, "the session page after upload");
    const s = await useful(120000);
    summary.timings.upload_to_first_useful_ms = Date.now() - started;
    second = await js(SESSION_ID);
    await nameNear();
    return { session: second, ms: summary.timings.upload_to_first_useful_ms, cards: s.cards, status: await js(STATUS_TEXT) };
  });

  await step("progress joins both sessions", async () => {
    await open(`/progress?player=${encodeURIComponent(PLAYER)}`);
    await until(READY.progress, 15000, "progress charts");
    const p = await js(`({ charts: document.querySelectorAll('.chart-card').length,
      points: Math.max(0, ...[...document.querySelectorAll('.chart-card')].map((c) => c.querySelectorAll('circle.pt').length)) })`);
    await shot("progress");
    check(p.points >= 2, `the best chart has ${p.points} point(s)`);
    return p;
  });

  await step("not a video", async () => {
    const path = join(tmp, "hello.mp4");
    writeFileSync(path, "hello");
    await uploadFile(path);
    const text = await failureText(60000);
    await shot("error-not-a-video");
    check(/not a video/i.test(text), text);
    return { message: text };
  });

  await step("longer than 3 minutes", async () => {
    const path = join(tmp, "long-182s.mp4");
    writeLongVideo(path);
    await uploadFile(path);
    const text = await failureText(60000);
    await shot("error-too-long");
    check(/longer than 3 minutes/i.test(text), text);
    return { message: text };
  });

  await step("not a video file type", async () => {
    const path = join(tmp, "notes.txt");
    writeFileSync(path, "notes");
    await uploadFile(path);
    const text = await failureText(10000);
    check(/\.mp4, \.mov or \.webm/.test(text), text);
    check(await js(`!${SESSION_ID}`), "a .txt file started a session");
    return { message: text };
  });

  const pages = [["home", "/", READY.home]];
  if (sid) pages.push(["session", `/?s=${sid}`, READY.session]);
  if (planUrl) pages.push(["plan", planUrl, READY.plan]);
  if (second) pages.push(["progress", `/progress?player=${encodeURIComponent(PLAYER)}`, READY.progress]);
  await step("layout at three sizes", () => layout(pages));
  await step("the viewer's styles stay in its box", () => viewerStylesStayInTheBox(pages));
}

async function liveFlow(sid) {
  await step("open the session", async () => {
    const started = Date.now();
    await open(`/?s=${sid}`);
    const s = await useful(60000);
    summary.timings.open_session_ms = Date.now() - started;
    const counts = await js(`(() => {
      const out = {};
      for (const f of coachApp.view.bundle.findings) out[f.player + ' ' + f.severity] = (out[f.player + ' ' + f.severity] || 0) + 1;
      return out;
    })()`);
    await shot("session");
    return { ms: summary.timings.open_session_ms, cards: s.cards, tier: s.tier, revision: s.revision, severities: counts,
      status: await js(STATUS_TEXT), warm: await js("document.getElementById('warm').textContent") };
  }, { critical: true });

  await step("worth knowing is in plain words", plainWarnings);
  const shown = await step("slow motion then freeze", moment);
  await step("the cards follow the viewer", viewerNext);
  const help = await step("how do I fix it, with GIFs", () => howAndGifs(shown && shown.finding));
  const rejected = help && help.shown[0];
  let planUrl = null;
  if (rejected) {
    await step("too hard", () => tooHard(rejected));
    await step("only a band", () => bandOnly(rejected));
  }
  await step("typing stays in the chat", keysStayInTheChat);
  if (rejected) {
    const made = await step("plan avoids the rejected exercise", () => plan(rejected));
    planUrl = made && made.url;
  }
  const pages = [["home", "/", READY.home], ["session", `/?s=${sid}`, READY.session]];
  if (planUrl) pages.push(["plan", planUrl, READY.plan]);
  await step("layout at three sizes", () => layout(pages));
  await step("the viewer's styles stay in its box", () => viewerStylesStayInTheBox(pages));
}

async function refusedUploadFlow(path, expect) {
  await step(`the app refuses it with the plain ${expect} message`, async () => {
    const { started } = await uploadFile(path);
    await until(`!window.__e2eOld && !!${SESSION_ID}`, 60000, "the session page after upload");
    const id = await js(SESSION_ID);
    const text = await failureText(600000);
    const ms = Date.now() - started;
    await shot("error");
    const server = await (await fetch(`${base}/api/session/${id}`)).json();
    check(server.status === "error", `the server says ${server.status}`);
    check(server.error.code === expect, `the server's error is ${server.error.code}, not ${expect}`);
    check(text.startsWith("session: ") && text.includes(server.error.message),
      `the page says "${text}", the server "${server.error.message}"`);
    const note = await js(NOTE);
    check(note === null, `a note from the analysis still shows above the refusal: "${note}"`);
    return { session: id, upload_to_message_ms: ms, message: text, code: server.error.code,
      n_frames: server.n_frames, fps: server.fps, duration_s: server.duration_s };
  }, { critical: true });

  await step("the app still answers", async () => {
    const health = await (await fetch(`${base}/api/health`)).json();
    await open("/");
    await until(READY.home, 15000, "the home page");
    return { warm: health.warm && health.warm.state };
  });
}

async function main() {
  mkdirSync(SHOTS, { recursive: true });
  if (args.url) {
    base = args.url.replace(/\/+$/, "");
  } else {
    app = await startApp();
    base = app.url;
  }
  summary.url = base;
  summary.backend = (await (await fetch(`${base}/api/health`)).json()).backend;
  edge = await startEdge(1440, 900);
  send = attach(edge.ws);
  send.on("Runtime.exceptionThrown", (p) => summary.page_errors.push({ step: current,
    text: (p.exceptionDetails.exception && p.exceptionDetails.exception.description) || p.exceptionDetails.text }));
  send.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error") summary.page_errors.push({ step: current, text: p.args.map((a) => a.value ?? a.description).join(" ") });
  });
  send.on("Log.entryAdded", (p) => {
    // Edge's own pages (the first blank tab loads edge:// theme files) are not the app.
    if (/\b(edge|chrome):\/\//.test(`${p.entry.url || ""} ${p.entry.text}`)) return;
    if (p.entry.level === "error") summary.page_errors.push({ step: current, text: `${p.entry.text} ${p.entry.url || ""}`.trim() });
  });
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Log.enable");
  await send("DOM.enable");
  await setSize(1440, 900);
  if (args.upload) await refusedUploadFlow(args.upload, args.expect ?? "no_court");
  else if (args.url) await liveFlow(args.session);
  else await replayFlow();
}

let crashed = null;
try {
  await main();
} catch (error) {
  crashed = error.message;
  log("STOPPED", error.message);
} finally {
  await stopEdge();
  await stopApp();
  const failed = summary.steps.filter((s) => !s.ok).map((s) => s.name);
  summary.finished = new Date().toISOString();
  summary.result = crashed || failed.length || summary.page_errors.length ? "fail" : "pass";
  summary.failed = failed;
  if (crashed) summary.stopped = crashed;
  writeFileSync(join(SHOTS, `${PREFIX}summary.json`), JSON.stringify(summary, null, 2));
  log(`${summary.result}: ${summary.steps.filter((s) => s.ok).length}/${summary.steps.length} steps, ` +
    `${summary.page_errors.length} page error(s); ${join(SHOTS, `${PREFIX}summary.json`)}`);
  for (let i = 0; i < 3; i++) {
    try {
      rmSync(tmp, { recursive: true, force: true });
      break;
    } catch {
      await sleep(1000);
    }
  }
  process.exitCode = summary.result === "pass" ? 0 : 1;
}
