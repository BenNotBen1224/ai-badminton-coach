// The session page when the app refuses a clip: app.js runs unchanged in a Node vm context, on stub elements made
// from the ids in app.html, each starting hidden or shown as the page does.
//
//   node coach\web\tests\refusal.mjs
//
// While a long clip is being analysed, the page notes that it takes longer. Once the app refuses the clip, that note
// is gone and the page shows the app's refusal message unchanged. The refused session is the saved state from the
// black-video check (work\C6\black-check\black-check.json, sessions[0]); this test analyses nothing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));

class Node {}

function element(hidden = false) {
  return Object.assign(new Node(), {
    hidden, disabled: false, title: "", className: "", textContent: "", dataset: {}, children: [],
    setAttribute() {},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    addEventListener() {},
  });
}

const page = new Map();
for (const [tag, id] of readFileSync(join(here, "..", "app.html"), "utf8").matchAll(/<\w+[^>]*?\sid="([\w-]+)"[^>]*>/g)) {
  page.set(id, element(/\shidden(?=[\s>=/])/.test(tag)));
}
const $ = (id) => page.get(id);

const context = vm.createContext({
  Node,
  performance,
  window: {},
  location: { search: "" },
  // The "so far" counter of the working line is not run.
  setTimeout: () => 0,
  clearTimeout: () => {},
  document: {
    title: "", body: element(), createElement: () => element(), getElementById: (id) => page.get(id) ?? null,
    querySelectorAll: () => [],
  },
});
vm.runInContext(readFileSync(join(here, "..", "app.js"), "utf8"), context, { filename: "app.js" });
// What /api/health tells the page (server.py health(); TESTED_SECONDS is 36 in pipeline.py).
vm.runInContext(`health = { tested_seconds: 36, export: true, plan: true,
  backend: { name: "turbo", label: "Turbo (resident preview, then sharpened)", sharpens_windows: true } }`, context);

const SAVED = {
  id: "b01dd9ea71c16b84",
  status: "error",
  error: {
    code: "no_court",
    message: "I could not find the court in this video; film with a still camera behind one baseline so the whole court is in view",
  },
  n_frames: null,
  fps: null,
  duration_s: 180.0,
  timings: {
    first: {
      probe: 0.0005870999302715063, total: 8.051377000054345, backend: "turbo", upload_s: 0.012309799902141094,
      queue_s: 0.0008642000611871481, warm_wait_s: 0.0,
    },
  },
};
// The other fields the server sends with every status (server.py status()). A 180 s clip is longer than the tested
// 36 s, and before any result the session's timings are the store's empty default.
const SENT = { video_name: "black-180s.mp4", detail: "", notice: "", revision: null, labels: {}, long_clip: true };
const ANALYSING = { ...SAVED, ...SENT, status: "preview", detail: "Analysing the clip", error: null, timings: {} };
const REFUSED = { ...SAVED, ...SENT };

async function show(status) {
  context.nextStatus = status;
  await vm.runInContext("applyStatus(nextStatus)", context);
}

let passed = 0;

function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok ${name}`);
}

await show(ANALYSING);
check("while the long clip is analysed, the page notes that it takes longer", () => {
  assert.equal($("notice").hidden, false);
  assert.equal($("notice").textContent, "This clip is longer than 36 s, so it takes longer to analyse.");
  assert.equal($("working").hidden, false);
  assert.equal($("failure").hidden, true);
});

await show(REFUSED);
check("once the clip is refused, the note is gone", () => {
  assert.equal($("notice").hidden, true);
});

check("the refusal shows the app's message unchanged", () => {
  assert.equal($("failure").hidden, false);
  assert.equal($("failure-title").textContent, "This clip could not be analysed");
  assert.equal($("failure-text").textContent, SAVED.error.message);
  assert.equal($("failure-foot").hidden, false);
  assert.equal($("working").hidden, true);
  assert.equal($("layout").hidden, true);
});

console.log(`${passed} passed`);
