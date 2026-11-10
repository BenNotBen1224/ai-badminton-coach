// The coach panel after a stop for pain: app.js runs unchanged in a Node vm context, on stub elements made from the
// ids in app.html, with the app's answers stubbed. Nothing is analysed and no coach runs.
//
//   node coach\web\tests\painstop.mjs
//
// The coach's stop reply has no suggestions, and the server keeps the stop for the whole conversation. The page must
// not fill that gap with its starter chips: not after the stop, not after a context the app refused, not when the
// tab is reloaded, and not in a new tab that has no saved chat. A conversation without a stop keeps them.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "..", "app.html"), "utf8");
const source = readFileSync(join(here, "..", "app.js"), "utf8");
const SAFETY = JSON.parse(readFileSync(join(here, "..", "..", "kb", "safety.json"), "utf8")).text;
const STARTERS = ["What should I work on first?", "How do I fix it?", "Show me the moment", "What exercises help?",
  "What did I do well?", "What can't you measure?"];
const ID = "0123456789abcdef";
const OTHER = "fedcba9876543210";

class Node {}

function element(hidden = false) {
  return Object.assign(new Node(), {
    hidden, disabled: false, title: "", className: "", textContent: "", dataset: {}, children: [],
    classList: { toggle() {} },
    setAttribute() {},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    addEventListener() {},
    focus() {},
    remove() {},
  });
}

// What a node shows: its own text, or its children's, as a turn keeps its words in a span and a chip in its label.
const text = (node) => (typeof node === "string" ? node
  : node.children.length ? node.children.map(text).join("") : node.textContent);

const settle = () => new Promise((resolve) => setImmediate(resolve));

// One load of app.html?s=<id> in a tab whose sessionStorage is `tab`, so a second load with the same map is a reload.
// Without `firstStatus` the session's status never comes back, so the page stays as it loads. With it, the first
// status is `firstStatus`; the viewer cannot load in this stub, which leaves the chips alone.
async function load(tab, id = ID, firstStatus = null) {
  const page = new Map();
  for (const [tag, name] of html.matchAll(/<\w+[^>]*?\sid="([\w-]+)"[^>]*>/g)) {
    page.set(name, element(/\shidden(?=[\s>=/])/.test(tag)));
  }
  const asked = [];
  const answers = [];
  const answer = (status, body) => ({ ok: status < 400, status, json: async () => body });
  const fetch = async (path) => {
    asked.push(path);
    if (path === "/api/health") return answer(200, { tested_seconds: 36, export: true, plan: true });
    if (path === "/api/exercises") return answer(200, { exercises: [] });
    if (path === "/api/chat") return answer(...answers.shift());
    if (path === `/api/session/${id}` && firstStatus) return answer(200, firstStatus);
    return new Promise(() => {});
  };
  const context = vm.createContext({
    Node,
    URLSearchParams,
    performance,
    fetch,
    window: {},
    location: { search: `?s=${id}` },
    setTimeout: () => 0,
    clearTimeout: () => {},
    sessionStorage: { getItem: (key) => tab.get(key) ?? null, setItem: (key, value) => tab.set(key, String(value)) },
    document: {
      title: "", body: element(), createElement: () => element(), getElementById: (name) => page.get(name) ?? null,
      querySelectorAll: () => [], addEventListener() {},
    },
  });
  vm.runInContext(source, context, { filename: "app.js" });
  // The chips are set before the page first asks for the session's status.
  for (let i = 0; !asked.includes(`/api/session/${id}`); i += 1) {
    assert.ok(i < 1000, `the page never asked for the session; it asked for ${asked.join(", ")}`);
    await settle();
  }
  for (let i = 0; firstStatus && !vm.runInContext("view.status != null", context); i += 1) {
    assert.ok(i < 1000, "the page never took the session's status");
    await settle();
  }
  // These replies carry no actions, so an empty bundle is enough to let the page send a question.
  vm.runInContext("view.bundle = { findings: [], players: {} }", context);
  return {
    chips: () => page.get("chips").children.map(text),
    context: () => JSON.parse(vm.runInContext("JSON.stringify(view.context)", context)),
    lastTurn: () => text(page.get("chat-log").children.at(-1)),
    async ask(question, status, body) {
      answers.push([status, body]);
      context.question = question;
      await vm.runInContext("sendChat(question)", context);
    },
  };
}

// After a stop the coach keeps pain_stop in every context it returns (unit C4's contract).
const stopped = (said, intent, suggestions = [], offer = null) => ({
  reply: { text: said, intent, confidence: 1, actions: [], suggestions },
  context: { player: "near", last_offer: offer, pain_stop: true },
});

let passed = 0;

function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok ${name}`);
}

const tab = new Map();
const first = await load(tab);
check("a new conversation starts with the starter chips", () => {
  assert.deepEqual(first.chips(), STARTERS);
});

await first.ask("What should I work on first?", 200, {
  reply: { text: "Start with how low you stay.", intent: "summary", confidence: 1, actions: [], suggestions: [] },
  context: { player: "near", last_offer: null, pain_stop: false },
});
check("before a stop, a reply without suggestions brings the starter chips back", () => {
  assert.deepEqual(first.chips(), STARTERS);
});

await first.ask("my knee has been painful since the lunges", 200, stopped(SAFETY, "safety"));
check("after the stop, the page offers no chip", () => {
  assert.equal(first.lastTurn(), SAFETY);
  assert.deepEqual(first.chips(), []);
});

await first.ask("give me a drill", 200, stopped(SAFETY, "exercise_request"));
check("a training question after the stop gets the stop again and still no chip", () => {
  assert.equal(first.lastTurn(), SAFETY);
  assert.deepEqual(first.chips(), []);
});

await first.ask("What did I do well?", 200, stopped("You stayed low through the lunge.", "strengths",
  ["Show me my best one"], "show_best"));
check("after the stop, the coach's own suggestions are shown and nothing else", () => {
  assert.deepEqual(first.chips(), ["Show me my best one"]);
});

await first.ask("yes", 400, { error: "the chat lost its place; reload the page" });
check("a context the app refused is reset but keeps the stop", () => {
  assert.equal(first.lastTurn(), "the chat lost its place; reload the page");
  assert.deepEqual(first.context(), { player: "near", pain_stop: true });
});

const reloaded = await load(tab);
check("reloading the tab after the stop shows no starter chip", () => {
  assert.deepEqual(reloaded.context(), { player: "near", pain_stop: true });
  assert.deepEqual(reloaded.chips(), []);
});

const another = await load(tab, OTHER);
check("another clip's conversation in the same tab still starts with the starter chips", () => {
  assert.deepEqual(another.chips(), STARTERS);
});

// A new tab has no saved chat, so only the app's status says the coach has stopped.
const ready = (id, chatStopped) => ({ id, stamp: 1, video_name: "rally.mp4", status: "ready", revision: 0,
  bundle_url: `/api/session/${id}/bundle?revision=0`, chat_stopped: chatStopped });
const secondTab = new Map();
const newTab = await load(secondTab, ID, ready(ID, true));
check("a new tab on the stopped conversation shows no starter chip", () => {
  assert.deepEqual(newTab.context(), { pain_stop: true });
  assert.deepEqual(newTab.chips(), []);
});

const newTabElsewhere = await load(new Map(), OTHER, ready(OTHER, false));
check("a new tab on a conversation without a stop still shows the starter chips", () => {
  assert.deepEqual(newTabElsewhere.context(), {});
  assert.deepEqual(newTabElsewhere.chips(), STARTERS);
});

// That tab saved the stop, so a reload shows no starter chip even before the app's status comes back.
const newTabReloaded = await load(secondTab);
check("reloading the new tab keeps the stop before the app answers", () => {
  assert.deepEqual(newTabReloaded.context(), { pain_stop: true });
  assert.deepEqual(newTabReloaded.chips(), []);
});

console.log(`${passed} passed`);
