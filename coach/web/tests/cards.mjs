// The finding cards of the session page: app.js runs unchanged in a Node vm context with stub elements, and card()
// gets findings shaped like the viewer's normalized bundle.
//
//   node coach\web\tests\cards.mjs
//
// A reading the analysis could not take says "not measured in this clip" next to the practice guide and shows no
// number. A measured reading shows its number, even when its confidence is 0.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

class Node {}

function element(tag) {
  return Object.assign(new Node(), {
    tag, className: "", textContent: "", dataset: {}, attributes: {}, children: [],
    setAttribute(key, value) { this.attributes[key] = value; },
    append(...nodes) { this.children.push(...nodes); },
    addEventListener() {},
  });
}

function texts(node) {
  if (typeof node === "string") return [node];
  return [node.textContent, ...node.children.flatMap(texts)].filter(Boolean);
}

function find(node, className) {
  if (typeof node === "string") return [];
  const here = node.className.split(" ").includes(className) ? [node] : [];
  return here.concat(node.children.flatMap((child) => find(child, className)));
}

const context = vm.createContext({
  Node,
  window: {},
  location: { search: "" },
  document: { body: element("body"), createElement: element, getElementById: () => null },
});
vm.runInContext(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "app.js"), "utf8"), context,
  { filename: "app.js" });

function card(finding) {
  context.finding = finding;
  return vm.runInContext("card(finding)", context);
}

const HIP = { name: "ready_hip_ratio", label: "Projected hip height", unit: "heights", basis: "projected_2d",
  target: [0.4, 0.5], series: "ready_hip_ratio" };
const REACH = { name: "contact_reach", label: "Overhead reach", unit: "heights", basis: "projected_2d",
  target: [1.0, 1.4], series: "contact_reach" };
const MISSING = { value: null, n_obs: 0, confidence: 0, best: null };

function finding(metrics, extra = {}) {
  return { id: "technique.stay_lower", player: "near", category: "technique", severity: "info", score: 0,
    headline: "Sit deeper while you wait", explain: "Lower hips get you to the shuttle sooner.", metrics,
    moments: [], exercise_ids: [], ...extra };
}

let passed = 0;

function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok ${name}`);
}

check("a reading that was not taken shows the guide and no number", () => {
  const node = card(finding([{ ...HIP, ...MISSING }]));
  assert.deepEqual(find(node, "metric").map((p) => p.textContent),
    ["Projected hip height: not measured in this clip · practice guide 0.40–0.50 × body height"]);
  assert.equal(texts(node).some((text) => /\d\.\d\d × body height ·/.test(text)), false);
  assert.equal(texts(node).includes("Show me in slow motion"), false);
});

check("a measured reading with confidence 0 shows its number", () => {
  const node = card(finding([{ ...HIP, value: 0.47, n_obs: 4, confidence: 0, best: 0.44 }],
    { severity: "improve", score: 0.3 }));
  assert.deepEqual(find(node, "metric").map((p) => p.textContent),
    ["Projected hip height: 0.47 × body height · practice guide 0.40–0.50 × body height"]);
});

check("a mixed finding shows the measured reading and never the missing one", () => {
  const node = card(finding([{ ...REACH, ...MISSING }, { ...HIP, value: 0.47, n_obs: 4, confidence: 0.8, best: 0.44 }],
    { severity: "improve", score: 0.3 }));
  assert.deepEqual(find(node, "metric").map((p) => p.textContent),
    ["Projected hip height: 0.47 × body height · practice guide 0.40–0.50 × body height"]);
  assert.equal(texts(node).some((text) => text.includes("Overhead reach")), false);
});

check("a finding without metrics has no reading line", () => {
  assert.deepEqual(find(card(finding([])), "metric"), []);
});

console.log(`${passed} passed`);
