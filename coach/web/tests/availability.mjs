import assert from "node:assert/strict";

function element(tag) {
  const children = [];
  return {
    tag,
    className: "",
    textContent: "",
    type: "",
    dataset: {},
    width: 0,
    height: 0,
    children,
    childNodes: children,
    classList: { add() {} },
    append(...nodes) {
      children.push(...nodes);
    },
    replaceChildren(...nodes) {
      children.splice(0, children.length, ...nodes);
    },
    addEventListener() {},
  };
}

function texts(el) {
  const out = [];
  if (el.textContent) out.push(el.textContent);
  for (const child of el.children || []) out.push(...texts(child));
  return out;
}

globalThis.window = {
  __viewerErrors: [],
  addEventListener() {},
  matchMedia() {
    return { matches: false };
  },
};
globalThis.document = {
  querySelector() {
    return null;
  },
  createElement(tag) {
    return element(tag);
  },
};
globalThis.location = { search: "", href: "http://127.0.0.1/viewer.html" };

const { Viewer, normalizeBundle } = await import("../viewer.js");

function metric(overrides) {
  return {
    name: "hip_height_ratio",
    label: "Hip height",
    value: 1.2,
    unit: "ratio",
    basis: "projected_2d",
    confidence: 0,
    target: [0.42, 0.48],
    series: "hip_height_ratio",
    n_obs: 0,
    best: null,
    ...overrides,
  };
}

function bundle(version, row, finding = {}) {
  return {
    schema_version: version,
    findings: [{
      id: "technique.stay_lower",
      severity: "info",
      score: 0,
      moments: [],
      ...finding,
      metrics: finding.metrics || [row],
    }],
  };
}

let passed = 0;

function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok ${name}`);
}

check("version 2 placeholder clears and leaves the input", () => {
  const row = metric();
  const source = bundle(2, row);
  const cleared = normalizeBundle(source);
  assert.equal(cleared.findings[0].metrics[0].value, null);
  assert.equal(cleared.schema_version, 2);
  assert.equal(row.value, 1.2);
  assert.notEqual(cleared.findings[0].metrics[0], row);
  assert.equal(cleared.findings[0].metrics[0].label, "Hip height");
  assert.equal(cleared.findings[0].metrics[0].n_obs, 0);
});

check("version 2 observed confidence 0 stays", () => {
  const stayed = normalizeBundle(bundle(2, metric({ value: 0.5, n_obs: 4, confidence: 0, best: null })));
  assert.equal(stayed.findings[0].metrics[0].value, 0.5);
  assert.equal(stayed.findings[0].metrics[0].confidence, 0);
});

check("missing n_obs is not a zero observation", () => {
  const row = metric({ n_obs: undefined });
  delete row.n_obs;
  const stayed = normalizeBundle(bundle(2, row));
  assert.equal(stayed.findings[0].metrics[0].value, 1.2);
});

check("version 3 numeric zero observation is rejected and not cleared", () => {
  const row = metric({ value: 1.2, n_obs: 0, confidence: 0, best: null });
  const source = bundle(3, row);
  assert.throws(() => normalizeBundle(source), /n_obs/);
  assert.equal(row.value, 1.2);
});

check("version 3 null observation stays null", () => {
  const stayed = normalizeBundle(bundle(3, metric({ value: null, n_obs: 0, confidence: 0, best: null })));
  assert.equal(stayed.findings[0].metrics[0].value, null);
});

check("non-finite placeholder throws", () => {
  assert.throws(() => normalizeBundle(bundle(2, metric({ value: Number.NaN }))));
});

check("missing schema_version throws", () => {
  assert.throws(() => normalizeBundle({ findings: [{ metrics: [metric()] }] }));
});

function paint(row) {
  const finding = {
    id: "technique.stay_lower",
    player: "near",
    headline: "Sit deeper between shots",
    explain: "The clip has no measured reading.",
    category: "technique",
    metrics: [row],
    moments: [],
    timelines: [],
    overlay_series: [],
    exercise_ids: [],
  };
  const viewer = new Viewer();
  viewer.findingId = finding.id;
  viewer.player = finding.player;
  viewer.bundle = { findings: [finding], warnings: [] };
  viewer.panel = element("div");
  viewer.renderPanel();
  return texts(viewer.panel);
}

check("missing value key is rejected", () => {
  for (const version of [2, 3]) {
    const row = metric({ n_obs: 0, confidence: 0, best: null });
    delete row.value;
    const source = bundle(version, row);
    assert.throws(() => normalizeBundle(source), /missing value/);
    assert.equal(Object.hasOwn(source.findings[0].metrics[0], "value"), false);
  }
});

check("bad moments on an all-missing finding are rejected", () => {
  for (const version of [2, 3]) {
    for (const moments of [0, false, "", "no", { start_frame: 1 }]) {
      const source = bundle(version, metric({ value: null, n_obs: 0, confidence: 0, best: null }), { moments });
      assert.throws(() => normalizeBundle(source), /moments must be a list/);
    }
    for (const omit of [true, false]) {
      const finding = {
        id: "technique.stay_lower",
        severity: "info",
        score: 0,
        metrics: [metric({ value: null, n_obs: 0, confidence: 0, best: null })],
      };
      if (!omit) finding.moments = null;
      const source = { schema_version: version, findings: [finding] };
      assert.throws(() => normalizeBundle(source), /moments must be a list/);
    }
    const empty = normalizeBundle(bundle(version, metric({ value: null, n_obs: 0, confidence: 0, best: null }), { moments: [] }));
    assert.equal(empty.findings[0].metrics[0].value, null);
    assert.deepEqual(empty.findings[0].moments, []);
  }
});

check("version 3 null with observations is rejected", () => {
  const source = bundle(3, metric({ value: null, n_obs: 3, confidence: 0.8, best: null }));
  assert.throws(() => normalizeBundle(source), /n_obs/);
  assert.equal(source.findings[0].metrics[0].value, null);
});

check("null with confidence or best is rejected", () => {
  for (const version of [2, 3]) {
    assert.throws(() => normalizeBundle(bundle(version, metric({ value: null, n_obs: 0, confidence: 0.4, best: null }))));
    assert.throws(() => normalizeBundle(bundle(version, metric({ value: null, n_obs: 0, confidence: 0, best: 0.2 }))));
  }
});

check("malformed observation count is rejected", () => {
  assert.throws(() => normalizeBundle(bundle(3, metric({ value: 0.5, n_obs: 1.5, confidence: 0.8, best: null }))), /integer/);
});

check("all-missing finding that overclaims is rejected", () => {
  const moment = [{ player: "near", start_frame: 1, key_frame: 2, end_frame: 3, label: "Hip" }];
  const cases = [
    [{ severity: "strength" }, /must be info/],
    [{ moments: moment }, /must not have a moment/],
    [{ score: 0.0005 }, /score must be 0/],
  ];
  for (const version of [2, 3]) {
    for (const [patch, pattern] of cases) {
      const row = metric({ value: version === 2 ? 1.2 : null, n_obs: 0, confidence: 0, best: null });
      const source = bundle(version, row, patch);
      assert.throws(() => normalizeBundle(source), pattern);
      assert.equal(source.findings[0].metrics[0].value, row.value);
    }
  }
});

check("measured confidence 0 and mixed findings stay", () => {
  const measured = normalizeBundle(bundle(2, metric({ value: 0.2, n_obs: 4, confidence: 0, best: 0.1 }), {
    severity: "improve",
    score: 0.4,
    moments: [{ player: "near", start_frame: 1, key_frame: 2, end_frame: 3, label: "Hip" }],
  }));
  assert.equal(measured.findings[0].metrics[0].value, 0.2);
  assert.equal(measured.findings[0].severity, "improve");
  assert.equal(measured.findings[0].moments.length, 1);

  const absent = metric({ name: "contact_reach", label: "Reach", value: 1.2, n_obs: 0, confidence: 0, best: null });
  const seen = metric({ value: 0.5, n_obs: 4, confidence: 0, best: 0.4 });
  const mixed = normalizeBundle(bundle(2, absent, {
    severity: "strength",
    score: 0,
    moments: [],
    metrics: [absent, seen],
  }));
  assert.equal(mixed.findings[0].metrics[0].value, null);
  assert.equal(mixed.findings[0].metrics[1].value, 0.5);
  assert.equal(mixed.findings[0].severity, "strength");
});

check("missing reading omits your number and keeps the guide", () => {
  const painted = paint(metric({ value: null, n_obs: 0, confidence: 0, best: null }));
  assert.ok(painted.includes("Sit deeper between shots"));
  assert.ok(painted.includes("Practice guide"));
  assert.equal(painted.includes("Your number, clip summary"), false);
  assert.equal(painted.includes("Hip height"), false);
  assert.equal(painted.some((text) => text === "low" || text === "medium" || text === "high"), false);
});

check("finite confidence 0 still shows the number", () => {
  const painted = paint(metric({ value: 0.5, n_obs: 4, confidence: 0, best: null }));
  assert.ok(painted.includes("Your number, clip summary"));
  assert.ok(painted.some((text) => text.includes("0.5")));
  assert.ok(painted.includes("low"));
});

console.log(`${passed} passed`);
