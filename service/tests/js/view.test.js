"use strict";
// Run from webservice/: node --test tests/js/view.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const V = require(path.join(__dirname, "..", "..", "static", "view.js"));

const FIXTURES = path.join(__dirname, "fixtures");
const fixtures = Object.fromEntries(fs.readdirSync(FIXTURES).filter(name => name.endsWith(".json")).sort()
  .map(name => [name.replace(/\.json$/, ""), JSON.parse(fs.readFileSync(path.join(FIXTURES, name), "utf8"))]));
const chroma = fill => { const [r, g, b] = fill.slice(1).match(/../g).map(h => parseInt(h, 16)); return Math.max(r, g, b) - Math.min(r, g, b); };
const P = V.PARTIAL;

test("heat colour endpoints and grayscale", () => {
  assert.equal(V.heatColor(0, true), "#e3e7e3");
  assert.equal(V.heatColor(100, true), "#b66529");
  assert.equal(V.heatColor(0, false), "#e3e7e3");
  assert.equal(V.heatColor(100, false), "#5c6360");
  assert.equal(V.heatColor(150, true), "#b66529");
  assert.equal(V.heatColor(null, true), null);
  assert.equal(V.heatColor(undefined, false), null);
  for (let i = 0; i <= 100; i += 5) assert.ok(chroma(V.heatColor(i, false)) <= 7, "gray scale has no amber");
});

test("mm:ss formatting", () => {
  assert.equal(V.mmss(0), "0:00");
  assert.equal(V.mmss(4.5), "0:04");
  assert.equal(V.mmss(8.5, Math.ceil), "0:09");
  assert.equal(V.mmss(119.99), "1:59");
  assert.equal(V.mmss(120, Math.ceil), "2:00");
});

test("regions sort by peak index descending, stable", () => {
  const sorted = V.sortRegions([{start: 0, end: 4, peak_index: 80}, {start: 10, end: 14, peak_index: 95},
    {start: 20, end: 24, peak_index: 80}, {start: 30, end: 34, peak_index: null}]);
  assert.deepEqual(sorted.map(r => r.start), [10, 0, 20, 30]);
  assert.deepEqual(V.sortRegions(undefined), []);
});

test("labels and tooltips", () => {
  assert.equal(V.regionLabel({start: 4.5, end: 8.5, peak_index: 81}, false), "0:04–0:09 · 최고 지표 81 · ▶ 이 구간 듣기");
  assert.equal(V.regionLabel({start: 4.5, end: 8.5, peak_index: null}, false), "0:04–0:09 · 최고 지표 — · ▶ 이 구간 듣기");
  assert.equal(V.slotTitle({start: 2, end: 4, index: 33}), "2–4초 · 지표 33 (참고용, 확률 아님)");
  assert.equal(V.slotTitle({start: 118, end: 119.56, index: null}), "118–119.6초 · 지표 — (참고용, 확률 아님)");
});

test("fixture set is complete", () => {
  assert.deepEqual(Object.keys(fixtures), ["p2_defer_suspected", "p2_display_missing_suspected", "p2_extrapolated_n4_suspected",
    "p2_extrapolated_n59_not_found", "p2_low_suspected", "p2_not_applicable", "p2_not_found", "p2_suspected_one_segment",
    "p2_suspected_several", "p2_unknown_status", "p2_warn_suspected", "v1_low", "v1_warn"]);
});

// F1: amber only for slots overlapping a flagged segment; checked exactly against the gray-scale formula.
test("F1 stripModel: zero amber outside flagged regions in every fixture", () => {
  const report = {};
  for (const [name, data] of Object.entries(fixtures)) {
    const model = V.stripModel(data);
    const source = V.partialSource(data);
    const suspected = Boolean(source.partial) && source.partial.status === "suspected";
    const segments = suspected ? source.partial.segments : [];
    assert.equal(model.slots.length, data.slots.length, name);
    let violations = 0, inRegion = 0;
    model.slots.forEach((slot, i) => {
      assert.equal(slot.start, data.slots[i].start, name);
      assert.equal(slot.end, data.slots[i].end, name);
      const overlap = segments.some(s => slot.start < s.end && slot.end > s.start);
      assert.equal(slot.inRegion, overlap, `${name} slot ${i} inRegion`);
      if (slot.inRegion) inRegion += 1;
      let expected;
      if (model.mode === "degraded") expected = V.NEUTRAL;
      else expected = V.heatColor(slot.index, model.mode === "suspected" && overlap);
      if (!slot.inRegion && slot.fill !== null && slot.fill !== V.heatColor(slot.index, false) && slot.fill !== V.NEUTRAL) violations += 1;
      if (!slot.inRegion && slot.fill !== null && chroma(slot.fill) > 7) violations += 1;
      assert.equal(slot.fill, expected, `${name} slot ${i} fill`);
    });
    assert.equal(violations, 0, `${name}: amber outside flagged regions`);
    if (model.mode !== "suspected") {
      assert.ok(model.slots.every(s => s.fill === null || chroma(s.fill) <= 7), `${name}: no amber in ${model.mode} mode`);
    }
    if (suspected) {
      assert.equal(model.outlines.length, segments.length, name);
      assert.ok(inRegion > 0, `${name}: suspected fixture has in-region slots`);
    } else {
      assert.equal(model.outlines.length, 0, name);
      assert.equal(inRegion, 0, name);
    }
    report[name] = {mode: model.mode, slots: model.slots.length, inRegion, violations};
  }
  // The multi-segment fixture includes unflagged warn-band windows, so the old rule would have painted amber outside regions.
  const several = V.stripModel(fixtures.p2_suspected_several);
  assert.ok(several.slots.some(s => !s.inRegion && s.index >= 67), "fixture exercises warn-band slots outside regions");
  assert.equal(Object.values(report).reduce((sum, r) => sum + r.violations, 0), 0);
});

test("F1 old rule would have violated (regression guard for the fixture)", () => {
  const several = V.stripModel(fixtures.p2_suspected_several);
  const oldAmberOutside = several.slots.filter(s => !s.inRegion && s.index !== null && chroma(V.heatColor(s.index, true)) > 7).length;
  assert.ok(oldAmberOutside > 0);
});

const EXPECT = {
  v1_warn: {state: "absent", combined: false, bridge: false},
  v1_low: {state: "absent", combined: false, bridge: false},
  p2_suspected_one_segment: {state: "suspected", combined: false, bridge: true},
  p2_suspected_several: {state: "suspected", combined: false, bridge: true},
  p2_low_suspected: {state: "suspected", combined: false, bridge: true},
  p2_defer_suspected: {state: "suspected", combined: false, bridge: true},
  p2_warn_suspected: {state: "suspected", combined: true, bridge: false},
  p2_not_found: {state: "not_found", combined: false, bridge: false},
  p2_not_applicable: {state: "not_applicable", combined: false, bridge: false},
  p2_extrapolated_n4_suspected: {state: "suspected", combined: false, bridge: true, extrapolated: true},
  p2_extrapolated_n59_not_found: {state: "not_found", combined: false, bridge: false, extrapolated: true},
  p2_unknown_status: {state: "unknown", combined: false, bridge: false},
  p2_display_missing_suspected: {state: "suspected", combined: false, bridge: true, degraded: true},
};

test("F2/F3/F4 partialView per fixture", () => {
  for (const [name, data] of Object.entries(fixtures)) {
    const expect = EXPECT[name];
    assert.ok(expect, name);
    const source = V.partialSource(data);
    const view = V.partialView(data.outcome_code, source.partial, source.degraded);
    assert.equal(view.state, expect.state, name);
    assert.equal(view.headline, P[expect.state].headline, name);
    const body = expect.degraded ? `${P[expect.state].body} ${V.TEXT.degraded}` : P[expect.state].body;
    assert.equal(view.body, body, name);
    assert.equal(view.combined, expect.combined, `${name} combined`);
    assert.equal(view.bridge, expect.bridge, `${name} bridge`);
    assert.ok(!(view.combined && view.bridge), `${name}: never both sentences`);
    assert.equal(view.bridge, view.state === "suspected" && ["low", "defer"].includes(data.outcome_code), name);
    assert.equal(view.combined, view.state === "suspected" && data.outcome_code === "warn", name);
    assert.equal(view.extrapolated, Boolean(expect.extrapolated), `${name} extrapolated`);
    assert.equal(view.degradedNote, Boolean(expect.degraded), `${name} degraded`);
    assert.equal(source.degraded, Boolean(expect.degraded), name);
    const model = V.stripModel(data);
    if (expect.degraded) {
      // F4: positions from data.partial.segments, peaks unknown, all slots neutral gray.
      assert.equal(model.mode, "degraded");
      assert.deepEqual(model.outlines.map(o => [o.start, o.end]), data.partial.segments.map(s => [s.start, s.end]));
      assert.ok(model.outlines.every(o => o.peak_index === null));
      assert.ok(model.slots.every(s => s.fill === V.NEUTRAL));
      assert.ok(model.outlines.length > 0, "degraded body refers only to drawn regions");
    }
    if (expect.state === "unknown") assert.equal(model.mode, "plain");
  }
});

test("F3 unknown status text and absent text differ", () => {
  assert.equal(V.partialView("low", {status: "weird"}).headline, "부분 합성 검사 결과를 표시할 수 없습니다");
  assert.equal(V.partialView("low", {status: "weird"}).body, "위의 전체 음성 검사 결과를 참고하세요.");
  assert.equal(V.partialView("low", {status: "absent"}).state, "unknown");
  assert.equal(V.partialView("low", null).headline, "현재 구성에서는 부분 합성 검사가 제공되지 않습니다");
  assert.equal(V.partialView("low", null).body, "");
});

test("F4 degraded note only when regions are drawn", () => {
  assert.equal(V.partialView("low", {status: "suspected", segments: []}, true).degradedNote, false);
  assert.equal(V.partialView("low", {status: "not_found", segments: []}, true).degradedNote, false);
  assert.equal(V.partialView("low", {status: "suspected", segments: [{start: 0, end: 4}]}, false).degradedNote, false);
});

test("F6 shouldRender", () => {
  const a = {name: "a.wav"}, b = {name: "a.wav"};
  assert.equal(V.shouldRender(3, 3, a, a), true);
  assert.equal(V.shouldRender(2, 3, a, a), false, "a newer request exists");
  assert.equal(V.shouldRender(3, 3, a, b), false, "file object changed even with the same name");
  assert.equal(V.shouldRender(3, 3, a, undefined), false);
  assert.equal(V.TEXT.stale, "파일이 바뀌어 이전 분석 결과를 표시하지 않았습니다. 다시 분석하세요.");
});
