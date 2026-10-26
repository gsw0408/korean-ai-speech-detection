"use strict";
// Pure Option C presentation helpers (no DOM). Loaded before app.js; also testable with `node --test`.
(function (root) {
  const NEUTRAL = [0xe3, 0xe7, 0xe3], AMBER = [0xb6, 0x65, 0x29], DARK_GRAY = [0x5c, 0x63, 0x60];
  const WIDTH = 800;
  const TEXT = {
    wholeTitle: "전체 음성 합성 검사",
    indexLabel: "AI 합성 흔적 지표",
    indexNote: "확률이 아닌 상대적 지표입니다. 높을수록 AI 합성 흔적이 강합니다.",
    wholeSub: "파일 전체 구간 점수의 평균으로 판단합니다. 일부 구간만 합성된 경우에는 이 결과가 낮게 나올 수 있습니다.",
    partialTitle: "부분 합성 검사 (실험적)",
    extrapolated: "이 길이의 파일은 검사 기준을 보정한 범위 밖이어서 오경보 수준을 보장하지 않습니다.",
    combined: "전체 음성 검사에서 이미 AI 합성 의심이 나왔습니다. 표시된 구간은 점수가 특히 높은 부분입니다.",
    bridge: "전체 평균 점수는 낮지만, 일부 구간 점수가 이 파일 길이에 적용된 검사 기준보다 높습니다.",
    degraded: "구간 지표를 계산하지 못해 위치만 표시합니다.",
    stripContext: "실제 음성에서도 일부 구간의 지표가 높게 나오는 경우가 흔합니다. 이 그래프만으로 합성 여부를 판단하지 마세요.",
    grayLegend: "진할수록 지표가 높습니다 (확률 아님)",
    regionLegend: "색이 있는 칸: 검토가 필요한 구간",
    stale: "파일이 바뀌어 이전 분석 결과를 표시하지 않았습니다. 다시 분석하세요.",
    listen: "▶ 이 구간 듣기",
    stop: "■ 멈추기",
    more: "더 보기"
  };
  const PARTIAL = {
    suspected: {headline: "검토가 필요한 구간이 있습니다", body: "아래 표시된 구간은 4초 분석 창 단위의 대략적인 위치입니다. 실제 음성에서도 이런 구간이 표시될 수 있으니, 직접 들어 보고 다른 방법으로도 확인하세요."},
    not_found: {headline: "부분 합성 근거를 찾지 못했습니다", body: "1–2초 정도의 짧은 삽입이나 음성 변환으로 바꾼 구간은 대부분 탐지하지 못합니다. 실제 음성이라는 뜻은 아닙니다."},
    not_applicable: {headline: "이 파일에는 부분 합성 검사를 적용하지 않았습니다", body: "분석 창 수가 검사 지원 범위(4–60개)를 벗어납니다. 위의 전체 음성 검사 결과를 참고하세요."},
    unknown: {headline: "부분 합성 검사 결과를 표시할 수 없습니다", body: "위의 전체 음성 검사 결과를 참고하세요."},
    absent: {headline: "현재 구성에서는 부분 합성 검사가 제공되지 않습니다", body: ""}
  };
  const KNOWN = ["suspected", "not_found", "not_applicable"];

  function hex(rgb) { return "#" + rgb.map(v => Math.round(v).toString(16).padStart(2, "0")).join(""); }
  function mix(a, b, t) { return a.map((v, i) => v + (b[i] - v) * t); }
  function known(index) { return index !== null && index !== undefined && Number.isFinite(index); }
  // Heat colour for a 0-100 index: neutral gray -> amber when highlighted, neutral gray -> dark gray otherwise.
  function heatColor(index, highlighted) {
    if (!known(index)) return null;
    const t = Math.min(100, Math.max(0, index)) / 100;
    return hex(mix(NEUTRAL, highlighted ? AMBER : DARK_GRAY, t));
  }
  function mmss(seconds, round) {
    const s = Math.max(0, round === Math.ceil ? Math.ceil(seconds - 1e-9) : Math.floor(seconds + 1e-9));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  function seconds(value) { return String(Number(Number(value).toFixed(1))); }
  function slotTitle(slot) {
    const shown = known(slot.index) ? slot.index : "—";
    return `${seconds(slot.start)}–${seconds(slot.end)}초 · 지표 ${shown} (참고용, 확률 아님)`;
  }
  function sortRegions(segments) {
    return (segments || []).map((s, order) => ({...s, order}))
      .sort((a, b) => (b.peak_index ?? -1) - (a.peak_index ?? -1) || a.order - b.order);
  }
  function regionLabel(region, playing) {
    const peak = known(region.peak_index) ? region.peak_index : "—";
    return `${mmss(region.start)}–${mmss(region.end, Math.ceil)} · 최고 지표 ${peak} · ${playing ? TEXT.stop : TEXT.listen}`;
  }
  function overlaps(slot, segment) { return slot.start < segment.end && slot.end > segment.start; }

  // Which partial result the page shows: display.partial when present; otherwise the raw partial
  // (display missing: indexes unknown, positions only).
  function partialSource(data) {
    if (data.display && data.display.partial) return {partial: data.display.partial, degraded: false};
    if (data.partial) {
      return {partial: {status: data.partial.status, calibration_range: null, threshold_index: null,
                        segments: (data.partial.segments || []).map(s => ({start: s.start, end: s.end, peak_index: null}))},
              degraded: !data.display};
    }
    return {partial: null, degraded: false};
  }

  // State -> partial card text and flags. `degraded` = display missing (positions only).
  function partialView(outcomeCode, partial, degraded) {
    const off = {extrapolated: false, combined: false, bridge: false, degradedNote: false};
    if (!partial) return {state: "absent", ...PARTIAL.absent, ...off};
    const state = KNOWN.includes(partial.status) ? partial.status : "unknown";
    const suspected = state === "suspected";
    const degradedNote = Boolean(degraded) && suspected && (partial.segments || []).length > 0;
    return {state, headline: PARTIAL[state].headline,
            body: degradedNote ? `${PARTIAL[state].body} ${TEXT.degraded}` : PARTIAL[state].body,
            extrapolated: (suspected || state === "not_found") && partial.calibration_range === "extrapolated",
            combined: suspected && outcomeCode === "warn",
            bridge: suspected && (outcomeCode === "low" || outcomeCode === "defer"),
            degradedNote};
  }

  // Strip model (viewBox width 800). Amber only for slots overlapping a flagged segment when the verdict is
  // `suspected`; every other slot is on the gray scale. Unknown indexes have fill null (hatched by the page),
  // except in the degraded path where all slots use the neutral gray because no index exists.
  function stripModel(data) {
    const duration = data.duration || 0;
    const {partial, degraded} = partialSource(data);
    const suspected = Boolean(partial) && partial.status === "suspected";
    const segments = suspected ? (partial.segments || []) : [];
    const slots = (data.display && data.display.slots) ||
      (data.slots || []).map(slot => ({start: slot.start, end: slot.end, index: null}));
    const scale = value => (duration ? WIDTH * value / duration : 0);
    const model = {mode: suspected ? (degraded ? "degraded" : "suspected") : "plain", slots: [], outlines: []};
    for (const slot of slots) {
      const inRegion = segments.some(segment => overlaps(slot, segment));
      let fill = heatColor(slot.index, inRegion && !degraded);
      if (degraded) fill = hex(NEUTRAL);
      model.slots.push({start: slot.start, end: slot.end, index: slot.index, inRegion,
                        x: scale(slot.start), width: Math.max(1, scale(slot.end - slot.start) - 1), fill, title: slotTitle(slot)});
    }
    for (const segment of segments) {
      model.outlines.push({start: segment.start, end: segment.end, peak_index: segment.peak_index ?? null,
                           x: scale(segment.start) + 1, width: Math.max(2, scale(segment.end - segment.start) - 2),
                           title: `${mmss(segment.start)}–${mmss(segment.end, Math.ceil)} · 최고 지표 ${known(segment.peak_index) ? segment.peak_index : "—"} (참고용)`});
    }
    return model;
  }

  // Stale-result guard: render only the latest request, and only while the same file is selected.
  function shouldRender(seq, latestSeq, sentFile, currentFile) {
    return seq === latestSeq && sentFile === currentFile;
  }

  const api = {TEXT, PARTIAL, NEUTRAL: hex(NEUTRAL), AMBER: hex(AMBER), DARK_GRAY: hex(DARK_GRAY), heatColor, mmss, slotTitle,
               sortRegions, regionLabel, partialSource, partialView, stripModel, shouldRender, VISIBLE_REGIONS: 3};
  root.OptionCView = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
