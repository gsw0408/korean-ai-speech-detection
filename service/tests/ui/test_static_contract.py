import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE_FILES = ("static/index.html", "static/app.js", "static/view.js")


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def page_by_id():
    return {attrs["id"]: (tag, attrs) for tag, attrs in Elements(read("static/index.html")).elements if "id" in attrs}


def test_consent_is_separate_unchecked_and_page_is_self_contained():
    elements = Elements((ROOT / "static/index.html").read_text(encoding="utf-8")).elements
    by_id = {attrs["id"]: attrs for tag, attrs in elements if "id" in attrs}
    assert "checked" not in by_id["analysis-consent"]
    assert "checked" not in by_id["donation-opt-in"]
    assert "required" in by_id["analysis-consent"]
    assert "required" not in by_id["donation-opt-in"]
    for tag, attrs in elements:
        for key in ("src", "href", "action"):
            if key in attrs:
                assert attrs[key].startswith("/")
    assert any(tag == "audio" and "controls" in attrs for tag, attrs in elements)
    assert any(tag == "svg" and attrs.get("id") == "timeline" for tag, attrs in elements)


def test_scripts_are_same_origin_and_view_helpers_load_first():
    scripts = [attrs["src"] for tag, attrs in Elements(read("static/index.html")).elements if tag == "script"]
    assert scripts == ["/static/view.js", "/static/app.js"]
    for relative in PAGE_FILES + ("static/style.css",):
        # The SVG namespace identifier is not a network request; anything else URL-like is.
        source = read(relative).replace("http://www.w3.org/2000/svg", "")
        assert not re.search(r"https?://|//[\w-]+\.[\w.-]+/", source), relative
        assert "@import" not in source


def test_option_c_structure_and_accessibility():
    by_id = page_by_id()
    for element_id in ("result", "error-view", "outcome", "reason", "verdicts", "whole-card", "whole-outcome", "file-index",
                       "file-meter", "partial-card", "partial-headline", "partial-body", "partial-combined", "partial-extrapolated",
                       "partial-info", "partial-scope", "partial-eval", "help", "playback", "player", "strip-open", "strip-details",
                       "strip-closed", "timeline", "slot-info", "scale", "index-scale", "regions", "region-list", "regions-more",
                       "region-status", "notice", "donation-action", "withdrawal-result", "repro",
                       "partial-bridge", "legend-open", "legend-closed", "gray-scale-open", "gray-scale-closed", "region-swatch", "legend-region-item"):
        assert element_id in by_id, element_id
    assert "partial-timeline" not in by_id
    assert "partial-timeline" not in read("static/app.js") and "partial-timeline" not in read("static/style.css")
    tag, timeline = by_id["timeline"]
    assert tag == "svg" and timeline["role"] == "img" and timeline["tabindex"] == "0"
    assert timeline["aria-label"] == "구간별 AI 합성 흔적 지표 (참고용)"
    assert by_id["region-status"][1]["aria-live"] == "polite"
    assert by_id["slot-info"][1]["aria-live"] == "polite"
    assert by_id["result"][1]["aria-live"] == "polite"
    assert by_id["strip-details"][0] == "details" and by_id["help"][0] == "details" and by_id["partial-info"][0] == "details"
    assert by_id["region-list"][0] == "ul"
    assert by_id["regions-more"][0] == "button" and by_id["regions-more"][1]["type"] == "button"
    app = read("static/app.js")
    assert 'document.createElement("button")' in app and 'button.type="button"' in app
    assert 'setAttribute("aria-pressed"' in app
    # Fix 1: new elements start hidden where they are state-dependent, and the strip renders from the pure model.
    assert "hidden" in by_id["partial-bridge"][1] and "hidden" in by_id["legend-open"][1]
    assert "hidden" not in by_id["legend-closed"][1]
    closed = read("static/index.html").split('id="strip-details"')[1].split("</details>")[0]
    assert 'id="legend-closed"' in closed and 'id="timeline"' in closed
    # The colored-region swatch is shown only when amber slots are actually drawn (not in the degraded path).
    assert '$("legend-region-item").hidden=model.mode!=="suspected"' in app
    assert "V.stripModel(data)" in app and "V.shouldRender(seq,requestSeq,sentFile,selectedFile)" in app


def test_partial_not_applicable_text_covers_all_unsupported_window_counts():
    source = read("static/app.js") + read("static/view.js")
    assert "지원 범위(4–60개)를 벗어납니다" in source
    assert "파일이 짧아" not in source


REQUIRED_TEXT = [
    "전체 음성 합성 검사", "AI 합성 흔적 지표", "확률이 아닌 상대적 지표입니다. 높을수록 AI 합성 흔적이 강합니다.",
    "파일 전체 구간 점수의 평균으로 판단합니다. 일부 구간만 합성된 경우에는 이 결과가 낮게 나올 수 있습니다.",
    "부분 합성 검사 (실험적)", "이 길이의 파일은 검사 기준을 보정한 범위 밖이어서 오경보 수준을 보장하지 않습니다.", "검사 정보",
    "재현 정보", "검토가 필요한 구간이 있습니다",
    "아래 표시된 구간은 4초 분석 창 단위의 대략적인 위치입니다. 실제 음성에서도 이런 구간이 표시될 수 있으니, 직접 들어 보고 다른 방법으로도 확인하세요.",
    "부분 합성 근거를 찾지 못했습니다",
    "1–2초 정도의 짧은 삽입이나 음성 변환으로 바꾼 구간은 대부분 탐지하지 못합니다. 실제 음성이라는 뜻은 아닙니다.",
    "이 파일에는 부분 합성 검사를 적용하지 않았습니다",
    "분석 창 수가 검사 지원 범위(4–60개)를 벗어납니다. 위의 전체 음성 검사 결과를 참고하세요.",
    "현재 구성에서는 부분 합성 검사가 제공되지 않습니다",
    "전체 음성 검사에서 이미 AI 합성 의심이 나왔습니다. 표시된 구간은 점수가 특히 높은 부분입니다.",
    "구간별 지표 보기", "실제 음성에서도 일부 구간의 지표가 높게 나오는 경우가 흔합니다. 이 그래프만으로 합성 여부를 판단하지 마세요.",
    "최고 지표", "▶ 이 구간 듣기", "더 보기", "(참고용, 확률 아님)", "구간별 AI 합성 흔적 지표 (참고용)", "어떻게 판단하나요?",
    "전체 평균 점수는 낮지만, 일부 구간 점수가 이 파일 길이에 적용된 검사 기준보다 높습니다.",
    "부분 합성 검사 결과를 표시할 수 없습니다", "위의 전체 음성 검사 결과를 참고하세요.",
    "구간 지표를 계산하지 못해 위치만 표시합니다.", "진할수록 지표가 높습니다 (확률 아님)", "색이 있는 칸: 검토가 필요한 구간",
    "파일이 바뀌어 이전 분석 결과를 표시하지 않았습니다. 다시 분석하세요.",
    "이 서비스는 두 가지를 따로 확인합니다. ① 전체 음성 합성 검사는 파일 전체의 평균 점수로, 목소리 전체가 AI로 만들어진 경우를 찾습니다. ② 부분 합성 검사는 가장 높은 4초 구간 점수를 파일 길이에 따라 높아지는 기준과 비교해, 일부만 합성된 경우를 찾습니다. 개발 평가(합성 데이터, 독립 검증 아님)에서 부분 합성 검사는 실제 음성 약 10개 중 1개에도 구간을 표시했고, 2초 삽입은 약 4개 중 1개만 찾았습니다. 결과는 참고 자료이며 화자나 진위를 확인하지 않습니다.",
]


def test_required_korean_text_is_verbatim():
    source = "".join(read(relative) for relative in PAGE_FILES)
    for text in REQUIRED_TEXT:
        assert text in source, text


# Only explicit negations of "확률" are allowed: the existing server notice and the prescribed Option C wording.
# Fix 1 adds the prescribed gray-legend sentence, which is itself a negation ("확률 아님").
ALLOWED_NEGATIONS = ("점수는 AI 확률이 아닙니다", "확률이 아닌 상대적 지표입니다", "(참고용, 확률 아님)", "진할수록 지표가 높습니다 (확률 아님)")
BANNED = [r"확률", r"가능성\s*[0-9]", r"신뢰도", r"실제 음성입니다", r"안전", r"조작 확인", r"조작됨", r"AI 구간", r"AI 의심"]


def test_banned_wording_scan():
    for relative in PAGE_FILES:
        source = read(relative)
        for allowed in ALLOWED_NEGATIONS:
            source = source.replace(allowed, "")
        for pattern in BANNED:
            assert not re.search(pattern, source), (relative, pattern)


def test_banned_wording_scan_detects_violations():
    for sample in ("AI 확률 80%", "가능성 70%", "신뢰도", "실제 음성입니다", "안전합니다", "조작됨", "AI 구간"):
        assert any(re.search(pattern, sample) for pattern in BANNED), sample


def test_new_strip_colours_are_neutral_or_amber_only():
    allowed = {"e3e7e3", "b66529", "5c6360", "203431", "eef0ee", "89938c"}
    for relative in ("static/view.js", "static/app.js"):
        source = read(relative)
        colours = {match.lower() for match in re.findall(r"#([0-9a-fA-F]{6})\b", source)}
        colours |= {"".join(match).lower() for match in re.findall(r"0x([0-9a-fA-F]{2}), ?0x([0-9a-fA-F]{2}), ?0x([0-9a-fA-F]{2})", source)}
        assert colours <= allowed, (relative, colours)


def test_raw_scores_are_only_in_reproduction_details():
    app = read("static/app.js")
    assert "toFixed(4)" not in app
    assert app.count("data.file_score") == 1 and "file_score:data.file_score" in app
    assert "statistic:data.partial.statistic" in app and "threshold:data.partial.threshold" in app
    assert "display_version:data.display?.version" in app


def test_donation_option_hidden_until_enabled_and_policy_text_present():
    source = (ROOT / "static/index.html").read_text(encoding="utf-8")
    by_id = {attrs["id"]: attrs for tag, attrs in Elements(source).elements if "id" in attrs}
    assert "hidden" in by_id["donation-details"]
    assert "hidden" not in by_id.get("withdraw-form", {})
    for phrase in ("동의 기록은 30일이 지나면 자동 삭제", "프로젝트 종료 시 모두 삭제", "확정하지 않으면 사본은 저장되지 않고 사라집니다",
                   "철회하면 동의 기록이 즉시 삭제됩니다", "동의 기록만 남고 사본은 바로 삭제", "코드를 잃어버리면"):
        assert phrase in source
    for removed in ("영구 삭제", "메타데이터 포함 원본", "연구 검토용", "30일까지 보관", "격리된 음성", "기부 음성", "저장된 음성"):
        assert removed not in source, removed
    # The age/other-speakers declaration sits in the confirmation box, starts unticked and is not pre-required by the form.
    action = source.split('id="donation-action"')[1].split("</div>")[0]
    assert 'id="donation-eligibility"' in action and "만 14세 이상" in action and "모든 사람에게서" in action
    assert "checked" not in by_id["donation-eligibility"] and "required" not in by_id["donation-eligibility"]
    assert source.count('href="/privacy"') == 3
    script = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert '$("donation-details").hidden=!data.donations_enabled' in script
    assert 'Boolean(config.donations_enabled)&&$("donation-opt-in").checked' in script
    assert "eligibility_confirmed:true" in script and 'if(!$("donation-eligibility").checked)' in script
    assert '$("donation-eligibility").checked=false' in script
    for reason in ("consent_version_changed", "donation_eligibility_required", "service_paused", "rate_limited"):
        assert reason + ":" in script


def test_demonstration_notice_sits_directly_below_consent_text():
    source = read("static/index.html")
    sentence = "본 프로젝트에서는 음성 데이터 제공 및 동의 철회 절차를 시연하며, 제공된 음성 파일은 연구용으로 별도 보관하지 않습니다."
    details = source.split('id="donation-details"')[1].split("</details>")[0]
    assert details.count(sentence) == 1
    assert '<span id="consent-text">' in details.split("</label>", 1)[0]
    after_label = details.split("</label>", 1)[1]
    assert after_label.startswith('<p id="demo-notice" class="small">' + sentence + "</p>")
    assert 'data.status==="consent_recorded"' in read("static/app.js")


def test_beta_feedback_link_hidden_until_configured_and_withdrawal_text_matches_no_audio_policy():
    by_id = page_by_id()
    assert "hidden" in by_id["feedback-item"][1] and "href" not in by_id["feedback-link"][1]
    assert by_id["feedback-link"][1]["rel"] == "noopener noreferrer"
    script = read("static/app.js")
    assert 'test(data.feedback_url)' in script
    assert "격리 음성" not in script and '"동의 기록을 삭제했습니다."' in script


def test_research_wording_removed_and_stale_result_is_cleared_on_new_file():
    page, script = read("static/index.html"), read("static/app.js")
    for removed in ("연구에 음성 기부", "연구용 기부", "기부 저장 공간", "이미 철회된 기부"):
        assert removed not in page + script, removed
    # The "격리 저장" heading remains only as the JS branch for DONATION_STORE_AUDIO=true deployments.
    assert "격리 저장되었습니다" not in page
    assert "선택 사항: 음성 데이터 제공 (시연)" in page and ">제공 확정</button>" in page
    by_id = page_by_id()
    assert by_id["repro-details"][0] == "details" and "hidden" in by_id["help-partial-off"][1]
    assert "function clearStaleResult()" in script and "resetView(); clearStaleResult();" in script
    assert '$("help-partial-off").hidden=Boolean(data.partial_enabled)' in script
    # A withdrawal code that is still on screen must not be hidden by choosing a new file.
    assert 'if($("withdrawal-result").hidden){$("result").hidden=true;}' in script


def test_legal_notices_and_performance_panel_present():
    page, script = read("static/index.html"), read("static/app.js")
    by_id = page_by_id()
    assert by_id["performance"][0] == "details" and "performance-body" in by_id
    assert "renderPerformance(data.performance)" in script and "textContent=text" in script
    for phrase in ("통신비밀보호법", "수집 항목: 업로드한 음성 파일", "보유 기간: 분석이 끝나면 바로 삭제",
                   "동의하지 않을 수 있으며, 이 경우 분석을 이용할 수 없습니다", "만 14세 미만은 이용할 수 없습니다"):
        assert phrase in page, phrase
    upload = page.split('id="upload-form"')[1].split("</form>")[0]
    assert upload.index('id="analysis-consent"') < upload.index('id="analysis-notice"')
