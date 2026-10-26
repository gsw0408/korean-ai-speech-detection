"""Measured performance shown to users, copied from the project's evaluation reports (no new evaluation here).

Every figure names its data set and caveat. Unfavourable results are listed alongside favourable ones.
Scores are never presented as probabilities: no calibrated probability exists because the share of
synthetic audio among real-world uploads is unknown.
"""

PERFORMANCE_VERSION = "performance-20261023"

WHOLE_FILE = {
    "title": "전체 음성 합성 검사 (NII-Small, 재학습 없음)",
    "rows": [
        {"label": "실제 음성을 잘못 경고한 비율", "value": "4.70%", "scope": "DEV 1,998개, 구성요소 분리 2-fold (개발 평가)"},
        {"label": "AI 음성을 경고한 비율", "value": "90.59%", "scope": "DEV 1,998개, 구성요소 분리 2-fold (개발 평가)"},
        {"label": "최종 시험셋 AUC", "value": "0.982 [0.969–0.992]", "scope": "FINAL v2 822개, 2026-10-09 단일 평가"},
        {"label": "최종 시험셋 탐지율 / 실제 음성 오경보", "value": "96.3% / 8.8%",
         "scope": "FINAL v2, CALIB에서 고정한 임계값 적용(사후 분석). 이 서비스의 임계값과는 다름"},
    ],
    "limits": [
        "최종 시험셋 결과는 이미 공개·확인된 데이터입니다. 서비스 모델 선택(10/10)이 이 결과를 본 뒤 이루어져 완전히 독립적인 추정이 아닙니다.",
        "평가 데이터는 한국어 낭독체와 합성 음성 중심입니다. 실제 통화·잡음 환경, 처음 보는 생성기에서는 성능이 낮을 수 있습니다.",
        "스마트폰 녹음 실제 음성에서 오경보가 더 높게 관찰되었습니다.",
    ],
}

PARTIAL = {
    "title": "부분 합성 검사 (실험적, partial2 W0)",
    "rows": [
        {"label": "실제 음성에 구간을 잘못 표시한 비율", "value": "9.85% [4.16–16.44]", "scope": "DEV-B 합성 composite, 실제 132개"},
        {"label": "삽입 구간 검출률 1초 / 2초 / 3초 / 전체", "value": "5.5% / 26.6% / 45.3% / 77.0%", "scope": "DEV-B 합성 composite"},
        {"label": "2초 삽입에서 이전 방식 대비 개선", "value": "입증 실패 (+0.78%p, CI −2.55~4.39)", "scope": "DEV-B 사전 등록 기준 G4 미통과"},
    ],
    "limits": [
        "파일이 길수록 실제 음성에 구간을 잘못 표시하는 비율이 높아졌습니다 (TRAIN 교차검증 90–120초 파일 26.0%).",
        "최종 시험셋(FINAL)으로는 평가하지 않았습니다. 실제 편집된 음성으로 검증하지 않았습니다.",
        "표시 구간은 4초 분석 창 단위의 대략적인 위치이며, 음성 변환으로 바꾼 구간은 대부분 찾지 못합니다.",
    ],
}

SCORES_NOTE = ("0–100 지표는 점수를 판정 기준에 맞춰 옮긴 상대적 지표이며 확률이 아닙니다. "
               "실제 이용 환경에서 AI 음성이 차지하는 비율을 알 수 없어 보정된 확률을 제공하지 않습니다.")
SOURCES = ["webservice/VERIFICATION_REPORT.md", "claude/artifacts/kaggle_execution_e2_20261008/reports/DATASET_V2_E2_FINAL_V2_REPORT_KO_READABLE_20261008.md",
           "claude/artifacts/posthoc_baseline_thresholds_20261009/POSTHOC_NII_SSL_CALIB_THRESHOLD_REPORT_KO.md",
           "webservice/work/nr/NR_REPORT.md", "webservice/work/bc/REPORT.md"]


def performance(partial_enabled):
    """The disclosure for the loaded configuration; the partial section appears only when that check is active."""
    return {"version": PERFORMANCE_VERSION, "whole_file": WHOLE_FILE, "partial": PARTIAL if partial_enabled else None,
            "scores_note": SCORES_NOTE, "sources": SOURCES}
