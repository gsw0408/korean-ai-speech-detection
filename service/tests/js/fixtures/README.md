# Option C UI fixtures

These are synthetic `/api/analyze` responses used by `tests/js/view.test.js`. They contain no audio, no DEV, CALIB or FINAL data, and no user data.

## How they were generated

Run from `webservice/`:

```
.venv\Scripts\python.exe -B tests\js\fixtures\generate_fixtures.py
```

The script uses `tests/api/optionc_harness.py`:
- It runs the real FastAPI app and the real `pipeline.service._worker` loop.
- Only `prepare()` and the detector are replaced, so fixed synthetic window scores flow through the unchanged `decide()`, partial and `build_display()` code.
- Scores are fixed numbers in the script. They were never derived from thresholds.
- `request_id` is set to `"fixture"`. Every other field is exactly what the service returned.

| Fixture | Bundle | Synthetic scores (window index: score) | Result |
|---|---|---|---|
| `v1_warn` | v1 | 26 s, all 12 windows 1.5 | warn, no partial |
| `v1_low` | v1 | 26 s, base −4.0, {5: 3.0} | low, no partial |
| `p2_suspected_one_segment` | partial2 | 26 s, base −2.6, {6: 3.0} | low, suspected, 1 segment |
| `p2_suspected_several` | partial2 | 60 s, base −4.0, {3: 3.5, 10: 3.0, 11: 3.1, 15: 1.0, 16: 1.2, 20: 4.2, 25: 2.6} | low, suspected, 4 segments; windows 15–16 are warn-band but below the cut, so they are not flagged |
| `p2_low_suspected` | partial2 | 26 s, base −4.0, {5: 3.0} | low, suspected |
| `p2_defer_suspected` | partial2 | 26 s, base −1.6, {2: 0.4, 8: 2.5} | defer, suspected; window 2 is warn-band and not flagged |
| `p2_warn_suspected` | partial2 | 26 s, base 2.0, {7: 4.0} | warn, suspected |
| `p2_not_found` | partial2 | 26 s, base −3.0, {5: 1.5} | low, not_found |
| `p2_not_applicable` | partial2 | 8 s, [4.5, 4.0, 5.0] (3 windows) | warn, not_applicable |
| `p2_extrapolated_n4_suspected` | partial2 | 10 s, [−3.5, −3.5, −3.5, 5.0] | defer, suspected, extrapolated (n = 4) |
| `p2_extrapolated_n59_not_found` | partial2 | 120 s, base −3.0, {30: 2.5} | low, not_found, extrapolated (n = 59) |
| `p2_unknown_status` | hand-edited | copy of `p2_not_found` with `partial.status` and `display.partial.status` set to `"inconclusive"` | unknown status |
| `p2_display_missing_suspected` | hand-edited | copy of `p2_suspected_several` with `display` removed | degraded path |

Only the last two fixtures were edited by hand. Everything in the others came from the service.
