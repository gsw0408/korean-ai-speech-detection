# Korean AI-speech detection research service

A local, Dockerized FastAPI service using unchanged NII-Small, a static Korean page, and SQLite donation quarantine. It accepts WAV/MP3/FLAC files of 4–120 seconds, up to 25 MiB, with at least four seconds of active frames. Results are research demonstrations, not validated speaker identity, safety, or AI probability claims.

## Run locally

Docker Engine and Compose v2 are required. From this folder:

```sh
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). For a background container, use `docker compose up -d --build --wait`. Health is `/api/health`. Stop with `docker compose down`; this preserves the donation volume. Do not remove volumes containing donations unless deletion is intended.

On this Windows host, Docker Engine is in the existing WSL2 Ubuntu distribution:

```powershell
wsl -d Ubuntu-24.04 -u root -- sh -lc 'cd /mnt/d/AI/학교자료/26년_2학기/오픈소스SW_프로젝트/webservice && docker compose up --build'
```

Keep the foreground command or a WSL session open while using the service; systemd services alone may not keep WSL running. The service is bound to loopback. No AWS resources or public deployment are needed.

## What the service does

ffmpeg decodes native-rate, all-channel PCM16. The original `canonical`, edge trim, normalization, and 64,000-sample / 32,000-hop windows are retained verbatim. Service validation always runs before inference. NII scores batches of eight using negative class-1 logits; the file mean determines the result. Two-second timeline slots use the maximum overlapping window score on the original file clock and are labeled indicative only.

The four outcomes are `AI 음성 의심 — 추가 확인 필요`, `AI 음성 증거가 낮음`, `판단 유보`, and `분석 불가`. Size, unsupported format, decode, duration, trimmed length, active speech, timeout, and capacity failures have explicit reasons. A worker process is terminated and restarted on inference timeout, including decoder children and temporary files. Four ONNX threads and one analysis worker were selected from CPU measurements. At most eight upload requests are admitted to bound queued memory.

The page supports a file picker and drag/drop, upload progress, local object-URL audio playback, keyboard/click timeline seeking, result scores, calibration notices, optional donation confirmation, and withdrawal codes. It has no external CDN, frontend framework, or build step.

## API and consent

- `GET /api/health`: readiness, bundle ID and hashes.
- `GET /api/config`: limits, donation consent version/text and development notices.
- `POST /api/analyze`: multipart `file`, `analysis_consent=true`, optional `donation_opt_in=true` (default false).
- `POST /api/donations`: JSON `token`, `consent_version`, `confirm=true`; persists quarantine and returns a withdrawal code.
- `DELETE /api/donations/{code}`: deletes the quarantined file and SQLite consent record.
- `DELETE /api/donation-tokens/{token}`: discards an unconfirmed in-memory copy.

Analysis consent is separate from donation consent. Without pre-upload donation opt-in, no donation copy or token exists. With opt-in, a separate bounded in-memory copy expires after 15 minutes and is stored only after explicit donation confirmation. Donations are never automatically consumed by training. Withdrawal codes are random; only their hashes are stored. Keep a code to withdraw later.

Analysis spools and decoder files are on `/tmp/uploads` tmpfs and are deleted on success, errors, and timeout. Logs contain only request ID, duration, outcome, latency, and bundle ID. Access logs are disabled. Original names, audio, transcripts, withdrawal codes, and score/user linkage are not logged. The named donation volume is private local storage; backups require a separate deletion/retention policy.

## Bundles and rollback

`ACTIVE_BUNDLE=bundles/nii-small-v1` selects a manifest containing detector, preprocessing, and calibration file hashes, the input contract, and score direction. Every response returns the selected bundle identity and hashes. Startup refuses hash mismatch, incompatible input contracts, invalid thresholds, or calibration binding mismatch. Preprocessing modules are loaded from the selected, verified manifest; the initial copies remain unchanged.

A replacement needs a compatible ONNX detector, verified preprocessing modules, and thresholds fitted for their hashes. Package a new version, retain the prior version, change `ACTIVE_BUNDLE`, and recreate the service. For local read-only external bundles/models, add approved volume mounts using the production Compose pattern. Rollback resets `ACTIVE_BUNDLE` and/or the image tag. Configuration mechanism tests do not replace the roadmap's independent model-validation gate; no candidate has been promoted automatically.

## Calibration and parity

Only the owner-approved TRAIN_DEV archive was used. There are 1,998 DEV files across five cells and both labels. Model weights and original research artifacts were not changed. No protected CALIB/FINAL data or results were accessed.

The thresholds are development calibration: `t_low=-2.132084210713705`, `t_warn=-0.5011004805564879`. The fitter uses float64 boundary neighbors to implement inclusive comparisons and tied scores. It fits the smallest warning threshold with REAL score-level false warnings ≤5%, and the largest low threshold with AI score-level false reassurance ≤1%, below the warning threshold. Results report both in-sample and component-group two-fold outcomes, retaining rejected inputs in every denominator. See `calibration/DEVELOPMENT_REPORT.json`.

Research score reproducibility and service rejection are independent checks. The earlier four rejected examples reproduce their research scores, while the service still rejects them. Across the DEV set, 158 files fail the required service validity check. The API never returns those research-only scores. Scoring parity does not establish service accuracy.

In-sample service rates: REAL warnings 4.30%; AI low-evidence errors 0.80%; AI warnings 90.39%. Component-held-out DEV rates: REAL warnings 4.70%; AI low-evidence errors 1.10%; AI warnings 90.59%. These do not pass the independent service promotion criteria. Results remain labeled a research demonstration.

## Reproduce checks

`work/` is local, DEV-only verification data and is excluded from the image and release ZIP. Initial source audit/extraction scripts use the exact owner-approved Windows paths and archive identities. Do not substitute protected datasets or extract archives wholesale. Source-container exceptions do not grant access to protected members.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/audit_inputs.py
.\.venv\Scripts\python.exe scripts/prepare_dev.py
.\.venv\Scripts\python.exe calibration/fit_nii_v1.py
```

With approved DEV fixtures already in `work/`, use these Linux/WSL commands from the service folder:

```sh
docker run --rm --cpus=4 --memory=2500m --tmpfs /tmp/uploads:uid=10001,gid=10001,mode=0700 -v "$PWD/work:/service/work" nii-speech:local python -m pytest tests -q --basetemp=/tmp/pytest-suite -o cache_dir=/tmp/pytest-cache --junitxml=work/container-tests.xml
docker run --rm --cpus=4 --memory=2500m --tmpfs /tmp/uploads:uid=10001,gid=10001,mode=0700 -e PARITY_OUTPUT=/service/work/PARITY_REPORT_CONTAINER.json -v "$PWD/work:/service/work" nii-speech:local python scripts/parity.py --full
docker run --rm --network webservice_default -e BASE_URL=http://service:8000 -v "$PWD/work:/service/work" nii-speech:local sh scripts/e2e_local.sh
docker run --rm --cpus=4 --memory=4g --tmpfs /tmp/uploads:uid=10001,gid=10001,mode=0700 -v "$PWD/work:/service/work" nii-speech:local python scripts/benchmark.py
sh scripts/verify_docker.sh
```

Tests execute as UID 10001. Executable deployment stubs use a separate container test directory; uploads retain tmpfs/noexec protections. The full suite intentionally needs local DEV fixtures for real parity regression. The image also excludes per-file development reports and the full host/source audit; the release ZIP excludes all `work/` data. A source-only recipient can run `tests/unit tests/api tests/ui tests/privacy tests/docker` once the model is present; full `tests/parity` requires approved DEV evidence. Do not obtain that evidence from forbidden source paths.

Benchmark output is written atomically to `work/BENCHMARK.json`; copy it to the root report after completion. Numeric checkpoints for full parity are in the chosen output path. Do not treat an interrupted `status=running` report as completed evidence.

## Results and scope

See `VERIFICATION_REPORT.md`, `PARITY_REPORT.json`, `BENCHMARK.json`, `REUSE_AUDIT.json`, and `UI_CHECKLIST.md` for measured results. On the local AMD host with a four-CPU container quota, the full benchmark gave 120-second warm p95 16.20 s and cold p95 21.36 s; the exact final-image check gave warm p95 12.38 s and peak parent-plus-child RSS 1.69 GB. These are local measurements, not guarantees on EC2, sustained load, or all audio conditions.

The model is licensed CC BY-NC-SA 4.0; preserve `MODEL_ATTRIBUTION.md`. This is non-commercial research. Active-speech validity is an energy heuristic, timelines are indicative, thresholds are DEV-based, and NII-Small has not passed independent service validation. `RUNBOOK_AWS.md`, production Compose, Caddy, and `scripts/deploy.sh` are prepared assets only. AWS deployment, registry pushes, purchases, credential use, and public exposure require owner approval.

## Experimental partial-insertion screen

2026-10-12 (Asia/Seoul). DEV synthetic insertion composites, not independent validation. The screen did not beat the baseline under the prescribed gate. It remains experimental. Clean REAL FPR 10.76%; same_full detection 75.43% versus v1 file-level warning 3.44%; paired gain CI [0.654350, 0.785850]. The optional `bundles/nii-small-v1-partial` screen reuses existing scores, uses clean TRAIN REAL-only count-bin thresholds, and reports indicative 4 s segments. It makes no extra detector calls. v1 remains the default; opt in locally with `ACTIVE_BUNDLE=bundles/nii-small-v1-partial docker compose up -d --force-recreate`. Restore with `docker compose up -d --force-recreate` after unsetting the override. Short/unsupported window counts retain the v1 decision. See `work/partial_mvp/PARTIAL_MVP_REPORT.md` and `work/partial_mvp/partial_eval.json` for rates, refusals, cluster intervals, parity, resource checks and limitations. AIHub identifiers/audio and per-case evidence remain in ignored `work/`; bundle files contain aggregates only.

## Option C presentation

2026-10-17 (Asia/Seoul). Presentation-only change; the model, preprocessing, scores, thresholds, outcome rules, partial logic and every bundle file are unchanged. Every response field that existed before is byte-identical; see `REPORT_OPTION_C.md`.

- Two independent verdicts are shown side by side. "전체 음성 합성 검사" is the unchanged v1 file mean and thresholds. "부분 합성 검사 (실험적)" is the unchanged partial2 W0 length-adjusted peak test. Each card states what it can miss.
- Successful analyses gain an additive `display` object (`pipeline/display.py`, version `display-v1-20261017`): a 0–100 "AI 합성 흔적 지표" (`file_index`, per-slot `index`, partial `threshold_index` and segment `peak_index`). It is piecewise-linear through -6.0→0, `t_low`→33.3, `t_warn`→66.7 and 5.5→100, clamped to the band of the unchanged outcome. `is_probability` is always false. Anchors -6.0/5.5 come from TRAIN single-clip score ranges; thresholds come from the loaded bundle. `cannot` responses have no `display`. `/api/config` adds `display_version`.
- The 2-second slots are one heat strip. Amber highlighting and outlined, playable regions (top three by peak index, "더 보기" for the rest) appear only when the partial verdict is `suspected`. Otherwise the strip is grayscale inside a collapsed "구간별 지표 보기" section with a warning that genuine speech often has high slots. Raw scores appear only in "재현 정보".
- With the default `bundles/nii-small-v1`, the partial card says "현재 구성에서는 부분 합성 검사가 제공되지 않습니다". Option C is fully visible only with `ACTIVE_BUNDLE=bundles/nii-small-v1-partial2 docker compose up -d --force-recreate --wait`. Restore the default by unsetting the override and recreating the service.
- The partial verdict is experimental. On DEV-B synthetic composites (not independent validation), genuine files were falsely flagged 9.85% of the time, and 1–2 s inserts and voice-conversion substitutions were mostly missed. Do not enable it publicly before a user comprehension test and a wording review.

## Donation storage (2026-10-22)

Donations are **off by default** (`DONATIONS_ENABLED=false`). When enabled, consent version `donation-consent-v4-2026-10-22` applies (demonstration wording: consent record only, no audio retained; `DONATION_STORE_AUDIO=true` would need storage wording again) (draft pending legal review, see `LEGAL_REVIEW.md` and the `/privacy` policy draft): explicit opt-in before upload, explicit confirmation after the result together with a declaration that the donor is 14 or older and every recorded speaker consented (`eligibility_confirmed`, enforced by the server), original audio (including file metadata) deleted 30 days after donation, withdrawal at any time with the one-time code, and deletion of all audio, consent records and copies when the project ends. Withdrawal (`DELETE /api/donations/{code}`) works even when donations are switched off.

Settings (environment): `DONATIONS_ENABLED`, `DONATION_MAX_BYTES` (4 GiB), `DONATION_MAX_FILES` (1000), `DONATION_DAILY_MAX` (200 per UTC day), `DONATION_HOURLY_MAX` (40 in any rolling hour, counted from the ledger so restarts cannot reset it), `DONATION_MIN_FREE_BYTES` (512 MiB on the donation filesystem), `DONATION_RETENTION_DAYS` (30; values above 30 are refused). `DONATION_STORE_AUDIO` (default **false**, since 2026-10-22): demonstration mode. Confirmation and withdrawal work as before, but only the consent record (code hash, time, consent version, outcome, bundle; no audio hash) is kept and the audio copy is dropped. Set `true` only if the project again decides to keep donated audio. A limit returns `donation_capacity_reached` (507 on confirmation; on analysis the result is returned without a token). Files are written atomically with mode 0600; startup removes orphans/temporaries and expired records; expiry runs hourly. Storage events are logged as counts only (`speech.storage`).

AWS host layout: `/srv/nii-donations` is a 5 GiB ext4 loop filesystem (`/var/lib/nii/donations.img`, `nosuid,nodev,noexec`, fstab `nofail`), mounted into the service at `/data/donations`. The unmounted mountpoint is immutable, and `docker.service` has `RequiresMountsFor=/srv/nii-donations`, so the service fails closed instead of writing to the root disk. `nii-storage-check.timer` logs free space and counts every 15 minutes (`journalctl -t nii-storage`). The public proxy is `caddy-rl:2.10.2-rl0.1.0` (Caddy 2.10.2 + `github.com/mholt/caddy-ratelimit@v0.1.0`), with per-IP limits (site 120/min, analysis 6/10 min, donation 5/h, withdrawal 30/h, token discard 30/h), limits shared by all clients (analysis 60/10 min, donation 40/10 min), and header/body/write/idle timeouts 10/120/180/60 s. Since 2026-10-22 the proxy configuration is shared by every release (`ops/`, installed to `/opt/nii/proxy`), so release switches keep the proxy and its counters; `ops/proxy_mode.sh emergency` pauses analysis and donation confirmation (503 `service_paused`) while withdrawal stays available.

Rollback order (see `RUNBOOK_AWS.md`): (1) `ops/proxy_mode.sh emergency` or `DONATIONS_ENABLED=false`; (2) `ops/switch_release.sh don-20261022-91acf94` (previous donation release, consent v2 text); pre-donation releases only after the project-end purge. `ops/switch_release.sh` refuses, while any donation is stored, a target whose image lacks withdrawal/expiry or whose overlay does not use the donation filesystem and shared proxy, and rolls back if withdrawal does not answer 404 after the switch.

**Project-end deletion** (owner action; irreversible):
1. Stop public access, then purge inside the service container: `docker compose ... run --rm --no-deps service python scripts/purge_donations.py --confirm DELETE-ALL-DONATIONS` (prints remaining files/rows, both must be 0).
2. Stop the stack, `umount /srv/nii-donations`, remove its `/etc/fstab` line, then delete `/var/lib/nii/donations.img` (and `chattr -i /srv/nii-donations` if the directory itself is removed). Also remove the legacy `release_donations` volume.
3. Confirm no backup copies exist with `python ops/aws_backup_audit.py` (EBS snapshots, AMIs, AWS Backup, DLM, audio in S3); there were none on 2026-10-22. Never create snapshots or AMIs of this instance while donations exist. Delete local test evidence that might contain copies (none contain audio).

## Update 2026-10-23

See `REPORT_FINAL_UPDATE_20261023.md`:
- **UI fixes:** wording and stale-result fixes; HEAD support.
- **FINAL audit:** FINAL v2 is already consumed, so no new run was made.
- **Performance disclosure:** measured figures (`app/performance.py`, shown as "측정된 성능과 한계").
- **Legal review and policy restructure.**
- **Contribution pipeline (dry-run only):** `donations/mlops/`, `scripts/mlops.py`.
- **Partial check:** enabled with `ACTIVE_BUNDLE=bundles/nii-small-v1-partial2` (deployment pending owner approval).

