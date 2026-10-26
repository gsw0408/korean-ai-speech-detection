from dataclasses import dataclass
import os
from pathlib import Path

GiB = 1024 ** 3


def _flag(name, default="false", *, strict=False):
    value = os.environ.get(name, default)
    if not strict:
        value = value.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(name + " must be true or false")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    active_bundle: str = "bundles/nii-small-v1"
    max_upload_bytes: int = 25 * 1024 * 1024
    timeout_seconds: int = 120
    onnx_threads: int = 4
    concurrency: int = 1
    root: Path = Path(__file__).resolve().parents[1]
    upload_dir: Path = Path("/tmp/uploads")
    donation_dir: Path = Path("/data/donations")
    ffmpeg: str = "ffmpeg"
    queue_timeout_seconds: float = 30
    donations_enabled: bool = False
    donation_max_bytes: int = 4 * GiB
    donation_max_files: int = 1000
    donation_daily_max: int = 200
    donation_hourly_max: int = 40
    donation_min_free_bytes: int = 512 * 1024 * 1024
    donation_retention_days: int = 30
    donation_store_audio: bool = False
    feedback_url: str = ""

    @classmethod
    def from_env(cls):
        return cls(active_bundle=os.environ.get("ACTIVE_BUNDLE", "bundles/nii-small-v1"),
                   timeout_seconds=float(os.environ.get("ANALYSIS_TIMEOUT", "120")),
                   onnx_threads=int(os.environ.get("ONNX_THREADS", "4")),
                   concurrency=int(os.environ.get("ANALYSIS_CONCURRENCY", "1")),
                   upload_dir=Path(os.environ.get("UPLOAD_DIR", "/tmp/uploads")),
                   donation_dir=Path(os.environ.get("DONATION_DIR", "/data/donations")),
                   ffmpeg=os.environ.get("FFMPEG", "ffmpeg"),
                   donations_enabled=_flag("DONATIONS_ENABLED"),
                   donation_max_bytes=int(os.environ.get("DONATION_MAX_BYTES", str(4 * GiB))),
                   donation_max_files=int(os.environ.get("DONATION_MAX_FILES", "1000")),
                   donation_daily_max=int(os.environ.get("DONATION_DAILY_MAX", "200")),
                   donation_hourly_max=int(os.environ.get("DONATION_HOURLY_MAX", "40")),
                   donation_min_free_bytes=int(os.environ.get("DONATION_MIN_FREE_BYTES", str(512 * 1024 * 1024))),
                   donation_retention_days=int(os.environ.get("DONATION_RETENTION_DAYS", "30")),
                   donation_store_audio=_flag("DONATION_STORE_AUDIO", strict=True),
                   feedback_url=os.environ.get("FEEDBACK_URL", "").strip())
