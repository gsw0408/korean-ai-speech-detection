import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.path_guard import (
    ProtectedPathError, guard_member, guard_path, guard_source_container, read_bytes, sha256_file,
)


class PathGuardTests(unittest.TestCase):
    def test_each_prohibited_marker(self):
        for marker in ("calibration", "CALIB", "final", "FINAL", "F_final",
                       "VL_data", "verification", "CaLiBrAtIoN"):
            with self.subTest(marker=marker):
                with self.assertRaises(ProtectedPathError):
                    guard_path("D:/research/" + marker + "/audio.wav")

    def test_explicit_private_roots(self):
        for root in ("e2fr", "fr"):
            for separator in ("/", "\\"):
                value = ("C:/Users/owner/.codex/tmp/" + root + "/x.wav")
                with self.subTest(root=root, separator=separator):
                    with self.assertRaises(ProtectedPathError):
                        guard_path(value.replace("/", separator))

    def test_exact_private_root(self):
        with self.assertRaises(ProtectedPathError):
            guard_path("C:/Users/owner/.codex/tmp/fr")

    def test_private_root_aliases_rejected_before_resolution(self):
        for path in ("/mnt/c/Users/owner/.codex/tmp/fr/x.wav", "//?/C:/Users/owner/.codex/tmp/e2fr/x.wav", "//server/share/audio.wav", "//?/UNC/server/share/audio.wav"):
            with self.subTest(path=path), patch.object(Path, "resolve") as resolve:
                with self.assertRaises(ProtectedPathError):
                    guard_path(path)
                resolve.assert_not_called()

    def test_private_root_normalization(self):
        with self.assertRaises(ProtectedPathError):
            guard_path("C:/Users/owner/.codex/tmp/allowed/../fr/x.wav")

    def test_lexical_private_root_rejected_before_traversal_out(self):
        with patch.object(Path, "resolve") as resolve:
            with self.assertRaises(ProtectedPathError):
                guard_path("C:/Users/owner/.codex/tmp/e2fr/../allowed/x.wav")
            resolve.assert_not_called()

    def test_resolved_destination_checked(self):
        with patch.object(Path, "resolve", return_value=Path("D:/CALIB/x.wav")):
            with self.assertRaises(ProtectedPathError):
                guard_path("D:/allowed/link.wav")

    def test_reject_before_resolving_forbidden_path(self):
        with patch.object(Path, "resolve") as resolve:
            with self.assertRaises(ProtectedPathError):
                guard_path("D:/FINAL/x.wav")
            resolve.assert_not_called()

    def test_source_containers_remain_blocked(self):
        for path in (
            "C:/Users/owner/.codex/tmp/dataset-v2-pre-final/transfer/FAKEGEN_v2_TRAIN_DEV_20261005.zip.bin",
            "D:/research/kaggle_execution_p5_final_20261008/input_uploads/evidence/POST_FREEZE_TRAINING_EVIDENCE.zip.bin",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ProtectedPathError):
                    guard_path(path)

    def test_protected_members(self):
        for member in ("audio/calibration/x.wav", "results/FINAL.json",
                       "verification/x.json", "audio/VL_data/x.wav"):
            with self.subTest(member=member):
                with self.assertRaises(ProtectedPathError):
                    guard_member(member)

    def test_member_traversal(self):
        for member in ("../x.wav", "audio/../../x.wav", "audio\\..\\x.wav",
                       "audio/./x.wav", "D:/audio/x.wav", "/audio/x.wav",
                       "\\\\server\\share\\x.wav", "audio/dev/x.wav:stream", "", "x\x00.wav"):
            with self.subTest(member=member):
                with self.assertRaises(ProtectedPathError):
                    guard_member(member)

    def test_permitted_member_names(self):
        for member in ("manifests/dev_required.jsonl", "audio/dev/x.wav",
                       "continuation/nii.json"):
            self.assertEqual(str(guard_member(member)), member)

    def test_permitted_file_read_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.wav"
            path.write_bytes(b"dev fixture")
            self.assertEqual(read_bytes(path), b"dev fixture")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"dev fixture").hexdigest())

    def test_similar_private_root_is_allowed(self):
        guard_path("C:/Users/owner/.codex/tmp/fresh/x.wav")

    def test_forbidden_read_never_opens_file(self):
        with patch.object(Path, "read_bytes") as read:
            with self.assertRaises(ProtectedPathError):
                read_bytes("D:/CALIB/x.wav")
            read.assert_not_called()

    def test_forbidden_hash_never_opens_file(self):
        with patch.object(Path, "open") as open_file:
            with self.assertRaises(ProtectedPathError):
                sha256_file("D:/FINAL/x.wav")
            open_file.assert_not_called()

    def test_container_exception_rejects_other_paths(self):
        for path in ("D:/FINAL/x.zip", "C:/Users/owner/.codex/tmp/e2fr/x.zip",
                     "C:/Users/owner/.codex/tmp/dataset-v2-pre-final/recovery/dataset_v2/audio/calibration/x.wav"):
            with self.assertRaises(ProtectedPathError):
                guard_source_container(path)

    def test_container_exception_checks_resolved_destination(self):
        source = "C:/Users/owner/.codex/tmp/dataset-v2-pre-final/transfer/FAKEGEN_v2_TRAIN_DEV_20261005.zip.bin"
        with patch.object(Path, "resolve", return_value=Path("D:/FINAL/x.zip")):
            with self.assertRaises(ProtectedPathError):
                guard_source_container(source)


if __name__ == "__main__":
    unittest.main()
