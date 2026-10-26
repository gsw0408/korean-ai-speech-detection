import ast
from pathlib import Path


def test_web_layer_uses_only_pipeline_interfaces():
    root = Path(__file__).resolve().parents[2]
    allowed = {"pipeline.bundle", "pipeline.decision", "pipeline.preprocess", "pipeline.service"}
    for source in (root / "app").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in {"numpy", "scipy", "onnxruntime"} for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not any(alias.name in {"decode_canonical", "scoring_windows", "validate_service"} for alias in node.names)
                assert not node.module or not node.module.startswith("pipeline.") or node.module in allowed
                assert not node.module or node.module.split(".")[0] not in {"numpy", "scipy", "onnxruntime"}
