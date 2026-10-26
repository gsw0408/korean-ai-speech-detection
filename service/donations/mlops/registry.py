"""Model registry with lineage: dataset version -> training plan -> candidate bundle -> evaluation -> promotion."""
from pathlib import Path

from .common import PolicyError, now, read_json, write_json


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.data = read_json(self.path) if self.path.exists() else {"models": {}, "active": None, "history": []}

    def _save(self):
        write_json(self.path, self.data)

    def register(self, model_id, *, bundle_hashes, dataset_version, plan_id, evaluation):
        if model_id in self.data["models"]:
            raise PolicyError("model ids are immutable")
        self.data["models"][model_id] = {"bundle_hashes": dict(bundle_hashes), "dataset_version": dataset_version, "plan_id": plan_id,
                                         "evaluation": {"decision": evaluation["decision"], "report_hash": evaluation["report_hash"]},
                                         "status": "candidate", "registered_at": now()}
        self._save()
        return self.data["models"][model_id]

    def promote(self, model_id, *, owner_approval):
        model = self.data["models"][model_id]
        if model["evaluation"]["decision"] != "promote":
            raise PolicyError("evaluation did not pass; promotion refused")
        if not owner_approval or not owner_approval.strip():
            raise PolicyError("promotion needs an explicit owner approval record")
        previous = self.data["active"]
        self.data["active"] = model_id
        model["status"] = "active"
        if previous:
            self.data["models"][previous]["status"] = "retired"
        self.data["history"].append({"at": now(), "action": "promote", "model": model_id, "previous": previous, "approval": owner_approval})
        self._save()
        return model

    def rollback(self, *, reason):
        entries = [h for h in self.data["history"] if h["action"] == "promote"]
        if not entries or not entries[-1]["previous"]:
            raise PolicyError("no earlier model to roll back to")
        current, previous = entries[-1]["model"], entries[-1]["previous"]
        self.data["models"][current]["status"] = "rolled_back"
        self.data["models"][previous]["status"] = "active"
        self.data["active"] = previous
        self.data["history"].append({"at": now(), "action": "rollback", "model": previous, "previous": current, "reason": reason})
        self._save()
        return previous

    def lineage(self, model_id):
        model = self.data["models"][model_id]
        return {"model": model_id, "dataset_version": model["dataset_version"], "plan_id": model["plan_id"],
                "evaluation": model["evaluation"], "status": model["status"]}
