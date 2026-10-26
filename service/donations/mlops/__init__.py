"""Contributed-data pipeline: intake -> human review -> dataset versions -> training plan -> evaluation gate -> registry.

Policy guards (all fail closed):
- No audio exists while the service runs with DONATION_STORE_AUDIO=false; intake then reports the policy and stops.
- Only consent versions listed in TRAINING_CONSENT_VERSIONS may enter training data. Every consent text used so far
  (v1-v4) says donations are not used for automatic training, so the set is empty until the owner approves a new text.
- Labels come only from human review; the model's own outcome is never used as a label.
- Withdrawals and expiries propagate: intake copies are deleted and dataset versions that held them are superseded.
- Training is planned, never executed, by this package. Evaluation refuses protected CALIB/FINAL data.
- Promotion requires a passing evaluation report and an explicit owner approval string.
"""

TRAINING_CONSENT_VERSIONS = frozenset()
