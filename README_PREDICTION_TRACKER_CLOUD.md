# Prediction Tracker — all-cloud architecture

```text
Prediction Tracker
      ↓
Cloud Run sharp-train-job
      ↓
Web Unblocker / residential proxy
      ↓
validation (challenge rejection + identity checks)
      ↓
GCS exact raw + immutable snapshots + feeder_manifest.json
      ↓
NCAAF GCS-first parser
      ↓
5 named rating systems + fixed META_MARGIN
      ↓
Research Miner external-rating family
```

There is no local feeder, PowerShell script, local Python, local gcloud, or desktop browser requirement.

The cloud feeder never overwrites a good raw object with a failed/challenge response. Completed valid historical seasons are reused. Invalid old Cloudflare blobs are replaced only after a new response passes validation. Current 2026/archive/live files are refreshed on every cloud feeder run.

The five external systems remain one correlated external-rating family for authority/dependency purposes. The external family remains research-only until it independently satisfies the existing validation gates.
