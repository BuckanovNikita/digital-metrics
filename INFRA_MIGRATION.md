# Infrastructure migration for digital-metrics

## Why this notice exists

The shared development stands managed by k8s-infra are being removed in one
sitting, and this project's migration is delayed. The wipe date is not set.
Tests that need those stands or their credentials will fail when the k8s-infra
wipe is carried out (plan phase P3), until this migration is complete.
The shared data is destroyed without backups or imports. The old service
helpers, run tags, registry contract and credentials are retired with the wipe.

## What stops working here

The plan identifies this project's ClearML key pair as the broken dependency.
Any live ClearML check using that pair must stop relying on the shared server.
The relevant repository paths are:

- `src/digital_metrics/tracking/clearml_tracker.py` creates ClearML tasks and
  publishes evaluation results through the optional tracking layer.
- `docs/tracking.md` and `docs/tracking.ru.md` describe that live tracking usage.
- `tests/test_clearml_tracker.py` uses injected fake tasks, so its checks do not
  demonstrate live authentication and do not depend on the removed stand.
- `docs/development/metric-internals.md` describes the optional ClearML boundary
  that the migration must preserve.

Repository searches found no checked-in shared host endpoints, Compose launcher,
or retired infrastructure-helper calls. Core metric computation is independent
of ClearML; this notice does not claim that its offline tests will fail.

## What the platform provides afterwards

The platform provides ingress, a certificate issued by the local CA, Docker
Desktop's default `hostpath` storage class, and one exclusive GPU slot without
time-slicing. Request `nvidia.com/gpu: 1` only for tests that need it.
The certificate covers single-label hosts directly under `k8s.docker.internal`,
not deeper names. Wildcard name lookup is possible later, only if phase P4
proves it. Until then, and if that proof fails, use `kubectl port-forward`.
Fresh dedicated ClearML and FiftyOne MongoDB services are not replacements for
this project's lost access; no current project receives a bundle at the start.

## Namespace lifetime

The sweeper deletes every namespace outside k8s-infra's protected list at
48 hours after creation, even while active, with no renewal. It also deletes
namespaces created before a PC or Docker Desktop restart. It runs every
30 minutes. A missing `ResourceQuota` is reported, but neither exempts a
namespace nor independently causes deletion; age and restart are the triggers.
Deletion is armed only after the plan's restart proofs and owner gate.
The sweeper manages Kubernetes namespaces, not host processes or worktrees.

## Migration rules

1. Deploy test infrastructure into this project's own namespaces using its own
   manifests or chart, with its own teardown.
2. Use a distinct namespace prefix such as `digital-metrics-it-<id>`. Never reuse
   protected names: `clearml`, `mongodb`, `ingress-nginx`, `infra-janitor`,
   `default`, or any `kube-*` name.
3. Every namespace created by the project must carry a project-chosen
   `ResourceQuota`. Migration is incomplete without it; the sweeper report is
   the check. Add a `LimitRange` as well, because a CPU or memory quota rejects
   pods that set no requests.
4. Omit `storageClassName` or use the default storage class. Prefer `emptyDir`
   for disposable data. Remove references to `local-path` and
   `/mnt/wsl/data/k8s-volumes`.
5. Finish runs within the namespace's lifetime of 48 hours or until a restart.
   Tooling must recreate a namespace that has disappeared.
6. Current workloads must not use the dedicated ClearML or FiftyOne services.
   Deploy this project's own ClearML and FiftyOne database wherever needed.
   Dedicated services require an owner-issued bundle file, and the owner issues
   none at the start.
7. Request `nvidia.com/gpu: 1` only where a test needs it; there is one slot.
8. Remove `INFRA_RUN_TAG`, `INFRA_PROJECT`, `K8S_INFRA_SKILL_DIR`,
   `INFRA_REGISTRY`, and calls to `infra.py`, `clearml.py`, `mongodb.py`,
   `cvat.py`, `lakefs.py`, and `postgres.py` from code, skills and shell profiles.
9. After P4 proves wildcard lookup, use a single-label host directly under
   `k8s.docker.internal` on the project's own Ingress with the platform
   certificate. Until then, and if P4 fails, use `kubectl port-forward`.

## Required outcome and project workflow

Every live ClearML check runs against a ClearML deployment owned by this project
in its own namespace, with a quota and teardown. Deployment design belongs to
this repository; this notice does not select an implementation.
Use the repository's contract and verification workflow in `AGENTS.md`,
`docs/development/api-contract.md` and `docs/local-release.md`. No named Spec Kit
or OpenSpec workflow is prescribed there. Preserve the optional tracking
boundary, use pytest for product behavior, and complete the configured prek
checks with `SKIP` unset after the final migration commit. Record live ClearML
acceptance separately from fake-task unit checks.

## Done when

The project owner has been told that digital-metrics has moved, its live checks
use its own ClearML, and the sweeper report shows its namespaces carrying a
quota. Then delete this file.

Source plan: `/mnt/wsl/data/nkt/k8s-infra/docs/plans/2026-11-09-managed-infrastructure-reset.md`.
