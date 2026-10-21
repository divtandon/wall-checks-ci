# wall-checks-ci

A small CI/CD pipeline (GitHub Actions + Docker + GitHub Container Registry) for checks that come from real bugs in
my capstone facade experiments. It runs on **synthetic walls only**; no project code or data is in this repo.

| Test | The real problem behind it |
|---|---|
| `test_iou_*` | the area-weighted overlap score every result depends on |
| `test_rebuilt_wall_triangle_count` | rebuilding a wall must give a predictable triangle count |
| `test_recessed_windows_have_negative_depth` | my depth map once showed windows in front of the wall (wrong normal sign) |
| `test_depth_rule_finds_the_windows` | the depth rule labels recessed samples as window |
| `test_deep_windows_fall_outside_the_30cm_band` | 35 cm deep windows were counted as wall by a 30 cm band |
| `test_same_seed_gives_same_points` | unseeded randomness changed scores between identical runs |

## Pipeline

On every push to `main`:

1. **test**: set up Python 3.12, install the pinned `requirements.txt`, and run `pytest -v`.
2. **build-and-push**: runs only if every test passed. It builds the Docker image and pushes it to
   `ghcr.io/<owner>/wall-checks-ci:<commit SHA>`.

## Run locally

```bash
pip install -r requirements.txt && pytest -v
docker build -t wall-checks-ci . && docker run --rm wall-checks-ci
```
