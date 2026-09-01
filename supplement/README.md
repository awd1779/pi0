# Supplementary video — provenance

`CGVD_supplement.mp4` (1280×720, 10 fps, 98.8 s, ~6.8 MB, no metadata) is the
anonymized supplementary reel for the ICRA submission. Rebuild any time with:

```
python3 analysis/build_supplement_reel.py
```

(needs only host ffmpeg + the DejaVu fonts; intermediates land in `supplement/build/`).

## Source recordings

All clips come from `analysis/record_supplement.sh` (run 2026-09-01 on this
machine, A10G, Docker `drodii/open-pi-zero:latest`), which re-runs seed-0 of the
paper's matched-seed protocol with `--recording`. Raw mp4s under
`logs/supplement_videos/` (gitignored; kept on disk):

| Arm | Task / condition | Episodes | Baseline | CGVD |
|---|---|---|---|---|
| `spoon_n18` | spoon-on-towel, semantic, n=18 | 0–9 | 5/10 | 7/10 |
| `carrot_n18` | carrot-on-plate, semantic, n=18 | 0–9 | 6/10 | 4/10 |
| `attr_n4` | green-handle spoon, attribute, n=4 | 0–10 | 4/11 | 7/11 |
| `byovla_n18` | spoon-on-towel, semantic, n=18, BYOVLA arm | 0–1 | — | 0/2 |

Directions match the paper: CGVD helps on spoon semantic and attribute, hurts
on carrot semantic. Single-seed rates differ from the 10-seed table means as
expected.

## Episode selection (hardcoded in the build script)

- **A (spoon n18)**: all three rescues of the run (eps 2, 3, 7: baseline FAIL →
  CGVD SUCCESS) **plus** one of the two regressions (ep 5: baseline SUCCESS →
  CGVD FAIL), shown under its own "distillation is not free" card.
- **B (carrot n18)**: two of the three regressions (eps 0, 1) — the paper's
  negative result, shown as such.
- **C (attribute n4)**: three of the four rescues (eps 2, 6, 10).
- **D (BYOVLA)**: episode 0 as a 3-panel baseline | BYOVLA | CGVD comparison
  (same seed/episode → identical layout). All three arms fail this episode; the
  clip demonstrates observation-stream stability, and the card says so.
  Measured inter-frame mean |Δ| on ep 0: baseline 3.03 (p95 4.89), BYOVLA 3.65
  (p95 7.30), CGVD 2.28 (p95 4.90) — BYOVLA's per-frame inpainting is ~1.5×
  spikier at p95 than either alternative.
- **E (pipeline internals)**: `pipeline_4panel_ep12.mp4`, assembled earlier from
  the cgvd_debug frames of `logs/replication_masks_n18/.../run_6/`, episode 12.
  Panel 4 keeps one segmenter-missed distractor in view; the caption points at
  it rather than hiding it.

Caption numbers (11.3 s/intervention BYOVLA, 16 ms/frame CGVD compositing) are
the measured values from the revision (see `analysis/out/` and the paper's
latency section).

## Anonymity

No names, affiliations, or repo URLs anywhere in the reel; ffmpeg strips
container metadata (`-map_metadata -1`). Simulation footage only.
