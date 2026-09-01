#!/usr/bin/env python3
"""Assemble the anonymized supplementary video reel from the recorded paired runs.

Inputs (recorded by analysis/record_supplement.sh, seed 0, matched-seed protocol):
  logs/supplement_videos/spoon_n18/   spoon-on-towel semantic n18, baseline+cgvd, eps 0-9
  logs/supplement_videos/carrot_n18/  carrot-on-plate semantic n18, baseline+cgvd, eps 0-9
  logs/supplement_videos/attr_n4/     green-handle spoon attribute n4, baseline+cgvd, eps 0-10
  logs/supplement_videos/byovla_n18/  spoon-on-towel semantic n18, BYOVLA arm, eps 0-1
  supplement/pipeline_4panel_ep12.mp4 4-panel debug video (n18 replication run 6, ep 12)

Output: supplement/CGVD_supplement.mp4 (1280x720, 10 fps, ~100 s, no metadata).

Episode selection is hardcoded below and covers both directions honestly:
rescues (baseline fail -> CGVD success) AND regressions (baseline success ->
CGVD fail), plus the carrot negative result and the BYOVLA comparison.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "supplement" / "build"
OUT = ROOT / "supplement" / "CGVD_supplement.mp4"

SPOON = ROOT / "logs/supplement_videos/spoon_n18/spoon/semantic/n18_e10_r1_20260901_003236/run_0"
CARROT = ROOT / "logs/supplement_videos/carrot_n18/carrot/semantic/n18_e10_r1_20260901_003916/run_0"
ATTR = ROOT / "logs/supplement_videos/attr_n4/spoon/attribute/n4_e11_r1_20260901_004557/run_0"
BYOVLA = ROOT / "logs/supplement_videos/byovla_n18/spoon/semantic/n18_e2_r1_20260901_005135/run_0"
PIPELINE = ROOT / "supplement/pipeline_4panel_ep12.mp4"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WHITE, GRAY, GREEN, RED = "0xF2F2F2", "0xB8B8C0", "0x6FD695", "0xE06868"
BG = "0x14141C"
ENC = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
       "-r", "10", "-map_metadata", "-1", "-an", "-y"]

_ntf = 0
def _textfile(text):
    global _ntf
    _ntf += 1
    p = BUILD / f"txt_{_ntf:03d}.txt"
    p.write_text(text)
    return p

def dt(text, y, size=24, color=WHITE, bold=False, xc=640):
    """One centered drawtext filter (centered on x=xc)."""
    tf = _textfile(text)
    font = FONTB if bold else FONT
    return (f"drawtext=fontfile={font}:textfile={tf}:fontsize={size}"
            f":fontcolor={color}:x={xc}-text_w/2:y={y}")

def run(args):
    subprocess.run(args, check=True, capture_output=True, text=True)

def card(name, dur, lines):
    """lines: list of (text, y, size, color, bold)."""
    vf = ",".join(dt(t, y, s, c, b) for t, y, s, c, b in lines)
    out = BUILD / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={BG}:s=1280x720:r=10:d={dur}", "-vf", vf] + ENC + [str(out)])
    return out

def pair(name, left, right, header, prompt, note, llabel, rlabel, lok, rok):
    """Side-by-side 640x480 pair on the 1280x720 canvas with labels."""
    texts = [
        dt(header, 24, 22, GRAY),
        dt(prompt, 58, 30, WHITE, bold=True),
        dt(f"{llabel} — {'SUCCESS' if lok else 'FAILURE'}", 648, 26,
           GREEN if lok else RED, bold=True, xc=320),
        dt(f"{rlabel} — {'SUCCESS' if rok else 'FAILURE'}", 648, 26,
           GREEN if rok else RED, bold=True, xc=960),
        dt(note, 692, 18, GRAY),
    ]
    fc = ("[0:v]scale=640:480[l];[1:v]scale=640:480[r];[l][r]hstack[v];"
          f"[v]pad=1280:720:0:150:color={BG}," + ",".join(texts) + "[out]")
    out = BUILD / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", fc, "-map", "[out]"] + ENC + [str(out)])
    return out

def triple(name, vids, header, note, labels):
    """Three panels scaled to 1280x320."""
    texts = [dt(header, 60, 28, WHITE, bold=True), dt(note, 640, 20, GRAY)]
    for lab, xc in zip(labels, (213, 640, 1066)):
        texts.append(dt(lab, 566, 22, WHITE, bold=True, xc=xc))
    fc = ("[0:v]scale=640:480[a];[1:v]scale=640:480[b];[2:v]scale=640:480[c];"
          "[a][b][c]hstack=inputs=3,scale=1280:320[v];"
          f"[v]pad=1280:720:0:230:color={BG}," + ",".join(texts) + "[out]")
    out = BUILD / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-i", str(vids[0]), "-i", str(vids[1]),
         "-i", str(vids[2]), "-filter_complex", fc, "-map", "[out]"] + ENC + [str(out)])
    return out

def wide(name, vid, header, lines):
    """The 1920x360 4-panel pipeline video scaled to 1280x240."""
    texts = [dt(header, 80, 28, WHITE, bold=True)]
    for i, t in enumerate(lines):
        texts.append(dt(t, 560 + 34 * i, 20, GRAY))
    fc = (f"[0:v]scale=1280:240,fps=10,pad=1280:720:0:280:color={BG},"
          + ",".join(texts) + "[out]")
    out = BUILD / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-i", str(vid),
         "-filter_complex", fc, "-map", "[out]"] + ENC + [str(out)])
    return out

def ep(run_dir, arm, task, cat, e, ok):
    tag = "SUCCESS" if ok else "FAILED"
    d = "cgvd" if arm == "byovla" else arm  # BYOVLA arm records under cgvd/
    return run_dir / d / f"try_widowx_{task}_{cat}_{d}_ep{e}_{tag}.mp4"

def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    segs = []

    segs.append(card("s00_title", 5, [
        ("Supplementary Video", 200, 44, WHITE, True),
        ("Concept-Gated Visual Distillation (CGVD)", 280, 30, WHITE, False),
        ("Anonymized ICRA submission", 330, 22, GRAY, False),
        ("All clips: SimplerEnv Bridge tasks, seed-matched paired episodes (seed 0)", 430, 22, GRAY, False),
        ("Left: raw observation (baseline policy)    Right: CGVD-distilled observation", 466, 22, GRAY, False),
    ]))

    # --- A: spoon semantic n18 rescues + one regression ---
    segs.append(card("s01_cardA", 3.5, [
        ("A. Semantic clutter — 18 utensil distractors", 270, 34, WHITE, True),
        ('"put the spoon on the towel"', 335, 26, WHITE, False),
        ("Three seed-matched episodes where the baseline fails and CGVD succeeds", 400, 22, GRAY, False),
    ]))
    for e in (2, 3, 7):
        segs.append(pair(f"s02_spoon_ep{e}",
            ep(SPOON, "baseline", "spoon_on_towel", "semantic", e, False),
            ep(SPOON, "cgvd", "spoon_on_towel", "semantic", e, True),
            "A. Semantic clutter, 18 distractors",
            '"put the spoon on the towel"',
            f"seed 0 · episode {e} · identical layout and RNG in both panels",
            "Baseline π0", "π0 + CGVD", False, True))
    segs.append(card("s03_cardA2", 3, [
        ("Distillation is not free", 300, 34, WHITE, True),
        ("Same run: an episode where the baseline succeeds and CGVD fails", 365, 22, GRAY, False),
    ]))
    segs.append(pair("s04_spoon_ep5",
        ep(SPOON, "baseline", "spoon_on_towel", "semantic", 5, True),
        ep(SPOON, "cgvd", "spoon_on_towel", "semantic", 5, False),
        "A. Semantic clutter, 18 distractors — regression",
        '"put the spoon on the towel"',
        "seed 0 · episode 5", "Baseline π0", "π0 + CGVD", True, False))

    # --- B: carrot negative result ---
    segs.append(card("s05_cardB", 3.5, [
        ("B. Negative result — carrot on plate, 18 distractors", 270, 34, WHITE, True),
        ('"put the carrot on the plate"', 335, 26, WHITE, False),
        ("CGVD lowers success on this task (paper Sec. IV); two representative regressions", 400, 22, GRAY, False),
    ]))
    for e in (0, 1):
        segs.append(pair(f"s06_carrot_ep{e}",
            ep(CARROT, "baseline", "carrot_on_plate", "semantic", e, True),
            ep(CARROT, "cgvd", "carrot_on_plate", "semantic", e, False),
            "B. Negative result — carrot on plate, 18 distractors",
            '"put the carrot on the plate"',
            f"seed 0 · episode {e}",
            "Baseline π0", "π0 + CGVD", True, False))

    # --- C: attribute grounding ---
    segs.append(card("s07_cardC", 3.5, [
        ("C. Attribute grounding — 4 same-category spoons", 270, 34, WHITE, True),
        ('"put the spoon with the green handle on the towel"', 335, 26, WHITE, False),
        ("Concept gating removes the other spoons and keeps the referent", 400, 22, GRAY, False),
    ]))
    for e in (2, 6, 10):
        segs.append(pair(f"s08_attr_ep{e}",
            ep(ATTR, "baseline", "spoon_on_towel", "attribute", e, False),
            ep(ATTR, "cgvd", "spoon_on_towel", "attribute", e, True),
            "C. Attribute grounding — 4 same-category spoons",
            '"put the spoon with the green handle on the towel"',
            f"seed 0 · episode {e}",
            "Baseline π0", "π0 + CGVD", False, True))

    # --- D: BYOVLA comparator, same episode three streams ---
    segs.append(card("s09_cardD", 4.5, [
        ("D. Comparator — BYOVLA-style per-frame inpainting", 250, 34, WHITE, True),
        ("Same episode, three observation streams (all three arms fail this episode)", 320, 22, WHITE, False),
        ("BYOVLA probes and re-inpaints every action chunk: 11.3 s per intervention", 380, 22, GRAY, False),
        ("CGVD composites each frame from a cached clean plate: 16 ms per frame", 414, 22, GRAY, False),
    ]))
    segs.append(triple("s10_byovla_ep0", [
        ep(SPOON, "baseline", "spoon_on_towel", "semantic", 0, False),
        ep(BYOVLA, "byovla", "spoon_on_towel", "semantic", 0, False),
        ep(SPOON, "cgvd", "spoon_on_towel", "semantic", 0, False),
    ], "D. Same episode (seed 0, episode 0), three observation streams",
       "Compare the temporal stability of the middle panel with the right panel",
       ["Baseline (raw)", "BYOVLA-style (per-frame)", "CGVD (cached clean plate)"]))

    # --- E: pipeline internals ---
    segs.append(card("s11_cardE", 4, [
        ("E. Pipeline internals — 18-distractor spoon episode", 250, 34, WHITE, True),
        ("raw  |  distractor query  |  safe set (green) + robot (blue)  |  distilled VLA input", 320, 22, WHITE, False),
        ("Red contours in the last panel mark the removal-mask boundary", 380, 22, GRAY, False),
    ]))
    segs.append(wide("s12_pipeline", PIPELINE,
        "E. Pipeline internals (seed-0 replication run, episode 12)",
        ["Panels: raw observation | distractor query | safe-set query | distilled VLA input",
         "One distractor missed by the segmenter remains in the distilled view —",
         "detection is not perfect; see the erasure and OOV audits in the paper"]))

    segs.append(card("s13_end", 4, [
        ("Clips show both improvements and regressions from seed-0 paired runs", 300, 24, WHITE, False),
        ("Full protocol and statistics (10 seeds × 20 episodes per condition): see paper", 345, 22, GRAY, False),
    ]))

    concat = BUILD / "concat.txt"
    concat.write_text("".join(f"file '{s}'\n" for s in segs))
    run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", "-map_metadata", "-1", "-y", str(OUT)])
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
