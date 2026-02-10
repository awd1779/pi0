#!/usr/bin/env python3
"""Plot baseline vs CGVD comparison from sweep results."""

import matplotlib.pyplot as plt
import numpy as np

# Data from sweep_report.md
distractors = [0, 2, 4, 6, 8]

# Baseline results
baseline_mean = [86.0, 48.0, 34.0, 32.0, 34.0]
baseline_std = [9.16, 14.69, 18.54, 16.0, 10.19]

# CGVD results (0 distractors = N/A, use baseline value for reference)
cgvd_mean = [86.0, 77.0, 67.0, 64.0, 61.0]  # Using baseline for 0 distractors
cgvd_std = [9.16, 10.04, 11.87, 14.96, 14.45]  # Using baseline std for 0

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with just dots and lines
x = np.array(distractors)

# Baseline
ax.plot(x, baseline_mean, 'o-', linewidth=2, markersize=10,
        color='#E74C3C', label='Baseline (No CGVD)')

# CGVD
ax.plot(x, cgvd_mean, 's-', linewidth=2, markersize=10,
        color='#27AE60', label='CGVD (Ours)')

# Add annotations for both lines
for i, (b, c, d) in enumerate(zip(baseline_mean, cgvd_mean, distractors)):
    # Baseline percentage
    ax.annotate(f'{b:.0f}%',
                xy=(d, b - 5),
                ha='center', fontsize=10, color='#E74C3C', fontweight='bold')
    # CGVD percentage
    ax.annotate(f'{c:.0f}%',
                xy=(d, c + 4),
                ha='center', fontsize=10, color='#27AE60', fontweight='bold')

# Styling
ax.set_xlabel('Number of Distractors', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Baseline vs CGVD Performance\n(widowx_spoon_on_towel task)', fontsize=14, fontweight='bold')
ax.set_xticks(distractors)
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(0, 105)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Add horizontal line at baseline 0-distractor performance
ax.axhline(y=86, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
ax.text(8.3, 87, '0-distractor\nbaseline', fontsize=9, color='gray', va='bottom')

# Tight layout
plt.tight_layout()

# Save
output_path = '/home/ubuntu/open-pi-zero/logs/full_eval_spoon_towel/baseline_vs_cgvd.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved chart to: {output_path}")

# Also save PDF for paper
pdf_path = '/home/ubuntu/open-pi-zero/logs/full_eval_spoon_towel/baseline_vs_cgvd.pdf'
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved PDF to: {pdf_path}")

plt.show()
