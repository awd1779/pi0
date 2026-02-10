#!/usr/bin/env python3
"""Plot comparison of PI0 vs PI0+CGVD success rates across distractor counts, split by category."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Read the sweep summary
data = defaultdict(lambda: defaultdict(lambda: {'baseline': [], 'cgvd': []}))

with open('logs/clutter_eval/pi0/spoon/sweep_summary.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        num_distractors = int(row['num_distractors'])
        category = row['category']
        data[category][num_distractors]['baseline'].append(float(row['baseline_rate']))
        data[category][num_distractors]['cgvd'].append(float(row['cgvd_rate']))

categories = ['control', 'semantic']

# Create the plot with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for idx, category in enumerate(categories):
    ax = axes[idx]
    cat_data = data[category]
    distractor_counts = sorted(cat_data.keys())

    pi0_means = []
    pi0_ses = []
    cgvd_means = []
    cgvd_ses = []

    print(f"\n{category.upper()} Distractors:")
    print(f"{'Distractors':>12} {'PI0 Mean':>10} {'PI0 SE':>8} {'CGVD Mean':>11} {'CGVD SE':>9} {'Improvement':>12}")
    print("-" * 70)

    for nd in distractor_counts:
        baseline = np.array(cat_data[nd]['baseline'])
        cgvd = np.array(cat_data[nd]['cgvd'])

        pi0_mean = np.mean(baseline)
        pi0_se = np.std(baseline) / np.sqrt(len(baseline))
        cgvd_mean = np.mean(cgvd)
        cgvd_se = np.std(cgvd) / np.sqrt(len(cgvd))

        pi0_means.append(pi0_mean)
        pi0_ses.append(pi0_se)
        cgvd_means.append(cgvd_mean)
        cgvd_ses.append(cgvd_se)

        improvement = cgvd_mean - pi0_mean
        print(f"{nd:>12} {pi0_mean:>10.1f} {pi0_se:>8.1f} {cgvd_mean:>11.1f} {cgvd_se:>9.1f} {improvement:>+12.1f}")

    x = np.array(distractor_counts)
    pi0_means = np.array(pi0_means)
    pi0_ses = np.array(pi0_ses)
    cgvd_means = np.array(cgvd_means)
    cgvd_ses = np.array(cgvd_ses)

    # Plot lines with dots
    ax.plot(x, pi0_means, marker='o', markersize=8, linewidth=2, label='PI0', color='#1f77b4')
    ax.plot(x, cgvd_means, marker='s', markersize=8, linewidth=2, label='PI0 + CGVD', color='#ff7f0e')

    # Customize the subplot
    ax.set_xlabel('Number of Distractors', fontsize=14)
    if idx == 0:
        ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title(f'{category.capitalize()} Distractors', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)


fig.suptitle('Spoon on Towel Task: PI0 vs PI0 + CGVD', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig('logs/clutter_eval/pi0/spoon/comparison_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('logs/clutter_eval/pi0/spoon/comparison_plot.pdf', bbox_inches='tight')
print("\nSaved plots to logs/clutter_eval/pi0/spoon/comparison_plot.png and .pdf")

# Calculate average improvements per category
print("\n" + "="*50)
for category in categories:
    cat_data = data[category]
    distractor_counts = sorted(cat_data.keys())
    improvements = []
    for nd in distractor_counts:
        baseline = np.mean(cat_data[nd]['baseline'])
        cgvd = np.mean(cat_data[nd]['cgvd'])
        improvements.append(cgvd - baseline)
    print(f"Average improvement ({category}): {np.mean(improvements):.1f}%")
