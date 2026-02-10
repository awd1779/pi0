#!/usr/bin/env python3
"""Plot comparison of PI0 vs PI0+CGVD success rates as grouped bar chart for IROS paper."""

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
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

bar_width = 0.35

for idx, category in enumerate(categories):
    ax = axes[idx]
    cat_data = data[category]
    distractor_counts = sorted(cat_data.keys())

    pi0_means = []
    pi0_ses = []
    cgvd_means = []
    cgvd_ses = []

    for nd in distractor_counts:
        baseline = np.array(cat_data[nd]['baseline'])
        cgvd = np.array(cat_data[nd]['cgvd'])

        pi0_means.append(np.mean(baseline))
        pi0_ses.append(np.std(baseline) / np.sqrt(len(baseline)))
        cgvd_means.append(np.mean(cgvd))
        cgvd_ses.append(np.std(cgvd) / np.sqrt(len(cgvd)))

    x = np.arange(len(distractor_counts))
    pi0_means = np.array(pi0_means)
    pi0_ses = np.array(pi0_ses)
    cgvd_means = np.array(cgvd_means)
    cgvd_ses = np.array(cgvd_ses)

    # Plot grouped bars with error bars
    bars1 = ax.bar(x - bar_width/2, pi0_means, bar_width, yerr=pi0_ses,
                   label='PI0', color='#1f77b4', capsize=4, error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x + bar_width/2, cgvd_means, bar_width, yerr=cgvd_ses,
                   label='PI0 + CGVD', color='#ff7f0e', capsize=4, error_kw={'linewidth': 1.5})

    # Customize the subplot
    ax.set_xlabel('Number of Distractors', fontsize=13)
    if idx == 0:
        ax.set_ylabel('Success Rate (%)', fontsize=13)
    ax.set_title(f'{category.capitalize()} Distractors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(distractor_counts)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=11)

fig.suptitle('Spoon on Towel Task: PI0 vs PI0 + CGVD', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('logs/clutter_eval/pi0/spoon/comparison_bar_chart.png', dpi=150, bbox_inches='tight')
plt.savefig('logs/clutter_eval/pi0/spoon/comparison_bar_chart.pdf', bbox_inches='tight')
print("Saved bar charts to logs/clutter_eval/pi0/spoon/comparison_bar_chart.png and .pdf")
