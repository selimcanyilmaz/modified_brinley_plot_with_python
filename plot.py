import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.stats as stats
import pandas as pd

# Read and prepare the data
data = pd.DataFrame([
    ['S249', 14, 9, 13, 14],
    ['S113', 10, 12, 8, 4],
    ['S189', 19, 17, 10, 10],
    ['S931', 10, 8, 6, 2]
], columns=['PUKI', 'Baseline', 'Intervention', 'Post-int', 'Follow-up'])

def create_modified_brinley_plot(x_scores, y_scores, x_label, y_label):
    """
    Create a modified Brinley plot with additional features
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the axis limits to 0-21
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)

    # Plot diagonal reference line
    ax.plot([0, 21], [0, 21], 'k--', alpha=0.5)

    # Plot vertical and horizontal reference lines at x=5 and y=5
    ax.axvline(x=5, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax.axhline(y=5, color='black', linestyle='-', linewidth=2, alpha=0.5)

    # Plot data points
    ax.scatter(x_scores, y_scores, alpha=0.6, s=100, color='black')

    # Calculate and plot means with + sign
    mean_x = np.mean(x_scores)
    mean_y = np.mean(y_scores)
    ax.plot(mean_x, mean_y, 'r+', markersize=20, markeredgewidth=2)

    # Add mean values annotation
    ax.text(mean_x, mean_y + 1, f'Means:\n{x_label}: {mean_x:.1f}\n{y_label}: {mean_y:.1f}',
            horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    # Calculate differences
    differences = np.array(y_scores) - np.array(x_scores)

    # Calculate Cohen's d
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)

    # Calculate standard error of Cohen's d
    n = len(differences)
    se = np.std(differences, ddof=1) / np.sqrt(n)

    # Calculate confidence interval
    ci = stats.t.ppf(0.975, df=n-1) * se

    # Add statistical information
    stats_text = (f"Cohen's d: {cohens_d:.2f}\n"
                   f"95% CI: [{cohens_d - ci:.2f}, {cohens_d + ci:.2f}]\n"
                   f"Mean diff: {np.mean(differences):.2f}")

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

    # Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Modified Brinley Plot: {x_label} vs {y_label}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.legend()
    return fig, ax

# Create all three comparisons
comparisons = [
    ('Baseline', 'Intervention'),
    ('Baseline', 'Post-int'),
    ('Baseline', 'Follow-up')
]

# Generate all plots
for x_label, y_label in comparisons:
    fig, ax = create_modified_brinley_plot(
        data[x_label].values,
        data[y_label].values,
        x_label,
        y_label
    )
    plt.show()
