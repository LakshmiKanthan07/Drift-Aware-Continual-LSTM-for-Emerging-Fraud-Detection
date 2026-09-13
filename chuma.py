import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
activities = [
    "Problem Identification",
    "Literature Review (20+ papers)",
    "Previous Work Analysis",
    "Research Gap Identification",
    "Problem Statement & Objectives",
    "Proposed Methodology",
]

weeks = list(range(1, 13))

ranges = {
    "Problem Identification": (1, 1),
    "Literature Review (20+ papers)": (1, 3),
    "Previous Work Analysis": (2, 3),
    "Research Gap Identification": (2, 3),
    "Problem Statement & Objectives": (3, 3),
    "Proposed Methodology": (3, 4),
}

fig, ax = plt.subplots(figsize=(16, 10))

for i, activity in enumerate(activities):
    start, end = ranges[activity]
    ax.barh(i, end - start + 1, left=start - 0.5, height=0.58)

ax.set_yticks(range(len(activities)))
ax.set_yticklabels(activities, fontsize=10)
ax.set_xticks(weeks)
ax.set_xlim(0.5, 12.5)
ax.set_xlabel("Project Week", fontsize=11)
ax.set_title("Updated Gantt Chart – Drift-Aware Continual LSTM for Emerging Fraud Detection",
             fontsize=15, pad=16, weight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.invert_yaxis()

for x in [3.5, 5.5, 7.5, 9.5, 11.5]:
    ax.axvline(x, linewidth=0.8, alpha=0.25)

plt.tight_layout()
path = "/mnt/data/updated_gantt_chart_fraud_detection.png"
plt.savefig(path, dpi=220, bbox_inches="tight")
plt.close()
print(path)