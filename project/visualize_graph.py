"""Dual-player skeleton graph figure for report — illustration style."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── COCO 17 joints ──
JOINT_NAMES = [
    "Nose", "L Eye", "R Eye", "L Ear", "R Ear",
    "L Shoulder", "R Shoulder", "L Elbow", "R Elbow",
    "L Wrist", "R Wrist", "L Hip", "R Hip",
    "L Knee", "R Knee", "L Ankle", "R Ankle",
]

# Edges grouped by body part with colors
LIMB_GROUPS = {
    "Head":      {"edges": [(0,1),(0,2),(1,3),(2,4)],   "color": "#E91E63"},
    "Shoulders": {"edges": [(5,6)],                      "color": "#FF9800"},
    "L Arm":     {"edges": [(5,7),(7,9)],                "color": "#4CAF50"},
    "R Arm":     {"edges": [(6,8),(8,10)],               "color": "#2196F3"},
    "Torso":     {"edges": [(5,11),(6,12),(11,12)],      "color": "#FF9800"},
    "L Leg":     {"edges": [(11,13),(13,15)],            "color": "#9C27B0"},
    "R Leg":     {"edges": [(12,14),(14,16)],            "color": "#00BCD4"},
}

NUM_JOINTS = 17

# ── Player 1 (top court, facing down) ──
P1 = {
    0: (0, 5.8),
    1: (-0.2, 6.1),   2: (0.2, 6.1),
    3: (-0.45, 6.35),  4: (0.45, 6.35),
    5: (-0.7, 5.0),   6: (0.7, 5.0),
    7: (-1.3, 4.1),   8: (1.3, 4.1),
    9: (-1.7, 3.2),   10: (1.7, 3.2),
    11: (-0.4, 3.5),  12: (0.4, 3.5),
    13: (-0.5, 2.2),  14: (0.5, 2.2),
    15: (-0.55, 0.9), 16: (0.55, 0.9),
}

# ── Player 2 (bottom court, facing up) ──
P2 = {k: (v[0], -v[1]) for k, v in P1.items()}

fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-8.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')

# ── Net ──
ax.plot([-2.5, 2.5], [0, 0], color='#424242', linewidth=3, solid_capstyle='round')
ax.text(2.8, 0, 'Net', fontsize=11, va='center', ha='left', color='#424242',
        fontweight='bold')

# ── Draw skeleton ──
def draw_player(pos, offset, player_label, label_y, label_color):
    # Draw limbs
    for group_name, group in LIMB_GROUPS.items():
        color = group["color"]
        for (i, j) in group["edges"]:
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            ax.plot([x0, x1], [y0, y1], color=color, lw=4.5,
                    solid_capstyle='round', zorder=2, alpha=0.85)

    # Draw joints
    for j in range(NUM_JOINTS):
        x, y = pos[j]
        # Outer circle
        ax.plot(x, y, 'o', color='white', markersize=16, zorder=3)
        ax.plot(x, y, 'o', color='#37474F', markersize=14, zorder=3,
                markeredgecolor='white', markeredgewidth=1.5)
        # Joint number
        ax.text(x, y, str(j + offset), fontsize=6, ha='center', va='center',
                color='white', fontweight='bold', zorder=4)

    # Player label
    ax.text(0, label_y, player_label, ha='center', fontsize=14,
            fontweight='bold', color=label_color)

draw_player(P1, 0, 'Player 1  (nodes 0–16)', 7.3, '#1565C0')
draw_player(P2, 17, 'Player 2  (nodes 17–33)', -7.3, '#C62828')

# ── L2 feature arrows ──
# Depth (along court)
ax.annotate('', xy=(3.3, -3.5), xytext=(3.3, 3.5),
            arrowprops=dict(arrowstyle='<->', color='#6A1B9A', lw=2))
ax.text(3.3, 0, 'depth\nto opp\n(signed)', ha='center', va='center', fontsize=8,
        color='#6A1B9A', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#F3E5F5', ec='#6A1B9A', lw=0.8))

# Lateral (across court)
ax.annotate('', xy=(1.8, -0.5), xytext=(-1.8, -0.5),
            arrowprops=dict(arrowstyle='<->', color='#00695C', lw=2))
ax.text(0, -1.1, 'lateral to opp (signed)', ha='center', fontsize=8,
        color='#00695C', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#E0F2F1', ec='#00695C', lw=0.8))

# ── "No inter-player edges" ──
ax.text(0, 0.4, 'No graph edges between players', ha='center', fontsize=8,
        color='#9E9E9E', style='italic')

# ── Joint name legend (right side) ──
legend_x = -5.2
legend_y_start = 6.5
ax.text(legend_x, legend_y_start + 0.8, 'Joint Index', fontsize=10, fontweight='bold',
        color='#212121')

for j in range(NUM_JOINTS):
    y = legend_y_start - j * 0.7
    ax.text(legend_x, y, f'{j}:', fontsize=8.5, ha='right', va='center',
            fontweight='bold', color='#37474F', family='monospace')
    ax.text(legend_x + 0.15, y, JOINT_NAMES[j], fontsize=8.5, ha='left',
            va='center', color='#616161')

# ── Adjacency info box (bottom-left) ──
info = ("3-subset adjacency (Yan et al.):\n"
        "  A\u2080 : Self-loops\n"
        "  A\u2081 : Centripetal  (toward root)\n"
        "  A\u2082 : Centrifugal (away from root)\n"
        "\n"
        "L2 court features (per player):\n"
        "  dist_to_net, dist_to_center,\n"
        "  depth_to_opp, lateral_to_opp")
ax.text(-5.3, -5.0, info, fontsize=7.5, va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#F5F5F5', ec='#BDBDBD', lw=0.8))

# ── Title ──
ax.set_title('Dual-Player Spatio-Temporal Graph\n'
             '34 nodes  (17 COCO keypoints \u00d7 2 players)',
             fontsize=15, fontweight='bold', pad=15, color='#212121')

# ── Limb group legend (bottom-right) ──
limb_patches = []
for name, grp in LIMB_GROUPS.items():
    if name in ("Shoulders", "L Leg", "R Leg", "L Arm", "R Arm", "Head"):
        limb_patches.append(plt.Line2D([0],[0], color=grp["color"], lw=4,
                                        label=name, solid_capstyle='round'))
ax.legend(handles=limb_patches, loc='lower right', fontsize=8.5,
          title='Limb segments', title_fontsize=9, framealpha=0.9,
          edgecolor='#BDBDBD', bbox_to_anchor=(1.0, 0.0))

plt.tight_layout()
plt.savefig('graph_structure.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("Saved to graph_structure.png")