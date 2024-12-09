import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_box_vertices(dimensions, center):
    """Create vertices for a box given dimensions and center."""
    l, w, h = dimensions
    x, y, z = center
    
    vertices = np.array([
        [x-l/2, y-w/2, z-h/2], [x+l/2, y-w/2, z-h/2],
        [x+l/2, y+w/2, z-h/2], [x-l/2, y+w/2, z-h/2],
        [x-l/2, y-w/2, z+h/2], [x+l/2, y-w/2, z+h/2],
        [x+l/2, y+w/2, z+h/2], [x-l/2, y+w/2, z+h/2]
    ])
    return vertices

def plot_box(ax, vertices, color='b', alpha=0.2):
    """Plot a 3D box given its vertices."""
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    collection = Poly3DCollection(faces, alpha=alpha)
    collection.set_facecolor(color)
    ax.add_collection3d(collection)

def viz_3d():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Table (brown)
    table_vertices = create_box_vertices([2.0, 2.0, 0.05], [0, 0, 0])
    plot_box(ax, table_vertices, color='brown', alpha=0.3)

    # Red box
    red_box_vertices = create_box_vertices([0.05, 0.05, 0.118], [-0.5, 0.5, 0.059])
    plot_box(ax, red_box_vertices, color='red', alpha=0.3)

    # Cyan box
    cyan_box_vertices = create_box_vertices([0.05, 0.05, 0.070], [0, 0.5, 0.035])
    plot_box(ax, cyan_box_vertices, color='cyan', alpha=0.3)

    # Blue box
    blue_box_vertices = create_box_vertices([0.05, 0.05, 0.106], [0.25, 0.5, 0.053])
    plot_box(ax, blue_box_vertices, color='blue', alpha=0.3)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set view angle
    ax.view_init(elev=30, azim=45)

    # Set axis limits
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-0.5, 0.5])

    plt.show()

def plot_bbox(ax, bbox, color='b', alpha=0.3, label=None):
    """
    Plot a bounding box given coordinates (x0, y0, x1, y1).
    Args:
        bbox: tuple of (x0, y0, x1, y1) where:
            (x0, y0) is the top-left corner
            (x1, y1) is the bottom-right corner
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    
    rect = patches.Rectangle(
        (x0, y0), width, height,
        linewidth=2,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        label=label
    )
    ax.add_patch(rect)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 10))

# Example bounding boxes (x0, y0, x1, y1)
bboxes = {
    'red_box': (70, 150, 90, 170),
    'cyan_box': (150, 150, 170, 170),
    'blue_box': (70, 170, 90, 190),
}

# Plot each bounding box with different colors
colors = {
    'table': 'brown',
    'red_box': 'red',
    'cyan_box': 'cyan',
    'blue_box': 'blue',
}

for obj_name, bbox in bboxes.items():
    plot_bbox(ax, bbox, color=colors[obj_name], label=obj_name)
    
padding = 50
all_x = [coord for bbox in bboxes.values() for coord in [bbox[0], bbox[2]]]
all_y = [coord for bbox in bboxes.values() for coord in [bbox[1], bbox[3]]]
ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
ax.set_aspect('equal')

plt.show()