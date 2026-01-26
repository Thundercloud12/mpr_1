# visualize_1d.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Optimizer1DVisualizer:
    def __init__(self, history_positions, obj_fn, bounds, title="Optimization"):

        self.history = history_positions
        self.obj_fn = obj_fn
        self.bounds = bounds
        self.title = title

        # Track best value over time
        self.best_history = []

        # Prepare function curve
        self.x_curve = np.linspace(bounds[0][0], bounds[0][1], 400)
        self.y_curve = np.array([obj_fn([x]) for x in self.x_curve])

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.plot(self.x_curve, self.y_curve, color="black", linewidth=2)

        self.scatter = self.ax.scatter([], [], color="red", s=60)

        self.ax.set_xlim(bounds[0])
        self.ax.set_ylim(min(self.y_curve), max(self.y_curve) * 1.1)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")

        # Text box for iteration + best value
        self.info_text = self.ax.text(
            0.02, 0.95, "",
            transform=self.ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    # --------------------------------------------------
    def update(self, frame):
        positions = self.history[frame]

        xs = [p[0] for p in positions]
        ys = [self.obj_fn([x]) for x in xs]

        # Update scatter
        self.scatter.set_offsets(np.c_[xs, ys])

        # Track best value so far
        current_best = min(ys)
        if frame == 0:
            self.best_history.append(current_best)
        else:
            self.best_history.append(min(self.best_history[-1], current_best))

        # Update title + info
        self.ax.set_title(self.title)
        self.info_text.set_text(
            f"Iteration: {frame}\nBest f(x): {self.best_history[-1]:.6e}"
        )

        return self.scatter, self.info_text

    # --------------------------------------------------
    def animate(self, interval=600):
        """
        interval (ms):
        100  = fast
        300  = readable (recommended)
        600+ = slow motion
        """
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.history),
            interval=interval,
            blit=True
        )
        plt.show()
