"""
Realtime interactive Stable Fluids scene.

Click/drag to add density and force. Press 'v' to toggle vorticity view.
Dark theme applied. Imports solver from watersim.solvers.stableFluids.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import PALETTES, addFooter, applyDarkTheme

GRID_SIZE = 128
DT = 0.1
VISCOSITY = 1e-6
FORCE_AMOUNT = 5.0
DENSITY_AMOUNT = 100.0
ANIMATION_INTERVAL = 1  # ms


class FluidAnimation:
    """Interactive Stable Fluids visualization. Mouse adds fluid and force."""

    BRUSH_RADIUS = 3

    def __init__(self, solver: StableFluidsSolver) -> None:
        self.solver = solver
        self.showVorticity = False

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
        addFooter(self.fig)

        self.im = self.ax.imshow(
            solver.density,
            origin="lower",
            cmap=PALETTES["sequential"],
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        self.ax.set_title(
            "Stable Fluids  ·  Interactive  ·  [v] toggle vorticity",
            color="#e6edf3",
            pad=8,
        )
        self.ax.axis("off")

        self.isMouseDown = False
        self.prevMousePos: tuple[float, float] | None = None
        self.frameCount = 0

        self.fig.canvas.mpl_connect("button_press_event", self._onMouseDown)
        self.fig.canvas.mpl_connect("button_release_event", self._onMouseUp)
        self.fig.canvas.mpl_connect("motion_notify_event", self._onMouseMove)
        self.fig.canvas.mpl_connect("key_press_event", self._onKeyPress)

    def _onMouseDown(self, event: object) -> None:
        self.isMouseDown = True
        if hasattr(event, "xdata") and event.xdata is not None:
            self.prevMousePos = (event.xdata, event.ydata)

    def _onMouseUp(self, event: object) -> None:
        self.isMouseDown = False
        self.prevMousePos = None

    def _onMouseMove(self, event: object) -> None:
        if not self.isMouseDown:
            return
        if not hasattr(event, "xdata") or event.xdata is None:
            return
        n = self.solver.size
        x = int(np.clip(event.xdata, 0, n - 1))
        y = int(np.clip(event.ydata, 0, n - 1))
        dx = dy = 0.0
        if self.prevMousePos is not None:
            dx = (event.xdata - self.prevMousePos[0]) * FORCE_AMOUNT
            dy = (event.ydata - self.prevMousePos[1]) * FORCE_AMOUNT
        r = self.BRUSH_RADIUS
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di**2 + dj**2 <= r**2:
                    xi = int(np.clip(x + di, 0, n - 1))
                    yj = int(np.clip(y + dj, 0, n - 1))
                    self.solver.densityPrev[xi, yj] += DENSITY_AMOUNT
                    self.solver.uPrev[xi, yj] += dx
                    self.solver.vPrev[xi, yj] += dy
        self.prevMousePos = (event.xdata, event.ydata)

    def _onKeyPress(self, event: object) -> None:
        if hasattr(event, "key") and event.key == "v":
            self.showVorticity = not self.showVorticity
            cmap = PALETTES["vorticity"] if self.showVorticity else PALETTES["sequential"]
            self.im.set_cmap(cmap)

    def _update(self, frame: int) -> list:
        self.solver.step()
        self.frameCount += 1

        if self.showVorticity:
            field = self.solver.getVorticity()
            vMax = float(np.percentile(np.abs(field + 1e-9), 99))
            self.im.set_data(field)
            self.im.set_clim(-max(vMax, 0.01), max(vMax, 0.01))
        else:
            self.im.set_data(self.solver.density)
            self.im.set_clim(0, max(float(self.solver.density.max()), 1e-6))

        return [self.im]

    def run(self) -> None:
        """Start the interactive animation loop."""
        self._anim = FuncAnimation(
            self.fig,
            self._update,
            interval=ANIMATION_INTERVAL,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()


def runRealtime() -> None:
    """Launch the interactive realtime fluid simulation."""
    applyDarkTheme()
    solver = StableFluidsSolver(size=GRID_SIZE, dt=DT, visc=VISCOSITY)
    anim = FluidAnimation(solver)
    anim.run()
