# Real-Time 2D Water Simulation

![Real-Time Simulation GIF](https://raw.githubusercontent.com/username/repo/main/Plots/realtime_simulation.gif)
*Note: The image above is a placeholder. This script runs an interactive simulation and does not save a file by default.*

## Objective

This script implements a real-time, interactive 2D fluid simulation using the **Stable Fluids** algorithm. It opens a window where the user can "paint" fluid density and apply forces by clicking and dragging the mouse. The primary goal is to provide an intuitive, hands-on demonstration of fluid dynamics principles like advection and pressure projection.

The simulation also includes a real-time Frames Per Second (FPS) counter to benchmark the performance of the solver.

## Theoretical Background

This simulation is built on the **Stable Fluids** method developed by Jos Stam. Unlike the non-interactive scripts, the focus here is on speed and stability to allow for user interaction. The key components are:

-   **Eulerian Grid:** The simulation space is a grid where velocity and density values are stored.
-   **Stable Advection:** A semi-Lagrangian advection scheme is used to move fluid properties through the velocity field. This method is unconditionally stable, preventing the simulation from crashing with large mouse movements.
-   **Pressure Projection:** To maintain incompressibility (i.e., the fluid doesn't spontaneously compress or expand), a projection step is performed. This involves solving a Poisson equation for pressure and subtracting the pressure gradient from the velocity field, making it divergence-free.
-   **User Interaction:** Mouse movements are translated into forces and density sources that are added to the grid, allowing the user to directly influence the fluid's behavior.

## Code Functionality

### 1. Fluid Solver (`RealTimeFluidSolver`)
-   Manages the state of the simulation grid, including velocity fields (`u`, `v`) and a density field.
-   Contains the core implementation of the Stable Fluids algorithm: `advect`, `diffuse` (though not heavily used here), and `project`.
-   The `step()` method advances the simulation by one time step, sequentially calling the velocity and density update steps.

```python
class RealTimeFluidSolver:
    """A real-time fluid dynamics solver using the Stable Fluids method."""
    def step(self):
        """Perform a full simulation step."""
        self.velocity_step()
        self.density_step()
        # Clear source arrays for the next frame
        self.u_prev.fill(0)
        self.v_prev.fill(0)
        self.density_prev.fill(0)
```

### 2. User Interaction and Animation (`FluidAnimation`)
-   Handles creating the `matplotlib` animation window.
-   Connects mouse events (`button_press_event`, `button_release_event`, `motion_notify_event`) to simulation logic.
-   When the user clicks and drags, it translates the mouse's position and velocity into density and force sources, which are added to the solver's "previous" state arrays (`density_prev`, `u_prev`, `v_prev`).
-   The `update()` function, called on each frame, steps the solver and redraws the density field.
-   Calculates and displays the current FPS.

```python
class FluidAnimation:
    """Handles user interaction and matplotlib animation."""
    def on_mouse_move(self, event):
        if self.is_mouse_down and event.inaxes == self.ax:
            # ... calculate mouse delta ...
            # Add density and force to the solver's source arrays
            self.solver.density_prev[...] += DENSITY_AMOUNT * mask
            self.solver.u_prev[...] += FORCE_AMOUNT * dx * mask
            self.solver.v_prev[...] += FORCE_AMOUNT * dy * mask
```

## How to Run

To run the interactive simulation, execute the script from the command line in the project's root directory:

```bash
python Code/realtime_simulation.py
```

A window will appear displaying the fluid simulation. **Click and drag your mouse** inside the window to add fluid and create ripples. The FPS counter in the top-left corner will show the current performance. Close the window to end the simulation. 