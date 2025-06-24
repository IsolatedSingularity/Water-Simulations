# Hybrid PIC/FLIP Fluid Simulation

![Hybrid Simulation GIF](https://raw.githubusercontent.com/username/repo/main/Plots/hybrid_simulation.gif)
*Note: The image above is a placeholder and will be replaced by the actual output `hybrid_simulation.gif` from the `Plots/` directory.*

## Objective

This script implements a hybrid **Particle-in-Cell (PIC)** fluid simulation, specifically using a **FLIP (Fluid-Implicit-Particle)** method. This approach combines the strengths of Lagrangian (particle-based) and Eulerian (grid-based) methods to produce realistic and detailed fluid animations, particularly for dynamic effects like splashing.

The simulation generates a "dam break" scenario and saves the resulting animation as a GIF, showcasing the high-quality, energetic motion characteristic of FLIP-based solvers.

## Theoretical Background

Hybrid methods like PIC/FLIP are a cornerstone of modern computer graphics for fluid simulation. They offer a powerful compromise between the particle and grid methods seen in other scripts:

-   **Particles for Advection:** Fluid properties, primarily velocity, are stored on particles. This means the fluid's momentum is carried by the particles as they move, which completely eliminates the numerical diffusion (blurring) seen in purely grid-based advection. This preserves fine details and splashy dynamics.
-   **Grid for Pressure Solve:** To enforce incompressibility, particle velocities are transferred to a background grid. On this grid, a pressure projection step (identical to the one in the Stable Fluids method) is performed. This is far more efficient than calculating pressure from particle-particle interactions (as in SPH).
-   **FLIP Velocity Update:** Instead of just copying the new grid velocities back to the particles (the original PIC method), the FLIP method updates particle velocities by adding the *change* in grid velocity. This significantly reduces simulation noise and improves realism. A blending factor, `FLIP_ALPHA`, allows for mixing pure FLIP (`alpha=1.0`) with pure PIC (`alpha=0.0`) to control stability.

The workflow for a single time step is:
1.  **Particles-to-Grid (P2G):** Transfer particle velocities to the grid.
2.  **Grid Operations:** Apply forces (like gravity) and perform the pressure projection on the grid.
3.  **Grid-to-Particles (G2P):** Update particle velocities based on the change in the grid velocity field.
4.  **Particle Advection:** Move particles according to their new velocities.

## Code Functionality

### 1. Hybrid Solver (`HybridSolver`)
-   Manages the state of both the particles (`self.pos`, `self.vel`) and the background grid (`self.grid_u`, `self.grid_v`).
-   `_particles_to_grid()`: Implements the P2G step, using bilinear interpolation to transfer particle velocities to the four nearest grid nodes.
-   `_project()`: The standard Stable Fluids pressure projection method.
-   `_grid_to_particles()`: Implements the G2P step, calculating the velocity update using the FLIP/PIC blended method.
-   `_apply_boundary_conditions()`: Enforces solid walls with an elastic response, causing particles to bounce realistically.

```python
class HybridSolver:
    """A 2D PIC/FLIP-based hybrid fluid solver."""
    def step(self):
        """Perform one full step of the PIC/FLIP simulation."""
        self._particles_to_grid()
        
        old_grid_u = self.grid_u.copy()
        old_grid_v = self.grid_v.copy()
        
        # Grid Operations
        self.grid_v += GRAVITY[1] * DT
        self._project()
        
        self._grid_to_particles(old_grid_u, old_grid_v)
        
        # Advect particles and enforce boundaries
        self.pos += self.vel * DT
        self._apply_boundary_conditions()
```

### 2. Visualization
-   The main execution block sets up a `matplotlib` scatter plot.
-   The `update()` function calls the solver's `step()` method and then updates the positions and colors of the particles in the scatter plot.
-   Particle color is mapped to velocity magnitude, providing a visual cue for the fluid's speed.
-   The final animation is saved as `hybrid_simulation.gif` in the `Plots/` directory.

## How to Run

To generate the hybrid simulation animation, run the script from the command line in the project's root directory:

```bash
python Code/hybrid_simulation.py
```

The script will print its progress to the console, including the current frame number and particle count. Once complete, the animation will be saved as `hybrid_simulation.gif` in the `Plots/` directory. 