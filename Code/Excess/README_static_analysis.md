# Static Analysis of 2D Water Simulation Models

![Static Comparison Plot](https://raw.githubusercontent.com/username/repo/main/Plots/static_comparison.png)
*Note: The image above is a placeholder and will be replaced by the actual output `static_comparison.png` from the `Plots/` directory.*

## Objective

This script performs a static, comparative analysis of two different 2D water simulation methods: **Smoothed Particle Hydrodynamics (SPH)** and a grid-based **Stable Fluids** solver. It generates a single, high-quality plot comparing key fluid properties (density, pressure, and velocity divergence) from both methods for a standardized "dam break" scenario.

The primary goal is to provide a clear, side-by-side visual assessment of the qualitative differences between these two popular fluid simulation techniques.

## Theoretical Background

### Smoothed Particle Hydrodynamics (SPH)
SPH is a Lagrangian method where the fluid is represented by a set of particles, each carrying properties like mass, position, velocity, and pressure. Fluid properties at any point are calculated by summing the contributions of nearby particles, weighted by a smooth kernel function (e.g., Poly6 or Spiky). Forces (pressure, viscosity) are computed based on particle interactions, and their positions are updated over time. SPH excels at modeling complex, free-surface phenomena like splashing.

### Stable Fluids
The Stable Fluids method, developed by Jos Stam, is an Eulerian (grid-based) approach. The simulation space is divided into a grid, and the fluid's velocity field is evolved over time. The core of the method is an advection-projection sequence:
1.  **Advect:** The velocity field moves itself.
2.  **Project:** The velocity field is forced to be divergence-free, ensuring the fluid is incompressible. This step involves solving a Poisson equation for pressure.

This method is unconditionally stable, meaning it does not blow up with large time steps, making it ideal for real-time applications.

## Code Functionality

### 1. SPH Solver (`SPHSolver`)
-   Initializes particles in a "dam" configuration on the left side of the domain.
-   Implements Poly6 and Spiky kernel functions for density and pressure calculations.
-   Computes pressure and viscosity forces between particles.
-   Integrates particle positions using Verlet integration.
-   Interpolates particle data onto a grid for direct comparison with the Stable Fluids method.

```python
class SPHSolver:
    """A simple 2D SPH solver for the dam break scenario."""
    def step(self):
        """Perform one step of the SPH simulation."""
        self._update_fluid_properties()
        self._compute_forces()
        self._integrate()
```

### 2. Stable Fluids Solver (`StableFluidsSolver`)
-   Initializes a density grid in a "dam" configuration.
-   Uses a semi-Lagrangian advection scheme to move density and velocity.
-   Implements a projection step with a Jacobi iterative solver to enforce incompressibility and calculate pressure.
-   Calculates velocity divergence from the final velocity field.

```python
class StableFluidsSolver:
    """A simple 2D grid-based Stable Fluids solver."""
    def step(self):
        """Perform one step of the Stable Fluids simulation."""
        self.v += GRAVITY[1] * DT  # Add gravity
        # ... Advection and Projection steps ...
        self.density = self._advect(self.density, self.u, self.v)
        return pressure
```

### 3. Visualization (`create_comparison_plot`)
-   Runs both simulations for a fixed number of steps.
-   Generates a 2x3 subplot comparing Density, Pressure, and Velocity Divergence for both SPH and Stable Fluids.
-   Uses project-approved `seaborn` color palettes (`"mako"` and `cubehelix_palette`) for clear and consistent visuals.
-   Saves the final plot as `static_comparison.png` in the `Plots/` directory.

```python
def create_comparison_plot(sph_data, sf_data):
    """Creates a multi-panel plot comparing SPH and Stable Fluids."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    # ... plotting logic ...
    fig.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
```

## How to Run

To generate the static comparison image, run the script from the command line in the project's root directory:

```bash
python Code/static_analysis.py
```

The script will create the `Plots/` directory if it doesn't exist and save the output image `static_comparison.png` inside it. 