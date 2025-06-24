"""
Generates and Saves a Fluid Simulation Animation

This script uses the Stable Fluids solver to create a dynamic fluid animation
and saves it as a GIF file. It programmatically simulates user input by having
two sources of fluid swirl around the center of the grid, creating an
interesting visual pattern.

This is a non-interactive script designed to produce a shareable animation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# --- Simulation Parameters ---
GRID_SIZE = 128       # Resolution of the simulation grid
DT = 0.1              # Time step
VISCOSITY = 0.000001  # Fluid viscosity
DENSITY_AMOUNT = 200.0 # Density added per frame
ANIMATION_FRAMES = 240 # Total frames in the animation
FORCE_SCALE = 5.0    # Scaling factor for force from motion

# --- Visualization Parameters ---
PLOT_DIR = 'Plots'
OUTPUT_FILENAME = os.path.join(PLOT_DIR, 'saved_simulation.gif')
SEQUENTIAL_CMAP = sns.color_palette("mako", as_cmap=True)


class RealTimeFluidSolver:
    """
    A real-time fluid dynamics solver using the Stable Fluids method.
    (This is the same solver as in the interactive version).
    """
    def __init__(self, size):
        self.size = size
        self.dt = DT
        self.visc = VISCOSITY

        # Velocity and density fields
        self.u = np.zeros((size, size))
        self.v = np.zeros((size, size))
        self.density = np.zeros((size, size))
        
        # Previous state fields
        self.u_prev = np.zeros((size, size))
        self.v_prev = np.zeros((size, size))
        self.density_prev = np.zeros((size, size))

    def add_source(self, x, s, dt):
        x += dt * s

    def set_boundaries(self, b, x):
        """Enforce boundary conditions."""
        if b == 1: # Reflective for u
            x[0, :] = -x[1, :]
            x[-1, :] = -x[-2, :]
            x[:, 0] = x[:, 1]
            x[:, -1] = x[:, -2]
        elif b == 2: # Reflective for v
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :]
            x[:, 0] = -x[:, 1]
            x[:, -1] = -x[:, -2]
        else: # b == 0, for density
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :]
            x[:, 0] = x[:, 1]
            x[:, -1] = x[:, -2]
        
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    def linear_solve(self, b, x, x0, a, c, iters=20):
        c_recip = 1.0 / c
        for _ in range(iters):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) * c_recip
            self.set_boundaries(b, x)

    def diffuse(self, b, x, x0, rate):
        a = self.dt * rate * (self.size - 2)**2
        self.linear_solve(b, x, x0, a, 1 + 4 * a)

    def advect(self, b, d, d0, u, v):
        dt0 = self.dt * (self.size - 2)
        i, j = np.meshgrid(np.arange(self.size), np.arange(self.size))
        x = np.clip(i - dt0 * u, 0.5, self.size - 1.5)
        y = np.clip(j - dt0 * v, 0.5, self.size - 1.5)
        
        i0, j0 = np.floor(x).astype(int), np.floor(y).astype(int)
        i1, j1 = i0 + 1, j0 + 1
        
        s1, t1 = x - i0, y - j0
        s0, t0 = 1 - s1, 1 - t1
        
        d[1:-1, 1:-1] = (s0[1:-1, 1:-1] * (t0[1:-1, 1:-1] * d0[j0[1:-1, 1:-1], i0[1:-1, 1:-1]] + t1[1:-1, 1:-1] * d0[j1[1:-1, 1:-1], i0[1:-1, 1:-1]]) +
                         s1[1:-1, 1:-1] * (t0[1:-1, 1:-1] * d0[j0[1:-1, 1:-1], i1[1:-1, 1:-1]] + t1[1:-1, 1:-1] * d0[j1[1:-1, 1:-1], i1[1:-1, 1:-1]]))
        self.set_boundaries(b, d)

    def project(self, u, v, p, div):
        h = 1.0 / (self.size - 2)
        div[1:-1, 1:-1] = -0.5 * h * (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1])
        p.fill(0)
        self.set_boundaries(0, div)
        self.set_boundaries(0, p)
        self.linear_solve(0, p, div, 1, 4)
        
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        self.set_boundaries(1, u)
        self.set_boundaries(2, v)
    
    def velocity_step(self):
        """Performs one step of the velocity simulation."""
        self.add_source(self.u, self.u_prev, self.dt)
        self.add_source(self.v, self.v_prev, self.dt)
        self.u_prev, self.u = self.u, self.u_prev
        self.v_prev, self.v = self.v, self.v_prev
        self.diffuse(1, self.u, self.u_prev, self.visc)
        self.diffuse(2, self.v, self.v_prev, self.visc)
        self.project(self.u, self.v, self.u_prev, self.v_prev)
        self.u_prev, self.u = self.u, self.u_prev
        self.v_prev, self.v = self.v, self.v_prev
        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev)
        self.advect(2, self.v, self.v_prev, self.u_prev, self.v_prev)
        self.project(self.u, self.v, self.u_prev, self.v_prev)

    def density_step(self):
        """Performs one step of the density simulation."""
        self.add_source(self.density, self.density_prev, self.dt)
        self.density_prev, self.density = self.density, self.density_prev
        self.diffuse(0, self.density, self.density_prev, 0) # No diffusion for density
        self.density_prev, self.density = self.density, self.density_prev
        self.advect(0, self.density, self.density_prev, self.u, self.v)
    
    def step(self):
        """Perform a full simulation step."""
        self.velocity_step()
        self.density_step()
        
        # Reset source arrays
        self.u_prev.fill(0)
        self.v_prev.fill(0)
        self.density_prev.fill(0)

def add_fluid_at_pos(solver, x, y, dx, dy):
    """Adds density and force to the solver at a given position."""
    radius = 4
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = xx**2 + yy**2 <= radius**2
    
    # Clamp coordinates to be within bounds
    x_min, x_max = max(0, x-radius), min(solver.size, x+radius+1)
    y_min, y_max = max(0, y-radius), min(solver.size, y+radius+1)
    
    mask_x_min = max(0, radius - x)
    mask_x_max = min(mask.shape[1], radius - x + solver.size)
    mask_y_min = max(0, radius - y)
    mask_y_max = min(mask.shape[0], radius - y + solver.size)

    # Add sources
    solver.density_prev[y_min:y_max, x_min:x_max] += DENSITY_AMOUNT * mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
    solver.u_prev[y_min:y_max, x_min:x_max] += FORCE_SCALE * dx * mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
    solver.v_prev[y_min:y_max, x_min:x_max] += FORCE_SCALE * dy * mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]

if __name__ == "__main__":
    print("--- Generating Fluid Simulation Animation ---")
    
    # Initialize solver and plotting
    fluid_solver = RealTimeFluidSolver(GRID_SIZE)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(fluid_solver.density, cmap=SEQUENTIAL_CMAP, origin='lower', vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Saved Fluid Simulation")
    
    # Store previous positions to calculate velocity for forces
    prev_pos1, prev_pos2 = None, None

    def update(frame):
        global prev_pos1, prev_pos2
        
        # Define a swirling path for two sources
        t = frame / ANIMATION_FRAMES
        angle1 = 2 * np.pi * 4 * t  # 4 rotations
        radius1 = GRID_SIZE / 4 * (1 - 0.5 * t)
        
        angle2 = 2 * np.pi * 4 * t + np.pi # Opposite side
        radius2 = GRID_SIZE / 4 * (1 - 0.5 * t)
        
        # Current positions
        pos1 = (int(GRID_SIZE/2 + radius1 * np.cos(angle1)), int(GRID_SIZE/2 + radius1 * np.sin(angle1)))
        pos2 = (int(GRID_SIZE/2 + radius2 * np.cos(angle2)), int(GRID_SIZE/2 + radius2 * np.sin(angle2)))

        # Calculate velocity to apply force
        dx1, dy1 = (0, 0) if prev_pos1 is None else (pos1[0] - prev_pos1[0], pos1[1] - prev_pos1[1])
        dx2, dy2 = (0, 0) if prev_pos2 is None else (pos2[0] - prev_pos2[0], pos2[1] - prev_pos2[1])

        # Add fluid from both sources
        add_fluid_at_pos(fluid_solver, pos1[0], pos1[1], dx1, dy1)
        add_fluid_at_pos(fluid_solver, pos2[0], pos2[1], dx2, dy2)

        # Update previous positions
        prev_pos1, prev_pos2 = pos1, pos2
        
        # Step the simulation
        fluid_solver.step()
        
        # Update the image
        im.set_data(fluid_solver.density)
        im.set_clim(0, np.maximum(1, np.max(fluid_solver.density)))
        
        # Progress update
        print(f"  Generating frame {frame+1}/{ANIMATION_FRAMES}", end='\\r')
        
        return [im]

    # Create the animation
    print("Initializing animation...")
    # Using blit=False can be more robust across different environments
    ani = animation.FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=False)

    # Save the animation as a GIF
    print(f"\\nSaving animation to {OUTPUT_FILENAME}...")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    try:
        # Check for Pillow writer availability
        if 'pillow' not in animation.writers.list():
            raise RuntimeError("Pillow writer not available. Please install Pillow: pip install Pillow")
        
        ani.save(OUTPUT_FILENAME, writer='pillow', fps=30)
        plt.close(fig)
        print(f"--- Animation saved successfully! ---")
    except Exception as e:
        print(f"\\nError saving animation: {e}")
        print("Please ensure you have necessary writers like Pillow installed (`pip install Pillow`).") 