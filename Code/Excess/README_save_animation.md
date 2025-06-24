# Saved Fluid Simulation Animation

![Saved Simulation GIF](https://raw.githubusercontent.com/username/repo/main/Plots/saved_simulation.gif)
*Note: The image above is a placeholder and will be replaced by the actual output `saved_simulation.gif` from the `Plots/` directory.*

## Objective

This script uses the **Stable Fluids** solver to generate and save a non-interactive fluid animation as a GIF file. Instead of relying on user input, it programmatically creates a dynamic visual by having two fluid sources swirl around the center of the grid.

The goal is to produce a shareable, high-quality animation that demonstrates complex fluid motion, such as the mixing of fluids and the formation of vortices, in a controlled and repeatable way.

## Theoretical Background

The script is built upon the same **Stable Fluids** engine used in the real-time interactive version. However, instead of capturing mouse events, it generates its own input sources within the animation loop.

-   **Algorithmic Sources:** On each frame, the positions of two fluid sources are calculated using trigonometric functions (`sin`, `cos`) to define circular paths. The sources spiral inwards as the animation progresses.
-   **Velocity Injection:** The script calculates the velocity of each source based on its change in position from the previous frame. This velocity is injected into the fluid, creating a swirling motion and more dynamic interactions than simply adding density.
-   **Animation Saving:** The `matplotlib.animation.FuncAnimation` object is used not for interactive display but to run the simulation for a fixed number of frames. The final animation is then saved to a file using an appropriate writer (e.g., `pillow`).

## Code Functionality

### 1. Fluid Solver (`RealTimeFluidSolver`)
-   This script uses the same `RealTimeFluidSolver` class as the interactive simulation, leveraging its stable advection and pressure projection capabilities.

### 2. Algorithmic Fluid Generation
-   The core logic resides in the `update(frame)` function, which is called for each frame of the animation.
-   It calculates the `(x, y)` positions of two opposing fluid sources on spiraling paths.
-   It determines the velocity (`dx`, `dy`) of each source to create a directional force.
-   The `add_fluid_at_pos()` helper function is called to inject density and velocity into the solver's grid at the calculated source locations.

```python
def update(frame):
    global prev_pos1, prev_pos2
    
    # Define a swirling path for two sources
    t = frame / ANIMATION_FRAMES
    angle1 = 2 * np.pi * 4 * t
    radius1 = GRID_SIZE / 4 * (1 - 0.5 * t)
    
    # Calculate current positions and velocities
    pos1 = (int(GRID_SIZE/2 + radius1 * np.cos(angle1)), ...)
    dx1, dy1 = (0, 0) if prev_pos1 is None else (pos1[0] - prev_pos1[0], ...)

    # Add fluid from both sources
    add_fluid_at_pos(fluid_solver, pos1[0], pos1[1], dx1, dy1)
    # ... add fluid for second source ...
    
    # Step the simulation
    fluid_solver.step()
```

### 3. Saving the Animation
-   After the `FuncAnimation` object is created, the `ani.save()` method is called.
-   It specifies the output filename, the writer (`pillow`), and the desired frames per second (fps).
-   The script includes progress updates printed to the console as it generates the frames.

```python
if __name__ == "__main__":
    # ... setup ...
    ani = animation.FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=False)

    print(f"\\nSaving animation to {OUTPUT_FILENAME}...")
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=30)
```

## How to Run

To generate the animation, run the script from the command line in the project's root directory:

```bash
python Code/save_animation.py
```

The script will print its progress to the console. Once complete, the final animation will be saved as `saved_simulation.gif` in the `Plots/` directory. 