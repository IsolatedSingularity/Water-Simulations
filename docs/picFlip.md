# PIC/FLIP Hybrid Solver

`watersim/solvers/picFlip.py` implements a hybrid Particle-in-Cell / Fluid-Implicit-Particle solver for the dam-break scene.

---

## Overview

PIC/FLIP combines the strengths of Lagrangian particles and Eulerian grids:

- Particles carry velocity: no numerical diffusion from advection.
- Grid handles pressure: incompressibility enforced cheaply via projection.
- FLIP update: particles receive the *change* in grid velocity rather than the raw value, preserving high-frequency detail.

---

## Domain

| Parameter | Value | Description |
|---|---|---|
| `gridWidth` | 192 | Grid columns |
| `gridHeight` | 96 | Grid rows (2:1 aspect ratio) |
| `nParticlesPerRow` | 50 | Particles per axis of initial dam block |
| `FLIP_ALPHA` | 0.95 | Blend: 1.0 = pure FLIP, 0.0 = pure PIC |
| `DT` | 0.04 | Time step |
| `GRAVITY` | (0, -19.6) | Body force |

Initial dam block occupies the left 30% width x 70% height of the domain.

---

## Step Pipeline

```
step()
  _particlesToGrid()     # P2G: scatter particle velocities to staggered MAC grid
  gridV += DT * gravity  # Apply gravity on grid
  _project()             # 80-iteration Jacobi pressure projection
  _gridToParticles()     # G2P: FLIP/PIC blended velocity update
  vel = clip(vel, -60, 60)  # stability clamp
  pos += DT * vel        # advect particles
  _applyBoundaries()     # elastic wall response (coefficient -0.5)
```

---

## P2G: Particle-to-Grid Transfer

Particle velocities are scattered to a staggered MAC (Marker-and-Cell) grid using bilinear weights. `np.add.at` accumulates contributions without Python loops:

```
u-component: offset (0.0, 0.5) in cell space
v-component: offset (0.5, 0.0) in cell space
```

Final grid velocity: weighted average of contributing particles.

---

## G2P: Grid-to-Particle Velocity Update

For each particle, interpolate both new and old grid velocities bilinearly. Apply FLIP/PIC blend:

$$\mathbf{v}_p^{n+1} = \alpha(\mathbf{v}_p^n + \Delta\mathbf{v}_{\text{grid}}) + (1-\alpha)\,\mathbf{v}_{\text{grid}}^{n+1}$$

where $\Delta\mathbf{v}_{\text{grid}} = \mathbf{v}_{\text{grid}}^{n+1} - \mathbf{v}_{\text{grid}}^n$ (the FLIP delta).

---

## Visualization

Particles are rendered as a scatter plot, colored by speed magnitude with a log-stretch colormap (`crest` palette). A 3-frame motion-blur trail is rendered at decaying alpha to highlight the fast splash front.

---

## See Also

- [theory.md](theory.md) for the projection method and PIC/FLIP blend equation
- Zhu & Bridson (2005): *Animating Sand as a Fluid* (PIC/FLIP reference)
