---
name: fluid-simulation
description: >
  Build interactive 2D fluid simulations using particle-based methods (SPH, PIC/FLIP).
  Use this skill whenever the user asks to create a fluid simulation, water simulation,
  liquid animation, particle-based physics, sloshing tank, dam break, wave simulation,
  or any real-time fluid dynamics visualization. Also trigger when the user mentions
  SPH (Smoothed Particle Hydrodynamics), PIC (Particle In Cell), FLIP (Fluid Implicit Particle),
  Navier-Stokes solver, pressure projection, incompressible flow, staggered grid,
  Eulerian/Lagrangian fluid methods, or asks for interactive physics demos involving liquids.
  Even if the user just says "make water move" or "simulate a fluid" or "particle physics demo",
  use this skill. Covers both the mathematical foundations and practical implementation
  in JavaScript/HTML Canvas or React.
---

# Fluid Simulation Skill

Create real-time, interactive 2D fluid simulations in the browser using particle-based methods. This skill covers two primary approaches: **SPH (Smoothed Particle Hydrodynamics)** and **PIC/FLIP (Particle-In-Cell / Fluid-Implicit-Particle)**. Both produce visually compelling fluid behavior suitable for demos, games, educational tools, and creative coding.

---

## Table of Contents

1. [When to Use Which Method](#method-selection)
2. [Core Physics Concepts](#core-physics)
3. [SPH Method — Theory & Implementation](#sph-method)
4. [PIC/FLIP Method — Theory & Implementation](#pic-flip-method)
5. [Implementation Checklist](#implementation-checklist)
6. [Rendering Techniques](#rendering)
7. [Performance Optimization](#performance)
8. [Common Pitfalls & Fixes](#pitfalls)
9. [Reference Resources](#references)

---

## 1. Method Selection <a name="method-selection"></a>

| Criteria | SPH | PIC/FLIP |
|---|---|---|
| **Best for** | Small-scale splashy effects, droplets, surface tension | Large bodies of water, tanks, waves, dam breaks |
| **Grid required?** | No (purely particle-based, meshless) | Yes (hybrid: particles + Eulerian grid) |
| **Incompressibility** | Approximate (equation of state or pressure solver) | Exact (pressure projection on grid) |
| **Visual character** | Organic, blobby, splashy | Smooth water surface with fine detail |
| **Complexity** | Moderate | Higher (grid solver + particle transfer) |
| **Industry usage** | Engineering CFD, real-time games | Film VFX (Houdini, WETA, ILM) |

**Decision rule:** If the user wants a contained body of water (tank, pool, dam break), use **PIC/FLIP**. If they want splashes, droplets, or a simpler particle effect, use **SPH**. If unsure, default to **PIC/FLIP** as it produces more visually stable results for general fluid scenes.

---

## 2. Core Physics Concepts <a name="core-physics"></a>

### The Navier-Stokes Equations (Simplified, Inviscid, Incompressible)

All fluid simulation methods approximate these two governing equations:

**Momentum equation:**
```
∂v/∂t + (v · ∇)v = -∇p/ρ + g
```
- `v` = velocity field
- `p` = pressure
- `ρ` = density
- `g` = gravity (external forces)

**Incompressibility constraint:**
```
∇ · v = 0
```
This says the divergence of the velocity field must be zero — fluid volume is conserved. No region should have net inflow or outflow.

### Eulerian vs. Lagrangian

- **Eulerian:** Fixed grid cells. Fluid flows *through* cells. Good for pressure solves but suffers from numerical diffusion during advection.
- **Lagrangian:** Particles move *with* the fluid. Naturally mass-conservative. No advection diffusion. But hard to compute spatial derivatives (pressure, viscosity).
- **Hybrid (PIC/FLIP):** Particles carry velocity and position (Lagrangian). Grid is used for the pressure solve (Eulerian). Best of both worlds.

### Staggered Grid (MAC Grid)

Used in PIC/FLIP. Instead of storing all velocity components at cell centers:
- **u-velocities** (horizontal) are stored on **vertical cell faces**
- **v-velocities** (vertical) are stored on **horizontal cell faces**
- **Pressure / density** are stored at **cell centers**

This prevents checkerboard pressure artifacts and makes the divergence computation clean:
```
divergence(i,j) = u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)
```

### Kernel Functions (SPH-specific)

A kernel `W(r, h)` is a bell-shaped weighting function that determines how nearby particles influence each other:
- `r` = distance between particles
- `h` = kernel radius (support radius)
- Closer particles get higher weight
- As `h → 0`, the kernel approaches a Dirac delta (exact pointwise value)
- Common choices: Cubic Spline, Wendland C2, Poly6, Spiky

Key property: `∫ W(r,h) dr = 1` (partition of unity)

---

## 3. SPH Method — Theory & Implementation <a name="sph-method"></a>

### Core Idea

The fluid is represented entirely by particles. Each particle carries mass, position, velocity, and density. Physical quantities at any point are reconstructed by a weighted sum over neighboring particles using kernel functions.

### Algorithm (Per Timestep)

```
1. NEIGHBOR SEARCH
   - Build spatial hash grid (cell size = kernel radius h)
   - For each particle, find all neighbors within radius h

2. COMPUTE DENSITY
   - For each particle i:
     ρ_i = Σ_j m_j * W(|x_i - x_j|, h)

3. COMPUTE PRESSURE
   - Using equation of state (Tait equation):
     p_i = k * ((ρ_i / ρ_0)^γ - 1)
   - k = stiffness constant, ρ_0 = rest density, γ = 7 (for water)

4. COMPUTE FORCES
   a) Pressure force (pushes particles from high to low pressure):
      F_pressure_i = -Σ_j m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W(|x_i - x_j|, h)

   b) Viscosity force (smooths velocity differences):
      F_viscosity_i = μ * Σ_j m_j * (v_j - v_i)/ρ_j * ∇²W(|x_i - x_j|, h)

   c) External forces:
      F_gravity = m_i * g

5. INTEGRATE
   - v_i += dt * (F_pressure + F_viscosity + F_gravity) / m_i
   - x_i += dt * v_i

6. BOUNDARY HANDLING
   - Reflect particles off walls
   - Apply friction damping at boundaries
```

### Key Parameters (SPH)

| Parameter | Typical Value | Notes |
|---|---|---|
| `h` (kernel radius) | 16–30 px (screen space) | Larger = smoother, slower |
| `ρ_0` (rest density) | 1000 | Water density |
| `k` (stiffness) | 50–500 | Higher = more incompressible, needs smaller dt |
| `μ` (viscosity) | 0.1–1.0 | Higher = honey-like |
| `dt` (timestep) | 0.001–0.01 | Must satisfy CFL condition |
| `γ` (Tait exponent) | 7 | Standard for water |
| Particles per scene | 500–5000 | For real-time browser performance |

### Neighbor Search: Spatial Hashing

This is the performance bottleneck. Naive O(n²) is too slow. Use spatial hashing:

```javascript
// Cell size = kernel radius h
function hashPosition(x, y, h) {
    const xi = Math.floor(x / h);
    const yi = Math.floor(y / h);
    return xi + yi * GRID_WIDTH;  // or use a hash map
}

// Build grid each frame
function buildSpatialGrid(particles, h) {
    const grid = new Map();
    for (const p of particles) {
        const key = hashPosition(p.x, p.y, h);
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(p);
    }
    return grid;
}

// Query: check 9 neighboring cells (3x3 in 2D)
function getNeighbors(particle, grid, h) {
    const neighbors = [];
    const cx = Math.floor(particle.x / h);
    const cy = Math.floor(particle.y / h);
    for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
            const key = (cx + dx) + (cy + dy) * GRID_WIDTH;
            const cell = grid.get(key);
            if (cell) {
                for (const p of cell) {
                    const dist = distance(particle, p);
                    if (dist < h) neighbors.push({ particle: p, dist });
                }
            }
        }
    }
    return neighbors;
}
```

### Kernel Functions (JavaScript)

```javascript
// Poly6 kernel (good for density estimation)
function poly6(r, h) {
    if (r >= h) return 0;
    const coeff = 4.0 / (Math.PI * Math.pow(h, 8));
    const diff = h * h - r * r;
    return coeff * diff * diff * diff;
}

// Spiky kernel gradient (good for pressure force)
function spikyGrad(rx, ry, r, h) {
    if (r >= h || r < 1e-6) return { x: 0, y: 0 };
    const coeff = -10.0 / (Math.PI * Math.pow(h, 5));
    const diff = h - r;
    const scale = coeff * diff * diff / r;
    return { x: scale * rx, y: scale * ry };
}

// Viscosity Laplacian kernel
function viscosityLaplacian(r, h) {
    if (r >= h) return 0;
    const coeff = 40.0 / (Math.PI * Math.pow(h, 5));
    return coeff * (h - r);
}
```

---

## 4. PIC/FLIP Method — Theory & Implementation <a name="pic-flip-method"></a>

### Core Idea

Particles carry position and velocity (Lagrangian tracking). A background **staggered MAC grid** handles pressure projection to enforce incompressibility. The key insight: transfer particle velocities to the grid, solve for pressure on the grid, then transfer corrected velocities back to particles.

### The PIC vs FLIP Distinction

- **PIC (Particle-In-Cell):** After the grid solve, particles receive the *absolute* new grid velocity. This is stable but overly smooth — it kills fine detail through numerical diffusion.
- **FLIP (Fluid-Implicit-Particle):** Particles receive only the *change* in grid velocity (delta). This preserves individual particle motion but can be noisy.
- **Practical blend:** Use `0.1 * PIC + 0.9 * FLIP` for best results. This gives detailed motion with controlled noise.

```javascript
// After grid pressure solve:
// prevGridVel = grid velocity BEFORE solve
// newGridVel  = grid velocity AFTER solve

// PIC update (stable but diffusive):
particleVel = interpolateFromGrid(newGridVel, particlePos);

// FLIP update (detailed but noisy):
deltaVel = interpolateFromGrid(newGridVel, particlePos)
         - interpolateFromGrid(prevGridVel, particlePos);
particleVel = particleVel + deltaVel;

// Blended (recommended):
const flipRatio = 0.9;
picVel = interpolateFromGrid(newGridVel, particlePos);
flipVel = particleVel + deltaVel;
particleVel = flipRatio * flipVel + (1 - flipRatio) * picVel;
```

### Algorithm (Per Timestep) — Based on Matthias Müller's "Ten Minute Physics" Tutorial #18

```
1. SIMULATE PARTICLES
   - Apply gravity: v_i += dt * g
   - Advect positions: x_i += dt * v_i
   - Push particles out of solid obstacles

2. TRANSFER VELOCITY: PARTICLES → GRID
   - Clear grid velocities and weight accumulators
   - For each particle, splat velocity onto 4 nearest grid nodes
     using bilinear weights (for u and v components separately)
   - Divide accumulated velocity by accumulated weights
   - Save a COPY of grid velocities (for FLIP delta computation)

3. CLASSIFY CELLS
   - Cells containing particles → FLUID
   - Boundary cells → SOLID
   - Everything else → AIR
   - Important: velocities between two AIR cells are UNDEFINED (not zero!)

4. MAKE VELOCITY FIELD INCOMPRESSIBLE (Pressure Projection)
   - Use Gauss-Seidel iteration with overrelaxation:
     For n iterations (40-100):
       For each FLUID cell (i,j):
         d = u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)  // divergence
         s = s(i+1,j) + s(i-1,j) + s(i,j+1) + s(i,j-1)  // sum of non-solid neighbors
         d *= overrelaxation  // typically 1.9
         // Optionally subtract density correction: d -= k * (density - restDensity)
         u(i,j)   += d * s(i-1,j) / s
         u(i+1,j) -= d * s(i+1,j) / s
         v(i,j)   += d * s(i,j-1) / s
         v(i,j+1) -= d * s(i,j+1) / s

5. TRANSFER VELOCITY: GRID → PARTICLES (PIC/FLIP blend)
   - Compute delta = newGridVel - savedGridVel
   - For each particle, interpolate both new velocity and delta
   - Blend: v_particle = flipRatio * (v_particle + delta) + (1-flipRatio) * newVel

6. HANDLE PARTICLE-PARTICLE SEPARATION (Anti-drift)
   - Push overlapping particles apart using spatial grid
   - Compute particle density per cell; modify divergence to account
     for regions that are too dense (density correction term)
```

### Bilinear Interpolation (Particle ↔ Grid Transfer)

The transfer between particles and grid uses bilinear weights based on distance within the cell:

```javascript
function transferParticleToGrid(particles, grid, h) {
    // Clear grid
    grid.u.fill(0); grid.v.fill(0);
    grid.uWeight.fill(0); grid.vWeight.fill(0);

    for (const p of particles) {
        // For u-component: grid is offset by h/2 in y
        const ux = p.x / h;
        const uy = (p.y - 0.5 * h) / h;  // Note y-offset for staggered grid
        const ui = Math.floor(ux);
        const uj = Math.floor(uy);
        const udx = ux - ui;
        const udy = uy - uj;

        // Bilinear weights
        const w00 = (1 - udx) * (1 - udy);
        const w10 = udx * (1 - udy);
        const w01 = (1 - udx) * udy;
        const w11 = udx * udy;

        // Accumulate weighted velocity
        grid.u[ui][uj]     += w00 * p.vx;  grid.uWeight[ui][uj]     += w00;
        grid.u[ui+1][uj]   += w10 * p.vx;  grid.uWeight[ui+1][uj]   += w10;
        grid.u[ui][uj+1]   += w01 * p.vx;  grid.uWeight[ui][uj+1]   += w01;
        grid.u[ui+1][uj+1] += w11 * p.vx;  grid.uWeight[ui+1][uj+1] += w11;

        // Same for v-component (offset by h/2 in x instead)
        // ...
    }

    // Normalize: divide velocity by weight
    for (let i = 0; i < grid.width; i++) {
        for (let j = 0; j < grid.height; j++) {
            if (grid.uWeight[i][j] > 0) grid.u[i][j] /= grid.uWeight[i][j];
            if (grid.vWeight[i][j] > 0) grid.v[i][j] /= grid.vWeight[i][j];
        }
    }
}
```

### Grid Offsets for Staggered Grid

This is a common source of bugs. Remember:
- **u-component** sampling position for particle at `(x, y)`: use `(x, y - h/2)`
- **v-component** sampling position for particle at `(x, y)`: use `(x - h/2, y)`
- **Density/pressure** sampling: use `(x - h/2, y - h/2)` (cell center)

### Pressure Solver: Gauss-Seidel with Overrelaxation

```javascript
function solvePressure(grid, numIterations, overrelaxation, densityCorrection) {
    for (let iter = 0; iter < numIterations; iter++) {
        for (let i = 1; i < grid.width - 1; i++) {
            for (let j = 1; j < grid.height - 1; j++) {
                if (grid.cellType[i][j] !== FLUID) continue;

                const sx0 = grid.solid[i-1][j] ? 0 : 1;  // left neighbor
                const sx1 = grid.solid[i+1][j] ? 0 : 1;  // right neighbor
                const sy0 = grid.solid[i][j-1] ? 0 : 1;  // bottom neighbor
                const sy1 = grid.solid[i][j+1] ? 0 : 1;  // top neighbor
                const s = sx0 + sx1 + sy0 + sy1;
                if (s === 0) continue;

                let div = grid.u[i+1][j] - grid.u[i][j]
                        + grid.v[i][j+1] - grid.v[i][j];

                // Density correction (prevents particle drift/clumping)
                if (densityCorrection) {
                    const stiffness = 1.0;
                    div -= stiffness * (grid.particleDensity[i][j] - grid.restDensity);
                }

                div *= overrelaxation;  // typically 1.9

                grid.u[i][j]   += div * sx0 / s;
                grid.u[i+1][j] -= div * sx1 / s;
                grid.v[i][j]   += div * sy0 / s;
                grid.v[i][j+1] -= div * sy1 / s;
            }
        }
    }
}
```

### Key Parameters (PIC/FLIP)

| Parameter | Typical Value | Notes |
|---|---|---|
| Grid resolution | 50×50 to 200×100 | Higher = more detail, slower |
| Cell size `h` | domainWidth / gridWidth | Physical size per cell |
| Particles per cell | 4–9 | Seeded randomly at init |
| `dt` (timestep) | 1/60 to 1/120 s | Fixed or adaptive |
| `flipRatio` | 0.9 | 0 = pure PIC, 1 = pure FLIP |
| `overrelaxation` | 1.9 | Range (1, 2), dramatically speeds convergence |
| Solver iterations | 40–100 | More = more accurate pressure |
| Density stiffness `k` | 1.0 | Drift compensation strength |

### Handling Particle Drift

A key problem in FLIP: particles can drift and clump because the pressure solver only sees velocities, not particle positions. Two fixes are needed:

1. **Push particles apart:** After advection, check for overlapping particles using a spatial grid and push them apart (like a soft collision). This is a direct physical correction.

2. **Density correction in the solver:** Compute a particle density at each cell center. If a cell is denser than the rest density `ρ_0`, add extra divergence to push fluid outward:
```
div -= k * (cellDensity - restDensity)
```
The rest density `ρ_0` is the average particle density across all fluid cells at initialization.

---

## 5. Implementation Checklist <a name="implementation-checklist"></a>

### For SPH:
- [ ] Particle data structure (position, velocity, density, pressure, force)
- [ ] Spatial hash grid for neighbor search
- [ ] Kernel functions (Poly6 for density, Spiky for pressure, Viscosity Laplacian)
- [ ] Density computation loop
- [ ] Pressure computation (Tait equation of state)
- [ ] Force accumulation (pressure + viscosity + gravity)
- [ ] Symplectic Euler integration
- [ ] Boundary collision (reflect + damp)
- [ ] Rendering (circles or metaballs)

### For PIC/FLIP:
- [ ] Particle data structure (position, velocity)
- [ ] Staggered MAC grid (u, v, cellType, solid flags)
- [ ] Particle initialization (random jitter within fluid cells)
- [ ] Gravity integration on particles
- [ ] Particle-to-grid transfer (bilinear weights, handle stagger offsets!)
- [ ] Save pre-solve grid velocities (for FLIP delta)
- [ ] Cell type classification (FLUID / AIR / SOLID)
- [ ] Gauss-Seidel pressure solve with overrelaxation
- [ ] Density correction term in divergence
- [ ] Grid-to-particle transfer (PIC/FLIP blend)
- [ ] Particle separation (push apart)
- [ ] Obstacle collision handling
- [ ] Rendering (particles as circles, or fill fluid cells)

---

## 6. Rendering Techniques <a name="rendering"></a>

### Simple: Draw Particles as Circles
```javascript
ctx.fillStyle = 'rgba(30, 120, 255, 0.7)';
for (const p of particles) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, particleRadius, 0, Math.PI * 2);
    ctx.fill();
}
```

### Better: Color by Velocity Magnitude
```javascript
const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
const t = Math.min(speed / maxSpeed, 1);
// Blue (slow) → White (fast)
const r = Math.floor(30 + 225 * t);
const g = Math.floor(120 + 135 * t);
const b = 255;
ctx.fillStyle = `rgb(${r},${g},${b})`;
```

### Advanced: Metaball / Screen-Space Fluid Rendering
For a smooth water surface look:
1. Render each particle as a Gaussian blob to an offscreen texture
2. Threshold the accumulated density field
3. Apply blur and shading for a liquid appearance

This requires WebGL or a Canvas ImageData approach.

### Grid-Based Rendering (FLIP)
Color cells based on type:
- FLUID cells → blue
- AIR cells → transparent
- SOLID cells → gray

---

## 7. Performance Optimization <a name="performance"></a>

1. **Typed Arrays:** Use `Float32Array` / `Float64Array` for all particle and grid data. Avoid arrays of objects.

2. **Flat grid indexing:** Use `index = i + j * width` instead of 2D arrays.

3. **Spatial hashing:** Essential for SPH neighbor search. Use cell size = kernel radius.

4. **Limit particle count:** 1000–3000 particles runs well in browser Canvas. 5000+ needs WebGL or careful optimization.

5. **Gauss-Seidel iterations:** 40–80 is usually sufficient with overrelaxation = 1.9. More iterations = better but slower.

6. **requestAnimationFrame:** Use for the render loop. Decouple physics steps if needed (fixed dt, multiple sub-steps per frame).

7. **Web Workers:** For heavy SPH neighbor searches, offload computation to a worker.

8. **WebGL compute:** For 10k+ particles, consider GPU-based approaches using transform feedback or compute shaders.

---

## 8. Common Pitfalls & Fixes <a name="pitfalls"></a>

| Problem | Cause | Fix |
|---|---|---|
| Particles explode | Timestep too large, or stiffness too high | Reduce `dt`, reduce `k`, add CFL check |
| Particles clump/drift | No density correction in solver | Add density correction term to divergence |
| Checkerboard pressure | Non-staggered grid | Use MAC (staggered) grid layout |
| Particles leak through walls | Boundary not checked after advection | Push particles out of solids after each step |
| FLIP is too noisy | Pure FLIP (ratio=1.0) | Blend with PIC: use flipRatio=0.9 |
| Water looks too viscous | PIC ratio too high | Increase flipRatio toward 1.0 |
| Simulation is slow | O(n²) neighbor search | Use spatial hash grid |
| Air cells cause artifacts | Accessing velocities between air cells | Only process FLUID cells; air velocities are undefined |
| Staggered grid bugs | Wrong offsets for u vs v components | u offset: (0, -h/2), v offset: (-h/2, 0) |
| Solver doesn't converge | No overrelaxation | Use overrelaxation factor 1.9 |

---

## 9. Reference Resources <a name="references"></a>

### Primary References (Provided by User)

1. **SPH Basics — Dive CAE Blog**
   - URL: https://www.divecae.com/blog/sph-basics
   - Covers: SPH discretization, kernel functions, continuous interpolation,
     Lagrangian nature, force computation (pressure, viscosity, gravity),
     advantages/disadvantages vs mesh-based methods
   - Key concepts: Kernel weighting, Dirac delta analogy, neighbor-based
     field reconstruction, free-surface handling

2. **FLIP Water Simulator — Matthias Müller (Ten Minute Physics #18)**
   - Code: https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html
   - PDF slides: https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf
   - Covers: Complete PIC/FLIP implementation, staggered grid, two-phase
     simulation (water + air), Gauss-Seidel solver with overrelaxation,
     particle-grid transfer, density correction, drift compensation,
     particle separation

### Supplementary References

3. **"Fluid Simulation for Computer Graphics" by Robert Bridson** — The canonical textbook.
   Free SIGGRAPH course notes available on the author's website.

4. **Zhu & Bridson (SIGGRAPH 2005)** — "Animating Sand as a Fluid." The foundational
   PIC/FLIP paper for computer graphics. Introduced the blended PIC/FLIP approach.

5. **Brackbill & Ruppel (1986)** — Original FLIP method paper. Key contribution:
   transferring velocity *changes* instead of absolute velocities to reduce diffusion.

6. **Ryan Guy's PIC/FLIP Simulator** — Detailed documentation with full C++ implementation.
   URL: http://rlguy.com/gridfluidsim/

7. **WebGL PIC/FLIP by Austin Eng** — GPU-based implementation showing shader-based
   particle-to-grid transfer and pressure solve.
   URL: https://github.com/austinEng/WebGL-PIC-FLIP-Fluid

### Key Equations Quick Reference

```
SPH Density:        ρ_i = Σ_j m_j W(|x_i - x_j|, h)
SPH Pressure:       p_i = k((ρ_i/ρ_0)^γ - 1)
Grid Divergence:    d = u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)
Overrelaxation:     d *= ω,  where 1 < ω < 2  (use 1.9)
Density Correction: d -= k(ρ_cell - ρ_0)
FLIP Blend:         v = α·v_FLIP + (1-α)·v_PIC,  α = 0.9
Bilinear Weight:    w = (1 - dx/h)(1 - dy/h)
```

---

## Output Format Guidelines

When building a fluid simulation artifact:

1. **Prefer a single HTML file or React component** — self-contained, no external dependencies beyond Canvas API or basic React.

2. **Include interactive controls:**
   - Play/pause/reset buttons
   - Slider for gravity, viscosity, or flipRatio
   - Click/drag to add fluid or create obstacles
   - Display particle count and FPS

3. **Use `requestAnimationFrame`** for the render loop.

4. **Initialize with a visually interesting setup** — dam break (block of fluid on one side), water drop, or sloshing tank.

5. **Canvas sizing:** Use a reasonable default (800×600 or responsive). Dark background makes blue water pop.

6. **Comment the code** — fluid simulation code can be dense. Add section comments for each phase of the algorithm.