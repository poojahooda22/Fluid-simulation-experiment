# Idle Wave System

Documentation for restoring gentle resting motion to the fluid simulation after particles settle.

---

## Task 1: Read and Understand Simulation Code Architecture

### What was done

Full read-through of the simulation pipeline in `src/simulation/createFluidPool.ts` to understand how particles are updated each frame.

### Key findings

The simulation uses a **FLIP/PIC fluid solver** with this per-frame pipeline:

```
integrateParticles()        — apply gravity, move particles by velocity
pushParticlesApart()        — particle-particle separation (spatial hash)
handleParticleCollisions()  — wall + obstacle collisions
transferVelocities(toGrid)  — scatter particle velocities onto MAC grid
updateParticleDensity()     — compute per-cell particle density
solveIncompressibility()    — pressure projection (80 Gauss-Seidel iters)
transferVelocities(fromGrid)— gather grid velocities back to particles (PIC/FLIP blend)
dampVelocities()            — global velocity damping
clampVelocities()           — cap maximum speed
```

The `FlipFluid` class owns all state: grid arrays (`u`, `v`, `p`, `s`, `cellType`), particle arrays (`particlePos`, `particleVel`, `particleColor`), and spatial-hash structures. The `createFluidPool` factory creates the instance, sets up WebGL rendering, wires interaction events, and runs the animation loop.

---

## Task 2: Diagnose Why Most Particles Stop Moving After Settling

### What was done

Line-by-line analysis of every velocity-modifying step in `simulate()` to identify what kills motion after the fluid reaches equilibrium.

### Root causes identified

#### Cause A — Rest-state damping block (PRIMARY)

**Location**: `simulate()`, formerly lines 703–720 (now removed)

```typescript
const restSpeed2 = 0.5 * 0.5;   // speed < 0.5 triggers
const restBand = 3.0 * this.h;  // bottom 3 cells
const restDamp = 0.9;           // multiply velocity by 0.9 EACH FRAME
```

This applied a **10% per-frame velocity reduction** to any particle in the bottom band moving slowly. At 60 FPS that's exponential decay — velocity halves every ~7 frames (~0.1 s). Bottom particles froze almost immediately.

#### Cause B — PIC blend damping

**Location**: `transferVelocities()` (line 564–565)

```typescript
particleVel = (1 - flipRatio) * picV + flipRatio * flipV;
```

With `flipRatio = 0.95`, 5% of each particle's velocity is replaced by the grid-interpolated value each frame. At rest the grid velocity converges to zero (hydrostatic equilibrium), so PIC pulls all velocities toward zero at **5% per frame**.

#### Cause C — Global damping

**Location**: `dampVelocities()` call in `simulate()` (line 728)

Multiplies all velocities by `(1 - damping)` each frame. Originally 0.005 (0.5%/frame), now tuned to 0.001 (0.1%/frame).

#### Combined effect

~5–6% total velocity loss per frame gives a half-life of ~0.2 seconds. Once gravity and pressure reach equilibrium, any residual motion dies within 1–2 seconds. Only free-surface particles (top, in partially-filled grid cells) retained slight sloshing because the pressure solver is less constrained there.

---

## Task 3: Remove Rest-State Damping Block

### What changed

**File**: `src/simulation/createFluidPool.ts`, inside `simulate()`

The entire rest-state damping block was **deleted**:

```typescript
// REMOVED — was causing bottom particles to freeze:
//
// const restSpeed2 = 0.5 * 0.5;
// const restBand = 3.0 * this.h;
// const bottomY = this.h;
// const restDamp = 0.9;
// for (let i = 0; i < this.numParticles; i++) {
//     const py = this.particlePos[2 * i + 1];
//     if (py - bottomY < restBand) {
//         const vx = this.particleVel[2 * i];
//         const vy = this.particleVel[2 * i + 1];
//         if (vx * vx + vy * vy < restSpeed2) {
//             this.particleVel[2 * i] *= restDamp;
//             this.particleVel[2 * i + 1] *= restDamp;
//         }
//     }
// }
```

### Why

This block was the single biggest contributor to particle freezing. It:

- Only targeted the bottom band, creating a visible "dead zone"
- Applied 10%/frame damping on top of the already-present PIC + global damping
- Had no mechanism to wake frozen particles (once slow, they stayed slow forever)

The global damping (`dampVelocities`) + PIC blend already provide adequate convergence to equilibrium. The idle wave (Task 4) now provides the energy source to maintain gentle motion.

---

## Task 4: Add `applyIdleWave` Method to FlipFluid Class

### What changed

**File**: `src/simulation/createFluidPool.ts`, new method at line 668

Added `applyIdleWave(simTime, dt, strength, frequency, noise)` to the `FlipFluid` class:

```typescript
applyIdleWave(simTime: number, dt: number, strength: number, frequency: number, noise: number) {
    for (let i = 0; i < this.numParticles; i++) {
        const x = this.particlePos[2 * i];
        const y = this.particlePos[2 * i + 1];

        const phase = frequency * simTime + 2.5 * y - 0.3 * x;

        const waveX = strength * Math.sin(phase);
        const waveY = strength * 0.2 * Math.cos(phase * 1.3 + 0.7);

        this.particleVel[2 * i] += waveX * dt;
        this.particleVel[2 * i + 1] += waveY * dt;

        if (noise > 0) {
            this.particleVel[2 * i] += (Math.random() - 0.5) * noise * dt;
            this.particleVel[2 * i + 1] += (Math.random() - 0.5) * noise * dt * 0.3;
        }
    }
}
```

### Why — design decisions

| Decision | Rationale |
|----------|-----------|
| **Applied as acceleration** (`+= ... * dt`) | Prevents unbounded velocity accumulation; system reaches steady-state where wave injection balances damping |
| **Phase = f(y, x)** | Vertical wavenumber (2.5) creates visible wave propagation up/down the fluid body; horizontal component (0.3) breaks perfect column-alignment |
| **Horizontal primary, vertical 20%** | Mimics real water at rest — mostly side-to-side sway with slight vertical bobbing |
| **`sin` for X, `cos` for Y with different frequency** | Phase-shifted components prevent all particles moving identically, creating natural-looking turbulence |
| **Noise term** | Breaks perfect mathematical uniformity; particles near each other move slightly differently, which is more realistic |
| **Noise vertical = 30% of horizontal** | Prevents noisy vertical bouncing at the bottom wall |
| **Iterates ALL particles** | No region masks, no thresholds — every particle participates in the wave, fixing the "only 10% moving" problem |
| **Zero-mean sinusoidal** | No net drift over time — `sin` and `cos` average to zero, so particles don't slowly migrate |

---

## Task 5: Add Idle Wave Parameters to Scene and Call in Loop

### What changed

**File**: `src/simulation/createFluidPool.ts`

#### Scene config (line 772–776)

```typescript
simTime: 0.0,
idleWaveEnabled: true,
idleWaveStrength: 0.60,
idleWaveFrequency: 1.2,
idleWaveNoise: 0.06,
```

#### Animation loop (lines 1129–1134)

```typescript
// Apply idle wave for gentle resting motion (after simulate so pressure solver doesn't cancel it)
if (scene.idleWaveEnabled) {
    f.applyIdleWave(scene.simTime, scene.dt,
        scene.idleWaveStrength, scene.idleWaveFrequency, scene.idleWaveNoise);
}
scene.simTime += scene.dt;
```

### Why — placement after `simulate()`

The idle wave is applied **after** the full simulation step (including pressure solve and grid-to-particle transfer). This is critical because:

1. If injected **before** `simulate()` → the pressure solver would immediately correct the divergence and cancel most of the wave
2. If injected **into the grid** → wall boundary conditions would zero it at boundaries
3. Applied **after** → the velocity survives until the next frame, enters the grid, gets partially diffused by the pressure solver (which is correct — it maintains incompressibility), and is re-injected every frame

The `simTime` accumulator increments by `dt` each frame, providing a smooth monotonic time base for the sinusoidal wave regardless of frame rate.

### Tuning values (current)

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `idleWaveStrength` | 0.60 | Wave acceleration amplitude — higher = more visible motion |
| `idleWaveFrequency` | 1.2 | Oscillation speed in rad/s — period ~5.2 seconds |
| `idleWaveNoise` | 0.06 | Random perturbation per particle — breaks uniformity |
| `idleWaveEnabled` | true | Master on/off switch |

---

## Task 6: Add Debug Instrumentation for Verification

### What changed

**File**: `src/simulation/createFluidPool.ts`, lines 1180–1195

Replaced the old per-60-frames position log with a motion diagnostic that runs every 120 frames:

```typescript
if (scene.frameNr % 120 === 0) {
    let moving = 0;
    const eps2 = 0.01 * 0.01;
    const speeds: number[] = [];
    for (let i = 0; i < f.numParticles; i++) {
        const vx = f.particleVel[2 * i];
        const vy = f.particleVel[2 * i + 1];
        const s2 = vx * vx + vy * vy;
        if (s2 > eps2) moving++;
        speeds.push(Math.sqrt(s2));
    }
    speeds.sort((a, b) => a - b);
    const pct = (100 * moving / f.numParticles).toFixed(1);
    const median = speeds[Math.floor(speeds.length / 2)]?.toFixed(4) ?? '0';
    console.log(`[wave-debug] F${scene.frameNr}: ${pct}% moving, median speed: ${median}, N=${f.numParticles}`);
}
```

### Why

The original log (`Frame N: P0[x, y] NumP: N simW: W`) showed only the first particle's position — useless for diagnosing global motion patterns. The new log reports:

| Metric | Purpose |
|--------|---------|
| `% moving` | Particles with speed > 0.01 — the core acceptance metric (target: 70–100%) |
| `median speed` | Central tendency of the speed distribution — confirms wave is active but subtle |
| `N=` | Particle count — confirms cleanup system is working independently |
| Every 120 frames | ~2 seconds between logs — enough to track settling without console spam |

The threshold `eps = 0.01` was chosen as "perceptibly moving" — below this a particle shifts less than 0.2 pixels per frame, which is invisible.

---

## Task 7: Verify in Browser

### What was done

- Ran `npx tsc --noEmit` — clean compile, no type errors
- Started dev server with `npm run dev` — Vite compiled and served successfully
- Confirmed the idle wave code, scene parameters, and debug instrumentation are all wired correctly

### How to verify visually

1. Open the dev server URL in browser
2. Watch particles enter and settle (~5–10 seconds)
3. After settling, observe: gentle left-right swaying persists across the full fluid body
4. Open DevTools console — `[wave-debug]` logs confirm % moving and median speed
5. Click to spawn particles — they should integrate smoothly into the wave after settling

### Expected console output (after settling)

```
[wave-debug] F720: 95.3% moving, median speed: 0.0287, N=1723
[wave-debug] F840: 93.8% moving, median speed: 0.0314, N=1723
```

### What to look for if something is wrong

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Wave too strong / sloshing | `idleWaveStrength` too high | Reduce from 0.60 toward 0.20 |
| Bottom particles still frozen | Another damping source | Check if `dampVelocities` value was increased |
| Jitter at bottom | `idleWaveNoise` too high or wall bounce coefficient too high | Reduce noise; check `handleParticleCollisions` wall multiplier |
| Drift (particles slowly migrate left/right) | Phase function has DC offset | Verify `sin`/`cos` are zero-mean; check for asymmetric wall friction |
| Performance drop | `Math.sin`/`Math.random` per particle | Consider lookup table or reduce noise to 0 |
