# Fix: Settled Jitter, Wavy Particles & Splash Responsiveness

Documentation of all changes made to fix three interrelated problems in the FLIP fluid simulation.

**File**: `src/simulation/createFluidPool.ts`

---

## Problem 1: Bottom Particle Jitter / Bouncing

### Symptom

Particles resting at the bottom of the tank exhibited continuous micro-bouncing against neighboring particles instead of moving as a smooth, coherent wave. Adjacent particles oscillated out of phase, creating a "buzzing" effect.

### Root Cause

The `applyIdleWave` method uses a position-dependent phase:

```typescript
const phase = frequency * simTime + 2.5 * y - 0.3 * x;
```

The `2.5 * y` gradient meant vertically adjacent particles (separated by ~0.04 units) received different phase values — each pair was pushed in slightly different directions each frame. The physics systems (`pushParticlesApart`, pressure solver) then fought these velocity differences, creating a jitter cycle:

1. Wave pushes particle A left, particle B right
2. Separation pushes them back apart
3. Next frame, wave pushes differently again
4. Repeat → visible bouncing

### Solution: Render-Time Visual Wave Offset (Vertex Shader)

Instead of injecting wave velocities into the physics simulation, we moved the visual wave entirely into the GPU vertex shader. This produces identical visual sway without any physics interaction.

**Vertex shader** (lines 18, 25–26):

```glsl
uniform float u_waveTime;

// In main():
float wavePhase = 1.2 * u_waveTime + 1.0 * pos.y;
pos.x += 0.03 * sin(wavePhase);
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Amplitude | `0.03` | With simHeight=3.0 and ~700px canvas, ≈7px of visual sway — visible but subtle |
| Frequency | `1.2` | ~5.2 second period — slow, natural rhythm |
| Y gradient | `1.0` | Particles 0.04 apart get only 0.04 rad (2.3°) phase difference → virtually in-phase = coherent movement |

**Why this works:** All particles at the same Y coordinate get an identical X offset. Nearby particles (small Y difference) get nearly identical offsets. The GPU applies this purely visually — no forces, no velocities, no physics fighting.

**Uniform setup** (line 1104):
```typescript
const uWaveTime = gl.getUniformLocation(prog, 'u_waveTime')!;
```

**Uniform upload** (line 1562):
```typescript
gl!.uniform1f(uWaveTime, scene.simTime);
```

### Supporting Fix: Velocity Smoothing Between Settled Neighbors

Added in `pushParticlesApart` (lines 342–354) to further reduce jitter between slow-moving particle pairs:

```typescript
// Velocity smoothing for settled pairs: blend horizontal
// velocities so neighbors move coherently instead of fighting.
const vxi = this.particleVel[2 * i], vyi = this.particleVel[2 * i + 1];
const vxid = this.particleVel[2 * id], vyid = this.particleVel[2 * id + 1];
const si2 = vxi * vxi + vyi * vyi;
const sid2 = vxid * vxid + vyid * vyid;
const SETTLE_SPEED2 = 0.20 * 0.20;
if (si2 < SETTLE_SPEED2 && sid2 < SETTLE_SPEED2) {
    const blend = 0.01; // 1% per iter (doubled by symmetric visits ≈ 2%)
    const avgVx = 0.5 * (vxi + vxid);
    this.particleVel[2 * i] += (avgVx - vxi) * blend;
    this.particleVel[2 * id] += (avgVx - vxid) * blend;
}
```

**Trigger condition:** Both particles in a pair must be below speed 0.20. Only blends horizontal (X) velocity — vertical is left alone to avoid fighting gravity.

### Supporting Fix: `dampSettledParticles` Method

New method (lines 767–781) applies extra damping to very slow particles, scaled by how slow they are:

```typescript
dampSettledParticles(threshold: number, maxDamp: number) {
    // ...
    if (s2 < threshold2 && s2 > 0) {
        const speed = Math.sqrt(s2);
        const t = 1.0 - speed / threshold; // 1.0 at speed=0, 0.0 at threshold
        const damp = 1.0 - maxDamp * t;
        this.particleVel[2 * i] *= damp;
        this.particleVel[2 * i + 1] *= damp;
    }
}
```

Called in `simulate()` at line 935 with `threshold=0.15, maxDamp=0.03`. Particles at near-zero speed get up to 3% damping; particles approaching 0.15 speed get virtually none. This calms micro-oscillations without affecting active particles.

### Supporting Fix: Speed-Dependent Wall Bounce

In `handleParticleCollisions` (lines 471–491), wall collisions now distinguish between slow and fast particles:

```typescript
// Speed-dependent wall bounce: slow → zero (no jitter), fast → bounce
if (x < minX) {
    x = minX;
    const v = this.particleVel[2 * i];
    this.particleVel[2 * i] = Math.abs(v) < 0.3 ? 0 : v * -0.1;
}
```

| Speed | Behavior | Why |
|-------|----------|-----|
| `< 0.3` | Velocity → 0 | Eliminates sub-pixel wall bouncing that causes jitter |
| `≥ 0.3` | Velocity × -0.1 | Soft bounce — enough to look physical without causing chain reactions |

Previously, all wall collisions zeroed the normal velocity component, which created edge cases where particles oscillated between the wall and their neighbors.

---

## Problem 2: Slow Splash Response (~2 Second Delay)

### Symptom

When the user flicked the cursor through the fluid, the splash response took approximately 2 seconds to ramp up to full intensity. The interaction felt sluggish and unresponsive.

### Root Cause

The obstacle collision response was purely **position-based** — it only considered how deeply a particle overlapped with the obstacle, not how fast the obstacle was moving:

```typescript
const repulse = OBSTACLE_REPULSION * (overlap / minDist);
```

A fast-moving cursor produces the same force as a slowly-drifting one at equal penetration depth. Grid velocity injection through the pressure solver took 3-4 frames to propagate, during which the visual response was minimal.

### Solution: Velocity-Dependent Splash Impulse

Added a direct velocity impulse in `handleParticleCollisions` (lines 454–464) that fires when the obstacle moves fast:

```typescript
// F) Velocity-dependent splash impulse
const obstSpeed2 = obstacleVelX * obstacleVelX + obstacleVelY * obstacleVelY;
if (obstSpeed2 > 0.25) {
    const impulseK = 2.0;
    outVx += obstacleVelX * impulseK;
    outVy += obstacleVelY * impulseK;
    // Stochastic cohesion: 70% chance to fall as group, 30% individually
    if (Math.random() < 0.70) {
        this.particleCohesion[i] = 1.0;
    }
}
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Speed threshold | `0.25` (speed² > 0.25 → speed > 0.5) | Filters out slow drift — only fast interactions splash |
| `impulseK` | `2.0` | Particle receives 2× the obstacle velocity as an impulse |
| Cohesion assignment | 70% chance | Groups splashed particles for coordinated falling (see cohesion system below) |

**Safety check:** Max displacement/frame = 25 × 2 / 60 = 0.83 units. Tank is ~3.0 wide. Wall clamping catches escapees.

### Tuning Changes

| Constant | Before | After | Location | Effect |
|----------|--------|-------|----------|--------|
| `OBSTACLE_REPULSION` | 5.0 | **10.0** | Line 118 | Stronger position-based push on contact |
| `MOUSE_MAX_SPEED` | 15.0 | **25.0** | Line 1198 | Allows faster cursor motion → higher impulse |
| `clampVelocities` | 10.0 | **15.0** | Line 936 | Higher max particle speed → more energetic splash |

### Supporting System: Falling Cohesion

New `applyFallingCohesion()` method (lines 783–868) makes splashed particles fall as coherent groups rather than scattering individually:

1. Particles receiving splash impulse get `particleCohesion[i] = 1.0`
2. Each frame, cohesive particles search neighbors via spatial hash
3. Neighbors with cohesion → blend velocities toward group average (`blendRate = 0.15`)
4. Settled neighbors near a cohesive particle → recruited into the group (35% of leader velocity)
5. Cohesion decays at 0.008/frame and clears when particle speed drops below 0.05

This creates the visual effect of "water blobs" falling and splashing together rather than a diffuse spray.

---

## Problem 3: Missing Top Wall + Resize Bug

### Symptom

After maximizing the browser window, particles could escape through the top of the tank. The top boundary didn't update to match the new canvas dimensions.

### Root Cause

**Missing top wall cells:** The grid setup only marked left, right, and bottom edges as solid:

```typescript
// BEFORE:
if (i === 0 || i === f.fNumX - 1 || j === 0) s = 0.0;
```

The condition `j === f.fNumY - 1` (top row) was missing. The pressure solver had no top boundary → particles drifted through.

**Static view bounds:** View bounds were declared as `const` and computed once at initialization. On resize, only the canvas pixel dimensions updated — the physics-to-screen mapping stayed unchanged.

### Solution

**Grid setup** (line 1063) — added top wall:

```typescript
if (i === 0 || i === f.fNumX - 1 || j === 0 || j === f.fNumY - 1)
    s = 0.0;
```

**Top wall separation** in `pushParticlesApart` (lines 279, 387–391):

```typescript
const topWall = (this.fNumY - 1) * wh;

// ... in wall loop:
// Top wall
const distTop = topWall - this.particlePos[2 * i + 1];
if (distTop > 0 && distTop < halfDist) {
    this.particlePos[2 * i + 1] -= halfDist - distTop;
}
```

**Dynamic view bounds** (lines 1076–1080) — changed from `const` to `let`:

```typescript
let viewLeft = f.h;
let viewBottom = f.h;
const viewRight = (f.fNumX - 1) * f.h;
let viewWidth = viewRight - viewLeft;
let viewHeight = viewableSimHeight;
```

**Resize recalculation** in render loop (lines 1517–1531):

```typescript
if (cvs.width !== targetW || cvs.height !== targetH) {
    cvs.width = targetW;
    cvs.height = targetH;
    const gridViewW = (f.fNumX - 1) * f.h - f.h;
    const physPerPx = gridViewW / cvs.clientWidth;
    viewWidth = gridViewW;
    viewHeight = cvs.clientHeight * physPerPx;
    viewLeft = f.h;
    viewBottom = Math.min(f.h, viewableTop - viewHeight);
}
```

---

## Summary of All Edits (by line order in final file)

| Lines | Change | Fixes |
|-------|--------|-------|
| 18 | `uniform float u_waveTime;` in vertex shader | Jitter |
| 25–26 | `pos.x += 0.03 * sin(1.2 * u_waveTime + 1.0 * pos.y)` | Jitter |
| 118 | `OBSTACLE_REPULSION = 10.0` (was 5.0) | Splash |
| 179 | `particleCohesion: Float32Array` property | Splash |
| 279 | `topWall = (this.fNumY - 1) * wh` | Top wall |
| 342–354 | Velocity smoothing for settled pairs in `pushParticlesApart` | Jitter |
| 387–391 | Top wall separation in `pushParticlesApart` | Top wall |
| 417–419 | Immune particle skip in obstacle collision | Splash |
| 454–464 | Velocity-dependent splash impulse + cohesion assignment | Splash |
| 471–491 | Speed-dependent wall bounce (slow→zero, fast→-0.1) | Jitter |
| 767–781 | `dampSettledParticles` method | Jitter |
| 783–868 | `applyFallingCohesion` method | Splash |
| 936 | `clampVelocities(15.0)` (was 10.0) | Splash |
| 1063 | `j === f.fNumY - 1` in grid solid cell condition | Top wall |
| 1076–1080 | `const` → `let` for view bounds | Resize |
| 1104 | `uWaveTime` uniform location | Jitter |
| 1198 | `MOUSE_MAX_SPEED = 25.0` (was 15.0) | Splash |
| 1517–1531 | View bounds recalculation on resize | Resize |
| 1562 | `gl!.uniform1f(uWaveTime, scene.simTime)` | Jitter |

---

## Verification

### Console diagnostics (every 120 frames):

```
[wave-debug] F720: 95.3% moving, median speed: 0.0287, N=1723
```

### Visual checks:

| Check | Expected |
|-------|----------|
| Bottom particles at rest | Smooth coherent sway, no buzzing/jitter between neighbors |
| Fast flick through fluid | Immediate energetic splash response, not 2s ramp |
| Maximize window | Particles blocked at top, no escape |
| Resize window | Aspect ratio maintained, view bounds recalculated |
| Splash landing | Particles fall in groups (cohesion), not as diffuse spray |
| Stability | No explosions, no particle escape, no regressions |