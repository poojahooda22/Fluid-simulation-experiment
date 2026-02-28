# Particle Cleanup System

## Problem

Each click spawns particles into the fluid simulation. Without cleanup, the canvas overfills after a few clicks and hits the hard buffer cap (`maxParticles`), at which point a ring-buffer recycler overwrites old particles — causing visual chaos.

## How It Works

### Geometry-Based Targeting

On startup, the system computes how many particles would fill the entire tank at the current hex-packing density, then targets **45% fill**:

```
fullNumX × fullNumY = tankCapacity  (total hex-packed slots in tank)
TARGET_PARTICLES = floor(0.45 × tankCapacity)
maxParticles = ceil(TARGET_PARTICLES × 2.5)  (typed-array buffer size)
```

`TARGET_PARTICLES` scales automatically with window size and particle spacing — no magic numbers.

### Spawn (on click)

Each click spawns **250 particles** (down from the original 540). When the buffer is >90% full, spawn throttles to 30% (75 particles).

### Excess-Based Removal Trigger

After spawning, the system checks:

```
excess = max(0, numParticles - TARGET_PARTICLES)
```

If `excess > pendingRemoval`, it queues removal and computes a batch size:

```
pendingRemoval = excess
removalBatchSize = ceil(excess / 30)  → about 8 particles/frame
```

**No removal happens on initial load** — `pendingRemoval` starts at 0 and is only set inside `dropParticles()`.

### Distributed Micro-Batch Removal

Each frame, while `pendingRemoval > 0`, the system removes `removalBatchSize` particles using **x-bin distributed removal**:

1. Divide the tank width into **16 vertical strips** (bins)
2. Assign every particle to its bin based on x-position
3. Sort each bin by y ascending (bottom-most first)
4. **Round-robin** across bins: take one bottom particle from bin 0, then bin 1, ..., bin 15, then back to bin 0, etc.
5. Swap-remove all selected particles (descending index order for correctness)

This ensures removal is spread **evenly across the full width** — no visible hole or blank band appears at the bottom.

### Timing

- 250 particles spawned per click
- ~250 excess queued for removal
- ~8 removed per frame across 16 bins
- Removal completes in ~30 frames (~0.5 seconds at 60fps)
- Visually imperceptible — less than 1 particle removed per bin per frame

## Flow Diagram

```
User clicks canvas
       │
       ▼
  dropParticles()
       │
       ├── Spawn 250 particles (throttled to 75 if buffer >90% full)
       │
       ├── Compute excess = numParticles - TARGET_PARTICLES
       │
       └── If excess > pendingRemoval:
              pendingRemoval = excess
              removalBatchSize = ceil(excess / 30)

  ─── Each frame (in animation loop) ───

  if pendingRemoval > 0:
       │
       ├── batch = min(pendingRemoval, removalBatchSize)
       │
       ├── removeBottomDistributed(batch)
       │       │
       │       ├── Partition all particles into 16 x-bins
       │       ├── Sort each bin by y (bottom first)
       │       ├── Round-robin: pick 1 bottom particle per bin
       │       └── Swap-remove selected particles
       │
       └── pendingRemoval -= batch
```

## Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `TARGET_FILL` | 0.45 | Target fill fraction of tank capacity |
| `TARGET_PARTICLES` | computed | `floor(0.45 × tankCapacity)` |
| `maxParticles` | computed | `ceil(TARGET_PARTICLES × 2.5)` — buffer size |
| `baseDropCount` | 250 | Particles spawned per click |
| `SPREAD_FRAMES` | 30 | Frames over which to spread removal |
| `NUM_BINS` | 16 | Vertical strips for distributed removal |
| `SPAWN_THROTTLE_THRESHOLD` | 0.90 | Buffer fraction before throttle kicks in |
| `SPAWN_THROTTLE_FACTOR` | 0.30 | Spawn reduction when throttled |

## What Changed (v5)

All changes are in `src/simulation/createFluidPool.ts`:

| Lines | Change |
|-------|--------|
| 805-811 | Replaced `maxParticles = numX*numY + 3000` with geometry-based `TARGET_PARTICLES` and `maxParticles` |
| 820-823 | Replaced `kLayer`/`MICRO_BATCH_FRAMES` with `SPREAD_FRAMES`/`NUM_BINS`/`removalBatchSize` |
| 993 | `baseDropCount` changed from 540 to 250 |
| 1024-1028 | Replaced `pendingRemoval += kLayer` with excess-based trigger |
| 1033-1072 | Replaced `removeBottomParticles()` with `removeBottomDistributed()` (x-bin round-robin) |
| 1134-1138 | Main loop uses `removalBatchSize` instead of `ceil(kLayer/MICRO_BATCH_FRAMES)` |

## Monitoring

The existing console log prints particle count every 120 frames:

```
[wave-debug] F120: 45.2% moving, median speed: 0.0312, N=1723
```

Watch `N=` to confirm it stabilizes near `TARGET_PARTICLES` after clicking.
