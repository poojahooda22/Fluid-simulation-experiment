
/**
 * createFluidPool — self-contained 2D particle-fluid pool.
 * Perfectly mimics the reference simulation.txt FlipFluid behavior
 * using physical coordinate domain (simHeight = 3.0).
 */

// ═══ Shaders (WebGL2) ══════════════════════════════════════════
const VERT = `#version 300 es
precision highp float;
in vec2 a_pos;
in vec3 a_color;
in float a_shape;
in float a_angle;
uniform vec2 u_res;
uniform vec2 u_offset;
uniform float u_ptSz;
uniform float u_waveTime;
out vec3 v_col;
flat out float v_shape;
flat out float v_angle;

void main(){
  vec2 pos = a_pos - u_offset;
  float wavePhase = 1.2 * u_waveTime + 1.0 * pos.y;
  pos.x += 0.03 * sin(wavePhase);
  vec2 screenTransform = vec2(2.0 / u_res.x, 2.0 / u_res.y);
  gl_Position = vec4(pos * screenTransform - vec2(1.0, 1.0), 0.0, 1.0);

  gl_PointSize = u_ptSz;
  v_col = a_color;
  v_shape = a_shape;
  v_angle = a_angle;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 v_col;
flat in float v_shape;
flat in float v_angle;
uniform float u_ptSz;
out vec4 o;

void main(){
  vec2 p = gl_PointCoord - 0.5;
  float ca = cos(v_angle); float sa = sin(v_angle);
  p = vec2(ca*p.x + sa*p.y, -sa*p.x + ca*p.y);
  int s = int(v_shape + 0.5);
  float a;

  if(s == 1){
    // Square — radius reduced by 1px
    float shrinkSq = 1.0 / u_ptSz;
    float outerSq = 0.42 - shrinkSq;
    float d = max(abs(p.x), abs(p.y));
    if(d > outerSq) discard;
    a = 1.0 - smoothstep(outerSq - 0.10, outerSq, d);
  } else if(s == 2){
    // Triangle pointing up
    float d1 = p.y + 0.30;
    float d2 = 0.40 - (1.732 * abs(p.x) + p.y);
    float d = min(d1, d2);
    if(d < 0.0) discard;
    a = smoothstep(0.0, 0.06, d);
  } else if(s == 3){
    // Plus
    float hw = 0.10;
    float hl = 0.42;
    bool inV = abs(p.x) < hw && abs(p.y) < hl;
    bool inH = abs(p.y) < hw && abs(p.x) < hl;
    if(!inV && !inH) discard;
    float dV = inV ? min(hw - abs(p.x), hl - abs(p.y)) : 0.0;
    float dH = inH ? min(hl - abs(p.x), hw - abs(p.y)) : 0.0;
    float d = max(dV, dH);
    a = smoothstep(0.0, 0.04, d);
  } else {
    // Circle (default, shape 0) — radius reduced by 1px
    float shrink = 1.0 / u_ptSz;
    float outerR = 0.47 - shrink;
    float d = length(p);
    if(d > outerR) discard;
    a = 1.0 - smoothstep(outerR - 0.10, outerR, d);
  }

  o = vec4(v_col, a);
}`;

// Mesh Shaders for Obstacle Disk
const MESH_VERT = `#version 300 es
precision highp float;
in vec2 a_pos;
uniform vec2 u_res;
uniform vec2 u_offset;
uniform vec3 u_color;
uniform vec2 u_translation;
uniform float u_scale;
out vec3 v_col;

void main(){
  vec2 v = u_translation + a_pos * u_scale - u_offset;
  vec2 screenTransform = vec2(2.0 / u_res.x, 2.0 / u_res.y);
  gl_Position = vec4(v * screenTransform - vec2(1.0, 1.0), 0.0, 1.0);
  v_col = u_color;
}`;

const MESH_FRAG = `#version 300 es
precision highp float;
in vec3 v_col;
out vec4 o;
void main(){
  o = vec4(v_col, 1.0);
}`;

// ═══ Constants ══════════════════════════════════════════════════
const FLUID_CELL = 0;
const AIR_CELL = 1;
const SOLID_CELL = 2;
const OBSTACLE_REPULSION = 10.0; // outward push strength, tunable 1.0–10.0

// ═══ Particle Cleanup Tuning ═════════════════════════════════════
const SPAWN_THROTTLE_THRESHOLD = 0.90; // fraction of maxParticles before throttling
const SPAWN_THROTTLE_FACTOR = 0.9;    // fraction of dropCount allowed when throttled

export interface PoolOptions {
    particleRadius?: number;
    numParticles?: number;
    gravity?: number;
    flipRatio?: number;
    dt?: number;
    pressureIters?: number;
    sepIters?: number;
    overRelax?: number;
    obstacleRadius?: number;
}

export interface PoolAPI {
    cleanup: () => void;
    reset: () => void;
    setFlipRatio: (v: number) => void;
    setPaused: (v: boolean) => void;
    setShowDebug: (v: boolean) => void;
    setTiltForce: (x: number, y: number) => void;
}

function clamp(x: number, min: number, max: number) {
    if (x < min) return min;
    else if (x > max) return max;
    else return x;
}

// ----------------- start of simulator ------------------------------
class FlipFluid {
    density: number;
    fNumX: number;
    fNumY: number;
    h: number;
    fInvSpacing: number;
    fNumCells: number;

    u: Float32Array;
    v: Float32Array;
    du: Float32Array;
    dv: Float32Array;
    prevU: Float32Array;
    prevV: Float32Array;
    p: Float32Array;
    s: Float32Array;
    cellType: Int32Array;
    cellColor: Float32Array;

    maxParticles: number;
    particlePos: Float32Array;
    particleColor: Float32Array;
    particleVel: Float32Array;
    particleShape: Float32Array;
    particleAngle: Float32Array;
    particleAngVel: Float32Array;
    particleImmune: Uint8Array;
    particleCohesion: Float32Array;
    particleDensity: Float32Array;
    particleRestDensity: number;

    particleRadius: number;
    particleGap: number;
    pInvSpacing: number;
    pNumX: number;
    pNumY: number;
    pNumCells: number;

    numCellParticles: Int32Array;
    firstCellParticle: Int32Array;
    cellParticleIds: Int32Array;

    numParticles: number;

    constructor(density: number, width: number, height: number, spacing: number, particleRadius: number, particleGap: number, maxParticles: number) {
        // fluid
        this.density = density;
        this.fNumX = Math.floor(width / spacing) + 1;
        this.fNumY = Math.floor(height / spacing) + 1;
        this.h = Math.max(width / this.fNumX, height / this.fNumY);
        this.fInvSpacing = 1.0 / this.h;
        this.fNumCells = this.fNumX * this.fNumY;

        this.u = new Float32Array(this.fNumCells);
        this.v = new Float32Array(this.fNumCells);
        this.du = new Float32Array(this.fNumCells);
        this.dv = new Float32Array(this.fNumCells);
        this.prevU = new Float32Array(this.fNumCells);
        this.prevV = new Float32Array(this.fNumCells);
        this.p = new Float32Array(this.fNumCells);
        this.s = new Float32Array(this.fNumCells);
        this.cellType = new Int32Array(this.fNumCells);
        this.cellColor = new Float32Array(3 * this.fNumCells);

        // particles
        this.maxParticles = maxParticles;
        this.particlePos = new Float32Array(2 * this.maxParticles);
        this.particleColor = new Float32Array(3 * this.maxParticles);
        for (let i = 0; i < this.maxParticles; i++) {
            this.particleColor[3 * i] = 1.0;
            this.particleColor[3 * i + 1] = 1.0;
            this.particleColor[3 * i + 2] = 1.0;
        }

        this.particleVel = new Float32Array(2 * this.maxParticles);
        this.particleShape = new Float32Array(this.maxParticles);
        for (let i = 0; i < this.maxParticles; i++) {
            this.particleShape[i] = i % 4;
        }
        this.particleAngle = new Float32Array(this.maxParticles);
        this.particleAngVel = new Float32Array(this.maxParticles);
        for (let i = 0; i < this.maxParticles; i++) {
            this.particleAngle[i] = Math.random() * Math.PI * 2;
            this.particleAngVel[i] = (0.3 + Math.random() * 0.7) * (Math.random() < 0.5 ? 1 : -1);
        }
        this.particleImmune = new Uint8Array(this.maxParticles);
        this.particleCohesion = new Float32Array(this.maxParticles);
        this.particleDensity = new Float32Array(this.fNumCells);
        this.particleRestDensity = 0.0;

        this.particleRadius = particleRadius;
        this.particleGap = particleGap;
        this.pInvSpacing = 1.0 / (2.2 * particleRadius + particleGap);
        this.pNumX = Math.floor(width * this.pInvSpacing) + 1;
        this.pNumY = Math.floor(height * this.pInvSpacing) + 1;
        this.pNumCells = this.pNumX * this.pNumY;

        this.numCellParticles = new Int32Array(this.pNumCells);
        this.firstCellParticle = new Int32Array(this.pNumCells + 1);
        this.cellParticleIds = new Int32Array(maxParticles);

        this.numParticles = 0;
    }

    integrateParticles(dt: number, gravity: number, tiltForceX: number = 0, tiltForceY: number = 0) {
        for (let i = 0; i < this.numParticles; i++) {
            this.particleVel[2 * i] += dt * tiltForceX;
            this.particleVel[2 * i + 1] += dt * (gravity + tiltForceY);
            this.particlePos[2 * i] += this.particleVel[2 * i] * dt;
            this.particlePos[2 * i + 1] += this.particleVel[2 * i + 1] * dt;
        }
    }

    pushParticlesApart(numIters: number) {
        const colorDiffusionCoeff = 0.001;
        const minDist = 2.0 * this.particleRadius + this.particleGap;
        const minDist2 = minDist * minDist;

        // Wall-particle separation: same spacing rule as between particles.
        // Wall surface sits at one cell width from domain edge (solid cell boundary).
        // halfDist = minDist/2: the clearance a particle center must keep from the wall,
        // identical to the half-correction each particle gets in pair separation.
        const wh = 1.0 / this.fInvSpacing;
        const halfDist = minDist * 0.5;
        const leftWall = wh;
        const rightWall = (this.fNumX - 1) * wh;
        const bottomWall = wh;
        const topWall = (this.fNumY - 1) * wh;

        for (let iter = 0; iter < numIters; iter++) {
            // Rebuild spatial hash each iteration so moved particles are found
            this.numCellParticles.fill(0);
            for (let i = 0; i < this.numParticles; i++) {
                const x = this.particlePos[2 * i];
                const y = this.particlePos[2 * i + 1];
                const xi = clamp(Math.floor(x * this.pInvSpacing), 0, this.pNumX - 1);
                const yi = clamp(Math.floor(y * this.pInvSpacing), 0, this.pNumY - 1);
                const cellNr = xi * this.pNumY + yi;
                this.numCellParticles[cellNr]++;
            }
            let first = 0;
            for (let i = 0; i < this.pNumCells; i++) {
                first += this.numCellParticles[i];
                this.firstCellParticle[i] = first;
            }
            this.firstCellParticle[this.pNumCells] = first;
            for (let i = 0; i < this.numParticles; i++) {
                const x = this.particlePos[2 * i];
                const y = this.particlePos[2 * i + 1];
                const xi = clamp(Math.floor(x * this.pInvSpacing), 0, this.pNumX - 1);
                const yi = clamp(Math.floor(y * this.pInvSpacing), 0, this.pNumY - 1);
                const cellNr = xi * this.pNumY + yi;
                this.firstCellParticle[cellNr]--;
                this.cellParticleIds[this.firstCellParticle[cellNr]] = i;
            }

            // Separation pass
            for (let i = 0; i < this.numParticles; i++) {
                const px = this.particlePos[2 * i];
                const py = this.particlePos[2 * i + 1];

                const pxi = Math.floor(px * this.pInvSpacing);
                const pyi = Math.floor(py * this.pInvSpacing);
                const x0 = Math.max(pxi - 1, 0);
                const y0 = Math.max(pyi - 1, 0);
                const x1 = Math.min(pxi + 1, this.pNumX - 1);
                const y1 = Math.min(pyi + 1, this.pNumY - 1);

                for (let xi = x0; xi <= x1; xi++) {
                    for (let yi = y0; yi <= y1; yi++) {
                        const cellNr = xi * this.pNumY + yi;
                        const cfirst = this.firstCellParticle[cellNr];
                        const clast = this.firstCellParticle[cellNr + 1];
                        for (let j = cfirst; j < clast; j++) {
                            const id = this.cellParticleIds[j];
                            if (id === i) continue;

                            let dx = this.particlePos[2 * id] - px;
                            let dy = this.particlePos[2 * id + 1] - py;
                            const d2 = dx * dx + dy * dy;
                            if (d2 > minDist2 || d2 === 0.0) continue;
                            const d = Math.sqrt(d2);
                            const s = 0.5 * (minDist - d) / d;
                            dx *= s;
                            dy *= s;
                            this.particlePos[2 * i] -= dx;
                            this.particlePos[2 * i + 1] -= dy;
                            this.particlePos[2 * id] += dx;
                            this.particlePos[2 * id + 1] += dy;

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

                            for (let k = 0; k < 3; k++) {
                                const color0 = this.particleColor[3 * i + k];
                                const color1 = this.particleColor[3 * id + k];
                                const color = (color0 + color1) * 0.5;
                                this.particleColor[3 * i + k] = color0 + (color - color0) * colorDiffusionCoeff;
                                this.particleColor[3 * id + k] = color1 + (color - color1) * colorDiffusionCoeff;
                            }
                        }
                    }
                }
            }

            // Wall-particle separation: push particles away from walls
            // using the same spacing rule as particle-particle pairs.
            // Wall is immovable so particle gets 100% of the correction.
            for (let i = 0; i < this.numParticles; i++) {
                // Bottom wall
                const distBot = this.particlePos[2 * i + 1] - bottomWall;
                if (distBot > 0 && distBot < halfDist) {
                    this.particlePos[2 * i + 1] += halfDist - distBot;
                }
                // Left wall
                const distLeft = this.particlePos[2 * i] - leftWall;
                if (distLeft > 0 && distLeft < halfDist) {
                    this.particlePos[2 * i] += halfDist - distLeft;
                }
                // Right wall
                const distRight = rightWall - this.particlePos[2 * i];
                if (distRight > 0 && distRight < halfDist) {
                    this.particlePos[2 * i] -= halfDist - distRight;
                }
                // Top wall
                const distTop = topWall - this.particlePos[2 * i + 1];
                if (distTop > 0 && distTop < halfDist) {
                    this.particlePos[2 * i + 1] -= halfDist - distTop;
                }
            }
        }
    }

    handleParticleCollisions(obstacleX: number, obstacleY: number, obstacleRadius: number, obstacleVelX: number, obstacleVelY: number) {
        const h = 1.0 / this.fInvSpacing;
        const r = this.particleRadius;
        const minDist = obstacleRadius + r;
        const minDist2 = minDist * minDist;

        const wallPad = r + this.particleGap * 0.5;
        const minX = h + wallPad;
        const maxX = (this.fNumX - 1) * h - wallPad;
        const minY = h + wallPad;
        const maxY = (this.fNumY - 1) * h - wallPad;

        for (let i = 0; i < this.numParticles; i++) {
            let x = this.particlePos[2 * i];
            let y = this.particlePos[2 * i + 1];

            const dx = x - obstacleX;
            const dy = y - obstacleY;
            const d2 = dx * dx + dy * dy;

            // Skip obstacle collision for immune particles (freshly spawned)
            if (this.particleImmune[i] > 0) {
                // immune: no obstacle interaction, still do wall collisions below
            } else if (d2 < minDist2) {
                const d = Math.sqrt(d2);

                // Unit normal: obstacle center → particle
                let nx: number, ny: number;
                if (d > 1e-8) {
                    nx = dx / d;
                    ny = dy / d;
                } else {
                    nx = 1.0; ny = 0.0;
                }

                // A) Project position to obstacle surface
                x = obstacleX + nx * minDist;
                y = obstacleY + ny * minDist;

                // B) Relative velocity (particle vs obstacle)
                const relVx = this.particleVel[2 * i] - obstacleVelX;
                const relVy = this.particleVel[2 * i + 1] - obstacleVelY;
                const relVn = relVx * nx + relVy * ny;

                // C) Strip inward normal component, keep tangential
                let outVx = relVx;
                let outVy = relVy;
                if (relVn < 0.0) {
                    outVx -= relVn * nx;
                    outVy -= relVn * ny;
                }

                // D) Outward repulsion proportional to penetration depth
                const overlap = minDist - d;
                const repulse = OBSTACLE_REPULSION * (overlap / minDist);
                outVx += repulse * nx;
                outVy += repulse * ny;

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

                // E) Convert back to world velocity
                this.particleVel[2 * i] = outVx + obstacleVelX;
                this.particleVel[2 * i + 1] = outVy + obstacleVelY;
            }

            // Speed-dependent wall bounce: slow → zero (no jitter), fast → bounce
            if (x < minX) {
                x = minX;
                const v = this.particleVel[2 * i];
                this.particleVel[2 * i] = Math.abs(v) < 0.3 ? 0 : v * -0.1;
            }
            if (x > maxX) {
                x = maxX;
                const v = this.particleVel[2 * i];
                this.particleVel[2 * i] = Math.abs(v) < 0.3 ? 0 : v * -0.1;
            }
            if (y < minY) {
                y = minY;
                const v = this.particleVel[2 * i + 1];
                this.particleVel[2 * i + 1] = Math.abs(v) < 0.3 ? 0 : v * -0.1;
            }
            if (y > maxY) {
                y = maxY;
                const v = this.particleVel[2 * i + 1];
                this.particleVel[2 * i + 1] = Math.abs(v) < 0.3 ? 0 : v * -0.1;
            }
            this.particlePos[2 * i] = x;
            this.particlePos[2 * i + 1] = y;
        }
    }

    updateParticleDensity() {
        const n = this.fNumY;
        const h = this.h;
        const h1 = this.fInvSpacing;
        const h2 = 0.5 * h;
        const d = this.particleDensity;

        d.fill(0.0);

        for (let i = 0; i < this.numParticles; i++) {
            let x = this.particlePos[2 * i];
            let y = this.particlePos[2 * i + 1];

            x = clamp(x, h, (this.fNumX - 1) * h);
            y = clamp(y, h, (this.fNumY - 1) * h);

            const x0 = Math.floor((x - h2) * h1);
            const tx = ((x - h2) - x0 * h) * h1;
            const x1 = Math.min(x0 + 1, this.fNumX - 2);

            const y0 = Math.floor((y - h2) * h1);
            const ty = ((y - h2) - y0 * h) * h1;
            const y1 = Math.min(y0 + 1, this.fNumY - 2);

            const sx = 1.0 - tx;
            const sy = 1.0 - ty;

            if (x0 < this.fNumX && y0 < this.fNumY) d[x0 * n + y0] += sx * sy;
            if (x1 < this.fNumX && y0 < this.fNumY) d[x1 * n + y0] += tx * sy;
            if (x1 < this.fNumX && y1 < this.fNumY) d[x1 * n + y1] += tx * ty;
            if (x0 < this.fNumX && y1 < this.fNumY) d[x0 * n + y1] += sx * ty;
        }

        if (this.particleRestDensity === 0.0) {
            let sum = 0.0;
            let numFluidCells = 0;

            for (let i = 0; i < this.fNumCells; i++) {
                if (this.cellType[i] === FLUID_CELL) {
                    sum += d[i];
                    numFluidCells++;
                }
            }

            if (numFluidCells > 0)
                this.particleRestDensity = sum / numFluidCells;
        }
    }

    transferVelocities(toGrid: boolean, flipRatio: number) {
        const n = this.fNumY;
        const h = this.h;
        const h1 = this.fInvSpacing;
        const h2 = 0.5 * h;

        if (toGrid) {
            this.prevU.set(this.u);
            this.prevV.set(this.v);

            this.du.fill(0.0);
            this.dv.fill(0.0);
            this.u.fill(0.0);
            this.v.fill(0.0);

            for (let i = 0; i < this.fNumCells; i++) {
                this.cellType[i] = this.s[i] === 0.0 ? SOLID_CELL : AIR_CELL;
            }

            for (let i = 0; i < this.numParticles; i++) {
                const x = this.particlePos[2 * i];
                const y = this.particlePos[2 * i + 1];
                const xi = clamp(Math.floor(x * h1), 0, this.fNumX - 1);
                const yi = clamp(Math.floor(y * h1), 0, this.fNumY - 1);
                const cellNr = xi * n + yi;
                if (this.cellType[cellNr] === AIR_CELL)
                    this.cellType[cellNr] = FLUID_CELL;
            }
        }

        for (let component = 0; component < 2; component++) {
            const dx = component === 0 ? 0.0 : h2;
            const dy = component === 0 ? h2 : 0.0;

            const f = component === 0 ? this.u : this.v;
            const prevF = component === 0 ? this.prevU : this.prevV;
            const d = component === 0 ? this.du : this.dv;

            for (let i = 0; i < this.numParticles; i++) {
                let x = this.particlePos[2 * i];
                let y = this.particlePos[2 * i + 1];

                x = clamp(x, h, (this.fNumX - 1) * h);
                y = clamp(y, h, (this.fNumY - 1) * h);

                const x0 = Math.max(0, Math.min(Math.floor((x - dx) * h1), this.fNumX - 2));
                const tx = clamp(((x - dx) - x0 * h) * h1, 0.0, 1.0);
                const x1 = Math.min(x0 + 1, this.fNumX - 2);

                const y0 = Math.max(0, Math.min(Math.floor((y - dy) * h1), this.fNumY - 2));
                const ty = clamp(((y - dy) - y0 * h) * h1, 0.0, 1.0);
                const y1 = Math.min(y0 + 1, this.fNumY - 2);

                const sx = 1.0 - tx;
                const sy = 1.0 - ty;

                const d0 = sx * sy;
                const d1 = tx * sy;
                const d2 = tx * ty;
                const d3 = sx * ty;

                const nr0 = x0 * n + y0;
                const nr1 = x1 * n + y0;
                const nr2 = x1 * n + y1;
                const nr3 = x0 * n + y1;

                if (toGrid) {
                    const pv = this.particleVel[2 * i + component];
                    f[nr0] += pv * d0;
                    d[nr0] += d0;
                    f[nr1] += pv * d1;
                    d[nr1] += d1;
                    f[nr2] += pv * d2;
                    d[nr2] += d2;
                    f[nr3] += pv * d3;
                    d[nr3] += d3;
                } else {
                    const offset = component === 0 ? n : 1;
                    const valid0 =
                        this.cellType[nr0] !== AIR_CELL ||
                            this.cellType[nr0 - offset] !== AIR_CELL
                            ? 1.0
                            : 0.0;
                    const valid1 =
                        this.cellType[nr1] !== AIR_CELL ||
                            this.cellType[nr1 - offset] !== AIR_CELL
                            ? 1.0
                            : 0.0;
                    const valid2 =
                        this.cellType[nr2] !== AIR_CELL ||
                            this.cellType[nr2 - offset] !== AIR_CELL
                            ? 1.0
                            : 0.0;
                    const valid3 =
                        this.cellType[nr3] !== AIR_CELL ||
                            this.cellType[nr3 - offset] !== AIR_CELL
                            ? 1.0
                            : 0.0;

                    const v = this.particleVel[2 * i + component];
                    const denom = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;

                    if (denom > 0.0) {
                        const picV =
                            (valid0 * d0 * f[nr0] +
                                valid1 * d1 * f[nr1] +
                                valid2 * d2 * f[nr2] +
                                valid3 * d3 * f[nr3]) /
                            denom;
                        const corr =
                            (valid0 * d0 * (f[nr0] - prevF[nr0]) +
                                valid1 * d1 * (f[nr1] - prevF[nr1]) +
                                valid2 * d2 * (f[nr2] - prevF[nr2]) +
                                valid3 * d3 * (f[nr3] - prevF[nr3])) /
                            denom;
                        const flipV = v + corr;

                        this.particleVel[2 * i + component] =
                            (1.0 - flipRatio) * picV + flipRatio * flipV;
                    }
                }

            }

            if (toGrid) {
                for (let i = 0; i < f.length; i++) {
                    if (d[i] > 0.0)
                        f[i] /= d[i];
                }
                for (let i = 0; i < this.fNumX; i++) {
                    for (let j = 0; j < this.fNumY; j++) {
                        const idx = i * n + j;
                        const solid = this.cellType[idx] === SOLID_CELL;
                        if (solid || (i > 0 && this.cellType[(i - 1) * n + j] === SOLID_CELL)) {
                            const wallFace = i <= 1 || i >= this.fNumX - 1 || j === 0;
                            this.u[idx] = wallFace ? 0.0 : this.prevU[idx];
                        }
                        if (solid || (j > 0 && this.cellType[i * n + j - 1] === SOLID_CELL)) {
                            const wallFace = i === 0 || i >= this.fNumX - 1 || j <= 1;
                            this.v[idx] = wallFace ? 0.0 : this.prevV[idx];
                        }
                    }
                }
            }
        }
    }

    solveIncompressibility(numIters: number, dt: number, overRelaxation: number, compensateDrift = true) {
        this.p.fill(0.0);
        this.prevU.set(this.u);
        this.prevV.set(this.v);

        const n = this.fNumY;
        const cp = this.density * this.h / dt;

        for (let iter = 0; iter < numIters; iter++) {
            for (let i = 1; i < this.fNumX - 1; i++) {
                for (let j = 1; j < this.fNumY - 1; j++) {

                    if (this.cellType[i * n + j] !== FLUID_CELL)
                        continue;

                    const center = i * n + j;
                    const left = (i - 1) * n + j;
                    const right = (i + 1) * n + j;
                    const bottom = i * n + j - 1;
                    const top = i * n + j + 1;

                    const sx0 = this.s[left];
                    const sx1 = this.s[right];
                    const sy0 = this.s[bottom];
                    const sy1 = this.s[top];
                    const s = sx0 + sx1 + sy0 + sy1;
                    if (s === 0.0)
                        continue;

                    let div = this.u[right] - this.u[center] +
                        this.v[top] - this.v[center];

                    if (this.particleRestDensity > 0.0 && compensateDrift) {
                        const k = 1.0;
                        const compression = this.particleDensity[i * n + j] - this.particleRestDensity;
                        if (compression > 0.0)
                            div = div - k * compression;
                    }

                    let p = -div / s;
                    p *= overRelaxation;
                    this.p[center] += cp * p;

                    this.u[center] -= sx0 * p;
                    this.u[right] += sx1 * p;
                    this.v[center] -= sy0 * p;
                    this.v[top] += sy1 * p;
                }
            }
        }
    }

    dampVelocities(damping: number) {
        const scale = 1.0 - damping;
        for (let i = 0; i < this.numParticles; i++) {
            this.particleVel[2 * i] *= scale;
            this.particleVel[2 * i + 1] *= scale;
        }
    }

    clampVelocities(maxSpeed: number) {
        const maxSpeed2 = maxSpeed * maxSpeed;
        for (let i = 0; i < this.numParticles; i++) {
            const vx = this.particleVel[2 * i];
            const vy = this.particleVel[2 * i + 1];
            const speed2 = vx * vx + vy * vy;
            if (speed2 > maxSpeed2) {
                const s = maxSpeed / Math.sqrt(speed2);
                this.particleVel[2 * i] *= s;
                this.particleVel[2 * i + 1] *= s;
            }
        }
    }

    dampSettledParticles(threshold: number, maxDamp: number) {
        const threshold2 = threshold * threshold;
        for (let i = 0; i < this.numParticles; i++) {
            const vx = this.particleVel[2 * i];
            const vy = this.particleVel[2 * i + 1];
            const s2 = vx * vx + vy * vy;
            if (s2 < threshold2 && s2 > 0) {
                const speed = Math.sqrt(s2);
                const t = 1.0 - speed / threshold; // 1.0 at speed=0, 0.0 at threshold
                const damp = 1.0 - maxDamp * t;
                this.particleVel[2 * i] *= damp;
                this.particleVel[2 * i + 1] *= damp;
            }
        }
    }

    applyFallingCohesion() {
        const minDist = 2.0 * this.particleRadius + this.particleGap;
        const recruitRadius = 4.0 * minDist;
        const recruitRadius2 = recruitRadius * recruitRadius;
        const h = 1.0 / this.fInvSpacing;
        const fallingSpeed2 = 0.05 * 0.05;
        const settledSpeed2 = 0.10 * 0.10;
        const blendRate = 0.15;
        const decayRate = 0.008;
        const recruitImpulseFrac = 0.35;

        for (let i = 0; i < this.numParticles; i++) {
            if (this.particleCohesion[i] <= 0) continue;

            const vx = this.particleVel[2 * i];
            const vy = this.particleVel[2 * i + 1];
            const speed2 = vx * vx + vy * vy;

            // Re-settled: clear cohesion
            if (speed2 < fallingSpeed2) {
                this.particleCohesion[i] = 0;
                continue;
            }

            const px = this.particlePos[2 * i];
            const py = this.particlePos[2 * i + 1];

            // Search neighbors via spatial hash (from last pushParticlesApart)
            const pxi = Math.floor(px * this.pInvSpacing);
            const pyi = Math.floor(py * this.pInvSpacing);
            const x0 = Math.max(pxi - 1, 0);
            const y0 = Math.max(pyi - 1, 0);
            const x1 = Math.min(pxi + 1, this.pNumX - 1);
            const y1 = Math.min(pyi + 1, this.pNumY - 1);

            let grpVx = vx, grpVy = vy, grpCount = 1;

            for (let xi = x0; xi <= x1; xi++) {
                for (let yi = y0; yi <= y1; yi++) {
                    const cellNr = xi * this.pNumY + yi;
                    const cfirst = this.firstCellParticle[cellNr];
                    const clast = this.firstCellParticle[cellNr + 1];
                    for (let j = cfirst; j < clast; j++) {
                        const id = this.cellParticleIds[j];
                        if (id === i) continue;

                        const ddx = this.particlePos[2 * id] - px;
                        const ddy = this.particlePos[2 * id + 1] - py;
                        const dd2 = ddx * ddx + ddy * ddy;
                        if (dd2 > recruitRadius2) continue;

                        // Same horizontal band check (within h)
                        if (Math.abs(ddy) > h) continue;

                        const nvx = this.particleVel[2 * id];
                        const nvy = this.particleVel[2 * id + 1];
                        const nspeed2 = nvx * nvx + nvy * nvy;

                        if (this.particleCohesion[id] > 0) {
                            // Both have cohesion: accumulate for group blend
                            grpVx += nvx;
                            grpVy += nvy;
                            grpCount++;
                        } else if (nspeed2 < settledSpeed2) {
                            // Recruit settled neighbor into group
                            this.particleCohesion[id] = 0.5;
                            this.particleVel[2 * id] += vx * recruitImpulseFrac;
                            this.particleVel[2 * id + 1] += vy * recruitImpulseFrac;
                        }
                    }
                }
            }

            // Blend velocity toward group average
            if (grpCount > 1) {
                const avgVx = grpVx / grpCount;
                const avgVy = grpVy / grpCount;
                this.particleVel[2 * i] += (avgVx - vx) * blendRate;
                this.particleVel[2 * i + 1] += (avgVy - vy) * blendRate;
            }

            // Decay cohesion
            this.particleCohesion[i] -= decayRate;
            if (this.particleCohesion[i] < 0) this.particleCohesion[i] = 0;
        }
    }

    applyIdleWave(simTime: number, dt: number, strength: number, frequency: number, noise: number) {
        for (let i = 0; i < this.numParticles; i++) {
            const x = this.particlePos[2 * i];
            const y = this.particlePos[2 * i + 1];

            // Phase varies with position for natural wave appearance
            const phase = frequency * simTime + 2.5 * y - 0.3 * x;

            // Horizontal wave (primary) applied as acceleration
            const waveX = strength * Math.sin(phase);
            // Subtle vertical response (secondary, phase-shifted)
            const waveY = strength * 0.2 * Math.cos(phase * 1.3 + 0.7);

            this.particleVel[2 * i] += waveX * dt;
            this.particleVel[2 * i + 1] += waveY * dt;

            // Tiny noise to break uniformity
            if (noise > 0) {
                this.particleVel[2 * i] += (Math.random() - 0.5) * noise * dt;
                this.particleVel[2 * i + 1] += (Math.random() - 0.5) * noise * dt * 0.3;
            }
        }
    }

    updateParticleColors() {
        for (let i = 0; i < this.numParticles; i++) {
            this.particleColor[3 * i] = 1.0;
            this.particleColor[3 * i + 1] = 1.0;
            this.particleColor[3 * i + 2] = 1.0;
        }
    }

    swapRemoveParticle(i: number) {
        const last = this.numParticles - 1;
        if (i < last) {
            this.particlePos[2 * i] = this.particlePos[2 * last];
            this.particlePos[2 * i + 1] = this.particlePos[2 * last + 1];
            this.particleVel[2 * i] = this.particleVel[2 * last];
            this.particleVel[2 * i + 1] = this.particleVel[2 * last + 1];
            this.particleColor[3 * i] = this.particleColor[3 * last];
            this.particleColor[3 * i + 1] = this.particleColor[3 * last + 1];
            this.particleColor[3 * i + 2] = this.particleColor[3 * last + 2];
            this.particleShape[i] = this.particleShape[last];
            this.particleAngle[i] = this.particleAngle[last];
            this.particleAngVel[i] = this.particleAngVel[last];
            this.particleImmune[i] = this.particleImmune[last];
            this.particleCohesion[i] = this.particleCohesion[last];
        }
        this.numParticles--;
    }

    simulate(dt: number, gravity: number, flipRatio: number, numPressureIters: number, numParticleIters: number, overRelaxation: number, compensateDrift: boolean, separateParticles: boolean, obstacleX: number, obstacleY: number, obstacleRadius: number, obstacleVelX: number, obstacleVelY: number, tiltForceX: number = 0, tiltForceY: number = 0) {
        const numSubSteps = 1;
        const sdt = dt / numSubSteps;

        for (let step = 0; step < numSubSteps; step++) {
            this.integrateParticles(sdt, gravity, tiltForceX, tiltForceY);
            if (separateParticles)
                this.pushParticlesApart(numParticleIters);
            this.handleParticleCollisions(obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY);
            this.transferVelocities(true, flipRatio);
            this.updateParticleDensity();
            this.solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift);
            this.transferVelocities(false, flipRatio);
            this.dampVelocities(0.001);
            this.dampSettledParticles(0.15, 0.03);
            this.clampVelocities(15.0);
        }

        // Decrement immunity counters
        for (let i = 0; i < this.numParticles; i++) {
            if (this.particleImmune[i] > 0) this.particleImmune[i]--;
        }

        this.updateParticleColors();
    }
}

// ----------------- end of simulator ------------------------------

function compile(gl: WebGL2RenderingContext, t: number, s: string) {
    const sh = gl.createShader(t)!;
    gl.shaderSource(sh, s); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const l = gl.getShaderInfoLog(sh); gl.deleteShader(sh);
        throw new Error(`Shader: ${l}`);
    }
    return sh;
}

// ═══ Factory ════════════════════════════════════════════════════
export function createFluidPool(
    wrapper: HTMLDivElement,
    canvas: HTMLCanvasElement,
    opts?: PoolOptions,
): PoolAPI {
    // ── Setup Scene Options ──
    const scene = {
        gravity: opts?.gravity !== undefined ? opts.gravity : -9.81,
        dt: opts?.dt ?? 1.0 / 60.0,
        flipRatio: opts?.flipRatio ?? 0.90,
        numPressureIters: opts?.pressureIters ?? 80,
        numParticleIters: opts?.sepIters ?? 3,
        overRelaxation: opts?.overRelax ?? 1.9,
        compensateDrift: true,
        separateParticles: true,
        obstacleX: 0.0,
        obstacleY: 0.0,
        obstacleRadius: 0.25,
        obstacleVelX: 0.0,
        obstacleVelY: 0.0,
        paused: false,
        frameNr: 0,
        showDebug: false,
        simTime: 0.0,
        idleWaveEnabled: true,
        idleWaveStrength: 0.60,
        idleWaveFrequency: 3.2,
        idleWaveNoise: 0.06,
        tiltForceX: 0.0,
        tiltForceY: 0.0,
    };

    const simHeight = 3.0;
    const MAX_DPR = 2.0;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    const cScale = (canvas.clientHeight || window.innerHeight) / simHeight;
    const simWidth = (canvas.clientWidth || window.innerWidth) / cScale;
    // Particle size is controlled via res: lower res → bigger h → bigger r.
    // r/h = 0.3 (reference ratio) keeps FLIP grid coupling stable.
    const TARGET_RADIUS_PX = 7;  // desired rendered particle radius in CSS px
    const GAP_PX = 4;            // desired visible gap between particle edges
    const res = Math.round(0.3 * (canvas.clientHeight || window.innerHeight) / TARGET_RADIUS_PX);

    const tankHeight = 1.0 * simHeight;
    const tankWidth = 1.0 * simWidth;
    const h = tankHeight / res;
    const density = 1000.0;

    // Mobile detection (used for fill height + touch listener gating)
    const isMobile = 'ontouchstart' in window || window.matchMedia('(pointer: coarse)').matches;

    // initial fill: on mobile spawn full-width so settled pool ≈ 50% height with no gap
    const relWaterHeight = isMobile ? 0.45 : 0.50;
    const relWaterWidth  = isMobile ? 1.0  : 0.60;

    const gapSim = GAP_PX / cScale;

    // compute number of particles — r/h = 0.3 matches reference for stable FLIP
    const r = 0.3 * h;
    const minDistSpawn = 2.0 * r + gapSim;
    const dx = minDistSpawn;
    const dy = Math.sqrt(3.0) / 2.0 * dx;

    const wallPad = r + gapSim * 0.5;
    const numX = Math.round((relWaterWidth * tankWidth - 2.0 * h - 2.0 * wallPad) / dx);
    const numY = Math.round((relWaterHeight * tankHeight - 2.0 * h - 2.0 * wallPad) / dy);

    // Geometry-based particle targets: full-tank hex capacity × fill fraction
    const fullNumX = Math.round((tankWidth - 2.0 * h - 2.0 * wallPad) / dx);
    const fullNumY = Math.round((tankHeight - h - 2.0 * wallPad) / dy);
    const tankCapacity = fullNumX * fullNumY;
    const TARGET_FILL = 0.45;
    const TARGET_PARTICLES = Math.floor(TARGET_FILL * tankCapacity);
    const maxParticles = Math.ceil(TARGET_PARTICLES * 2.5);

    // create fluid
    const f = new FlipFluid(density, tankWidth, tankHeight, h, r, gapSim, maxParticles);

    // create particles
    const initialParticles = numX * numY;
    f.numParticles = initialParticles;
    let spawnIndex = initialParticles;
    const SPREAD_FRAMES = 30;                 // spread excess removal over ~0.5s
    const NUM_BINS = 16;                      // x-axis bins for distributed removal
    let pendingRemoval = 0;                   // total particles still queued for removal
    let removalBatchSize = 1;                 // particles removed per frame during cleanup
    let p = 0;
    const jitter = 0.2 * r;
    for (let i = 0; i < numX; i++) {
        for (let j = 0; j < numY; j++) {
            f.particlePos[p++] = h + wallPad + dx * i + (j % 2 === 0 ? 0.0 : 0.5 * dx)
                + (Math.random() - 0.5) * jitter;
            f.particlePos[p++] = h + wallPad + dy * j
                + (Math.random() - 0.5) * jitter;
        }
    }

    // setup grid cells for tank
    const n = f.fNumY;
    for (let i = 0; i < f.fNumX; i++) {
        for (let j = 0; j < f.fNumY; j++) {
            let s = 1.0;
            if (i === 0 || i === f.fNumX - 1 || j === 0 || j === f.fNumY - 1)
                s = 0.0;
            f.s[i * n + j] = s;
        }
    }

    // ── View bounds: flush with particle collision boundaries ──
    // Particles are confined to [h + wallPad, (fNumY-1)*h - wallPad] vertically.
    // Set view top = topmost particle visual edge so there's no gap at the canvas top.
    const collisionPad = r + gapSim * 0.5; // same wallPad used in handleParticleCollisions
    const viewableTop = (f.fNumY - 1) * f.h - collisionPad + r; // top particle visual edge
    const viewableSimHeight = viewableTop - f.h;

    let viewLeft = f.h;
    let viewBottom = f.h;
    const viewRight = (f.fNumX - 1) * f.h;
    let viewWidth = viewRight - viewLeft;
    let viewHeight = viewableSimHeight;

    console.log(`[FluidPool] init: ${f.numParticles} particles, grid ${f.fNumX}x${f.fNumY}, h=${f.h.toFixed(4)}, r=${r.toFixed(4)}, viewW=${viewWidth.toFixed(3)}, viewH=${viewHeight.toFixed(3)}, canvas=${canvas.clientWidth}x${canvas.clientHeight}`);

    // ── WebGL2 Rendering Setup ──
    const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
    if (!gl) {
        console.warn('No WebGL2');
        return { cleanup() { }, reset() { }, setFlipRatio() { }, setPaused() { }, setShowDebug() { }, setTiltForce() { } };
    }

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs); gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(prog);
        console.error('[FluidPool] Particle program link failed:', log);
    }

    const uRes = gl.getUniformLocation(prog, 'u_res')!;
    const uOff = gl.getUniformLocation(prog, 'u_offset')!;
    const uPt = gl.getUniformLocation(prog, 'u_ptSz')!;
    const uWaveTime = gl.getUniformLocation(prog, 'u_waveTime')!;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const pBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, pBuf);
    gl.bufferData(gl.ARRAY_BUFFER, f.particlePos.byteLength, gl.DYNAMIC_DRAW);
    const aP = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(aP);
    gl.vertexAttribPointer(aP, 2, gl.FLOAT, false, 0, 0);

    const cBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuf);
    gl.bufferData(gl.ARRAY_BUFFER, f.particleColor.byteLength, gl.DYNAMIC_DRAW);
    const aC = gl.getAttribLocation(prog, 'a_color');
    gl.enableVertexAttribArray(aC);
    gl.vertexAttribPointer(aC, 3, gl.FLOAT, false, 0, 0);

    const sBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, sBuf);
    gl.bufferData(gl.ARRAY_BUFFER, f.particleShape.byteLength, gl.DYNAMIC_DRAW);
    const aS = gl.getAttribLocation(prog, 'a_shape');
    gl.enableVertexAttribArray(aS);
    gl.vertexAttribPointer(aS, 1, gl.FLOAT, false, 0, 0);

    const angBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, angBuf);
    gl.bufferData(gl.ARRAY_BUFFER, f.particleAngle.byteLength, gl.DYNAMIC_DRAW);
    const aA = gl.getAttribLocation(prog, 'a_angle');
    gl.enableVertexAttribArray(aA);
    gl.vertexAttribPointer(aA, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Mesh setup for obstacle
    const m_vs = compile(gl, gl.VERTEX_SHADER, MESH_VERT);
    const m_fs = compile(gl, gl.FRAGMENT_SHADER, MESH_FRAG);
    const m_prog = gl.createProgram()!;
    gl.attachShader(m_prog, m_vs); gl.attachShader(m_prog, m_fs);
    gl.linkProgram(m_prog);
    if (!gl.getProgramParameter(m_prog, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(m_prog);
        console.error('[FluidPool] Mesh program link failed:', log);
    }

    const m_uRes = gl.getUniformLocation(m_prog, 'u_res')!;
    const m_uOff = gl.getUniformLocation(m_prog, 'u_offset')!;
    const m_uCol = gl.getUniformLocation(m_prog, 'u_color')!;
    const m_uTrans = gl.getUniformLocation(m_prog, 'u_translation')!;
    const m_uScale = gl.getUniformLocation(m_prog, 'u_scale')!;

    // Create disk mesh
    const numSegs = 50;
    const diskVerts = new Float32Array(2 * numSegs + 2);
    let vp = 0;
    diskVerts[vp++] = 0.0; diskVerts[vp++] = 0.0;
    const dphi = 2.0 * Math.PI / numSegs;
    for (let i = 0; i < numSegs; i++) {
        diskVerts[vp++] = Math.cos(i * dphi);
        diskVerts[vp++] = Math.sin(i * dphi);
    }
    const dBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, dBuf);
    gl.bufferData(gl.ARRAY_BUFFER, diskVerts, gl.STATIC_DRAW);

    const diskIds = new Uint16Array(3 * numSegs);
    let ip = 0;
    for (let i = 0; i < numSegs; i++) {
        diskIds[ip++] = 0;
        diskIds[ip++] = 1 + i;
        diskIds[ip++] = 1 + (i + 1) % numSegs;
    }
    const dIBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, dIBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, diskIds, gl.STATIC_DRAW);

    const mVao = gl.createVertexArray()!;
    gl.bindVertexArray(mVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, dBuf);
    const amP = gl.getAttribLocation(m_prog, 'a_pos');
    gl.enableVertexAttribArray(amP);
    gl.vertexAttribPointer(amP, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, dIBuf);
    gl.bindVertexArray(null);

    // ── Interaction ──
    const rms: (() => void)[] = [];
    let dead = false;

    const listen = (el: EventTarget, t: string, fn: EventListener, p?: boolean) => {
        const o = p ? { passive: true } as AddEventListenerOptions : undefined;
        el.addEventListener(t, fn, o);
        rms.push(() => el.removeEventListener(t, fn, o));
    };

    const MOUSE_MAX_SPEED = 25.0; // clamp crazy interactions

    function setObstacle(x: number, y: number, reset: boolean) {
        let vx = 0.0;
        let vy = 0.0;
        if (!reset) {
            vx = (x - scene.obstacleX) / scene.dt;
            vy = (y - scene.obstacleY) / scene.dt;

            // Limit maximum obstacle speed to prevent physics explosions
            const speed = Math.sqrt(vx * vx + vy * vy);
            if (speed > MOUSE_MAX_SPEED) {
                vx = (vx / speed) * MOUSE_MAX_SPEED;
                vy = (vy / speed) * MOUSE_MAX_SPEED;
            }
        }

        scene.obstacleX = x;
        scene.obstacleY = y;
        const R = scene.obstacleRadius;
        const n = f.fNumY;

        for (let i = 1; i < f.fNumX - 2; i++) {
            for (let j = 1; j < f.fNumY - 2; j++) {
                f.s[i * n + j] = 1.0;
                const cdx = (i + 0.5) * f.h - x;
                const cdy = (j + 0.5) * f.h - y;
                if (cdx * cdx + cdy * cdy < R * R) {
                    f.s[i * n + j] = 0.0;
                    f.u[i * n + j] = vx;
                    f.u[(i + 1) * n + j] = vx;
                    f.v[i * n + j] = vy;
                    f.v[i * n + j + 1] = vy;
                }
            }
        }

        scene.obstacleVelX = vx;
        scene.obstacleVelY = vy;
    }

    // ── Spawn queue for timed wave/stream emission ──
    const SPAWN_BATCH_SIZE = 18;       // particles emitted per frame
    const SPAWN_IMMUNE_FRAMES = 90;    // ~1.5s of obstacle immunity
    let spawnQueue: {
        x: number; y: number;
        remaining: number;
        emitted: number;
        totalCount: number;
        phase: number;
    } | null = null;

    function startSpawn(cx: number, cy: number) {
        const rc = canvas.getBoundingClientRect();
        const mx = cx - rc.left;
        const my = cy - rc.top;
        const x = viewLeft + (mx / canvas.clientWidth) * viewWidth;
        const y = viewBottom + ((canvas.clientHeight - my) / canvas.clientHeight) * viewHeight;

        const baseDropCount = 250;
        let dropCount = baseDropCount;
        if (f.numParticles / f.maxParticles > SPAWN_THROTTLE_THRESHOLD) {
            dropCount = Math.floor(baseDropCount * SPAWN_THROTTLE_FACTOR);
        }

        // Proactive cleanup: remove one settled bottom layer per click
        removeBottomLayer();

        spawnQueue = {
            x, y,
            remaining: dropCount,
            emitted: 0,
            totalCount: dropCount,
            phase: Math.random() * Math.PI * 2,
        };
    }

    function emitSpawnBatch() {
        if (!spawnQueue || spawnQueue.remaining <= 0) {
            spawnQueue = null;
            return;
        }

        const q = spawnQueue;
        const batch = Math.min(SPAWN_BATCH_SIZE, q.remaining);
        const colWidth = 0.08;
        const colHeight = 0.6;
        const waveAmp = 0.03;
        const waveFreq = 12.0;

        for (let b = 0; b < batch; b++) {
            let idx = f.numParticles;
            if (f.numParticles >= f.maxParticles) {
                const poolSize = f.maxParticles - initialParticles;
                if (poolSize <= 0) break;
                idx = initialParticles + ((spawnIndex - initialParticles) % poolSize);
            } else {
                f.numParticles++;
            }
            spawnIndex++;

            const t = q.emitted / q.totalCount;
            const yOff = -t * colHeight;
            const xWave = waveAmp * Math.sin(waveFreq * t + q.phase);
            const jitterX = (Math.random() - 0.5) * colWidth;
            const jitterY = (Math.random() - 0.5) * (colHeight / q.totalCount) * 2;

            f.particlePos[2 * idx] = q.x + xWave + jitterX;
            f.particlePos[2 * idx + 1] = q.y + yOff + jitterY;
            f.particleVel[2 * idx] = (Math.random() - 0.5) * 0.3;
            f.particleVel[2 * idx + 1] = -1.5 - Math.random() * 1.5;
            f.particleShape[idx] = q.emitted % 4;
            f.particleAngle[idx] = Math.random() * Math.PI * 2;
            f.particleAngVel[idx] = (0.3 + Math.random() * 0.7) * (Math.random() < 0.5 ? 1 : -1);
            f.particleImmune[idx] = SPAWN_IMMUNE_FRAMES;
            f.particleCohesion[idx] = 0.0;

            q.emitted++;
            q.remaining--;
        }

        // Queue excess for distributed removal
        const excess = Math.max(0, f.numParticles - TARGET_PARTICLES);
        if (excess > pendingRemoval) {
            pendingRemoval = excess;
            removalBatchSize = Math.max(1, Math.ceil(excess / SPREAD_FRAMES));
        }
    }

    // Remove `count` bottom-most particles distributed across x-bins via swap-remove.
    function removeBottomDistributed(count: number) {
        if (count <= 0 || f.numParticles <= 0) return;

        const tankL = f.h;
        const tankR = (f.fNumX - 1) * f.h;
        const binW = (tankR - tankL) / NUM_BINS;

        // Assign particles to x-bins
        const bins: number[][] = [];
        for (let b = 0; b < NUM_BINS; b++) bins.push([]);
        for (let i = 0; i < f.numParticles; i++) {
            const b = clamp(Math.floor((f.particlePos[2 * i] - tankL) / binW), 0, NUM_BINS - 1);
            bins[b].push(i);
        }

        // Sort each bin by y ascending (bottom-most first)
        for (const bin of bins) {
            bin.sort((a, b) => f.particlePos[2 * a + 1] - f.particlePos[2 * b + 1]);
        }

        // Round-robin across bins: take one bottom particle per bin per cycle
        const toRemove: number[] = [];
        const ptr = new Array(NUM_BINS).fill(0);
        while (toRemove.length < count) {
            let addedThisCycle = false;
            for (let b = 0; b < NUM_BINS && toRemove.length < count; b++) {
                if (ptr[b] < bins[b].length) {
                    toRemove.push(bins[b][ptr[b]]);
                    ptr[b]++;
                    addedThisCycle = true;
                }
            }
            if (!addedThisCycle) break;
        }

        // Swap-remove in descending index order (required for correctness)
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            if (idx < f.numParticles) f.swapRemoveParticle(idx);
        }
    }

    // Remove all non-immune particles within one dy band from the bottom. Returns count removed.
    function removeBottomLayer(): number {
        if (f.numParticles <= 0) return 0;

        // Pass 1: find minY among non-immune settled particles
        let minY = Infinity;
        for (let i = 0; i < f.numParticles; i++) {
            if (f.particleImmune[i] > 0) continue;
            const y = f.particlePos[2 * i + 1];
            if (y < minY) minY = y;
        }
        if (minY === Infinity) return 0;

        // Pass 2: collect indices in bottom layer band [minY, minY + dy*0.5]
        const threshold = minY + dy * 0.5;
        const toRemove: number[] = [];
        for (let i = 0; i < f.numParticles; i++) {
            if (f.particleImmune[i] > 0) continue;
            if (f.particlePos[2 * i + 1] <= threshold) {
                toRemove.push(i);
            }
        }

        // Swap-remove in descending index order
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            if (idx < f.numParticles) f.swapRemoveParticle(idx);
        }

        return toRemove.length;
    }

    function handlePointerMove(cx: number, cy: number) {
        const rc = canvas.getBoundingClientRect();
        const mx = cx - rc.left;
        const my = cy - rc.top;
        const x = viewLeft + (mx / canvas.clientWidth) * viewWidth;
        const y = viewBottom + ((canvas.clientHeight - my) / canvas.clientHeight) * viewHeight;
        setObstacle(x, y, false);
    }

    function handlePointerLeave() {
        scene.obstacleVelX = 0.0;
        scene.obstacleVelY = 0.0;
        setObstacle(-10, -10, true);
    }

    // Set obstacle far away initially
    setObstacle(-10, -10, true);

    // On mobile, no pointer/touch/mouse listeners — tilt is the sole interaction.
    // On desktop, attach all mouse + touch listeners for spawn/drag.
    if (!isMobile) {
        listen(wrapper, 'mousedown', ((e: MouseEvent) => {
            startSpawn(e.clientX, e.clientY);
        }) as EventListener);
        listen(wrapper, 'mousemove', ((e: MouseEvent) => handlePointerMove(e.clientX, e.clientY)) as EventListener);
        listen(wrapper, 'mouseleave', handlePointerLeave as EventListener);
        listen(window, 'mouseup', handlePointerLeave as EventListener);

        listen(wrapper, 'touchstart', ((e: TouchEvent) => {
            e.preventDefault();
            handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
            startSpawn(e.touches[0].clientX, e.touches[0].clientY);
        }) as EventListener, false);
        listen(wrapper, 'touchmove', ((e: TouchEvent) => {
            e.preventDefault();
            handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
        }) as EventListener, false);
        listen(wrapper, 'touchend', ((e: TouchEvent) => {
            e.preventDefault();
            handlePointerLeave();
        }) as EventListener, false);
    }

    // ── Loop ──
    let rafId = 0;
    function loop() {
        try {
            if (!dead) {
                if (scene.frameNr === 0) {
                    const cvs = gl!.canvas as HTMLCanvasElement;
                    console.log(`[FluidPool] first frame: canvas ${cvs.width}x${cvs.height}, clientW=${cvs.clientWidth}, clientH=${cvs.clientHeight}, dpr=${dpr}`);
                }
                if (!scene.paused) {
                    f.simulate(
                        scene.dt, scene.gravity, scene.flipRatio, scene.numPressureIters, scene.numParticleIters,
                        scene.overRelaxation, scene.compensateDrift, scene.separateParticles,
                        scene.obstacleX, scene.obstacleY, scene.obstacleRadius, scene.obstacleVelX, scene.obstacleVelY,
                        scene.tiltForceX, scene.tiltForceY
                    );

                    // Group falling cohesion (before idle wave)
                    f.applyFallingCohesion();

                    // Idle wave: gentle coordinated sway (after simulate so pressure solver doesn't cancel it)
                    if (scene.idleWaveEnabled) {
                        f.applyIdleWave(scene.simTime, scene.dt,
                            scene.idleWaveStrength, scene.idleWaveFrequency, scene.idleWaveNoise);
                    }

                    scene.simTime += scene.dt;

                    // Process spawn queue (timed emission)
                    emitSpawnBatch();

                    // Update particle rotation angles
                    for (let i = 0; i < f.numParticles; i++) {
                        f.particleAngle[i] += f.particleAngVel[i] * scene.dt;
                    }

                    scene.frameNr++;

                    // Distributed bottom removal (only active after a click-spawn)
                    if (pendingRemoval > 0) {
                        const batch = Math.min(pendingRemoval, removalBatchSize);
                        removeBottomDistributed(batch);
                        pendingRemoval -= batch;
                    }

                    // Debug: highlight particles near obstacle
                    if (scene.showDebug && scene.obstacleX > -5.0) {
                        const debugR2 = (scene.obstacleRadius + f.particleRadius) ** 2;
                        let maxOverlap = 0, nearCount = 0;
                        for (let i = 0; i < f.numParticles; i++) {
                            const ddx = f.particlePos[2 * i] - scene.obstacleX;
                            const ddy = f.particlePos[2 * i + 1] - scene.obstacleY;
                            const dist2 = ddx * ddx + ddy * ddy;
                            if (dist2 < debugR2) {
                                f.particleColor[3 * i] = 1.0;
                                f.particleColor[3 * i + 1] = 0.3;
                                f.particleColor[3 * i + 2] = 0.0;
                                nearCount++;
                                const overlap = Math.sqrt(debugR2) - Math.sqrt(dist2);
                                if (overlap > maxOverlap) maxOverlap = overlap;
                            }
                        }
                        if (scene.frameNr % 120 === 0 && nearCount > 0) {
                            console.log(`[DEBUG] ${nearCount} particles in obstacle zone, max overlap: ${maxOverlap.toFixed(4)}`);
                        }
                    }
                }

                // Render
                const cvs = gl!.canvas as HTMLCanvasElement;
                const targetW = Math.round(cvs.clientWidth * dpr);
                const targetH = Math.round(cvs.clientHeight * dpr);
                if (cvs.width !== targetW || cvs.height !== targetH) {
                    cvs.width = targetW;
                    cvs.height = targetH;
                    const physPerPx = viewableSimHeight / cvs.clientHeight;
                    viewWidth = cvs.clientWidth * physPerPx;
                    viewHeight = cvs.clientHeight * physPerPx;
                    viewLeft = f.h;
                    viewBottom = f.h;
                }
                gl!.viewport(0, 0, cvs.width, cvs.height);
                gl!.clearColor(0, 0, 0, 0);
                gl!.clear(gl!.COLOR_BUFFER_BIT | gl!.DEPTH_BUFFER_BIT);
                gl!.enable(gl!.BLEND);
                gl!.blendFunc(gl!.SRC_ALPHA, gl!.ONE_MINUS_SRC_ALPHA);

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

                const basePtSz = 2.0 * f.particleRadius / viewWidth * cvs.width;
                const ptSz = isMobile ? basePtSz - 2 * dpr : basePtSz;

                gl!.useProgram(prog);
                gl!.uniform2f(uRes, viewWidth, viewHeight);
                gl!.uniform2f(uOff, viewLeft, viewBottom);
                gl!.uniform1f(uPt, ptSz);
                gl!.uniform1f(uWaveTime, scene.simTime);

                gl!.bindVertexArray(vao);
                gl!.bindBuffer(gl!.ARRAY_BUFFER, pBuf);
                gl!.bufferData(gl!.ARRAY_BUFFER, f.particlePos, gl!.DYNAMIC_DRAW);
                gl!.bindBuffer(gl!.ARRAY_BUFFER, cBuf);
                gl!.bufferData(gl!.ARRAY_BUFFER, f.particleColor, gl!.DYNAMIC_DRAW);
                gl!.bindBuffer(gl!.ARRAY_BUFFER, sBuf);
                gl!.bufferData(gl!.ARRAY_BUFFER, f.particleShape, gl!.DYNAMIC_DRAW);
                gl!.bindBuffer(gl!.ARRAY_BUFFER, angBuf);
                gl!.bufferData(gl!.ARRAY_BUFFER, f.particleAngle, gl!.DYNAMIC_DRAW);

                gl!.drawArrays(gl!.POINTS, 0, f.numParticles);
                gl!.bindVertexArray(null);

                // Draw obstacle disk (debug only)
                if (scene.showDebug && scene.obstacleX > -5.0) {
                    gl!.useProgram(m_prog);
                    gl!.uniform2f(m_uRes, viewWidth, viewHeight);
                    gl!.uniform2f(m_uOff, viewLeft, viewBottom);
                    gl!.uniform3f(m_uCol, 1.0, 0.0, 0.0);
                    gl!.uniform2f(m_uTrans, scene.obstacleX, scene.obstacleY);
                    gl!.uniform1f(m_uScale, scene.obstacleRadius + f.particleRadius);

                    gl!.bindVertexArray(mVao);
                    gl!.drawElements(gl!.TRIANGLES, 3 * numSegs, gl!.UNSIGNED_SHORT, 0);
                    gl!.bindVertexArray(null);
                }

                rafId = requestAnimationFrame(loop);
            }
        } catch (err) {
            console.error('[FluidPool] loop error:', err);
        }
    }
    rafId = requestAnimationFrame(loop);

    // ── Cleanup ──
    function cleanup() {
        if (dead) return;
        dead = true;
        cancelAnimationFrame(rafId);
        for (const fn of rms) fn();
        gl!.deleteProgram(prog); gl!.deleteShader(vs); gl!.deleteShader(fs);
        gl!.deleteProgram(m_prog); gl!.deleteShader(m_vs); gl!.deleteShader(m_fs);
    }

    return {
        cleanup,
        reset: () => { },
        setFlipRatio: (v: number) => { scene.flipRatio = v; },
        setPaused: (v: boolean) => { scene.paused = v; },
        setShowDebug: (v: boolean) => { scene.showDebug = v; },
        setTiltForce: (x: number, y: number) => { scene.tiltForceX = x; scene.tiltForceY = y; },
    };
}
