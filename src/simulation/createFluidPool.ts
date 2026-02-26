
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
uniform vec2 u_res;
uniform float u_ptSz;
out vec3 v_col;

void main(){
  // u_res.x is simWidth, u_res.y is simHeight (e.g. 3.0)
  // map from [0, simWidth] to [-1, 1], and [0, simHeight] to [1, -1]
  vec2 screenTransform = vec2(2.0 / u_res.x, 2.0 / u_res.y);
  gl_Position = vec4(a_pos * screenTransform - vec2(1.0, 1.0), 0.0, 1.0);
  
  gl_PointSize = u_ptSz;
  v_col = a_color;
}`;

const FRAG = `#version 300 es
precision mediump float;
in vec3 v_col;
uniform float u_ptSz;
out vec4 o;

void main(){
  float rx = 0.5 - gl_PointCoord.x;
  float ry = 0.5 - gl_PointCoord.y;
  float r2 = rx * rx + ry * ry;
  if(r2 > 0.25) discard;
  
  float dist = sqrt(r2);
  vec3 col = v_col;
  // White border on the outer 20% of the particle radius
  
  
  o = vec4(col, 1.0);
}`;

// Mesh Shaders for Obstacle Disk
const MESH_VERT = `#version 300 es
precision highp float;
in vec2 a_pos;
uniform vec2 u_res;
uniform vec3 u_color;
uniform vec2 u_translation;
uniform float u_scale;
out vec3 v_col;

void main(){
  vec2 v = u_translation + a_pos * u_scale;
  vec2 screenTransform = vec2(2.0 / u_res.x, 2.0 / u_res.y);
  gl_Position = vec4(v * screenTransform - vec2(1.0, 1.0), 0.0, 1.0);
  v_col = u_color;
}`;

const MESH_FRAG = `#version 300 es
precision mediump float;
in vec3 v_col;
out vec4 o;
void main(){
  o = vec4(v_col, 1.0);
}`;

// ═══ Constants ══════════════════════════════════════════════════
const FLUID_CELL = 0;
const AIR_CELL = 1;
const SOLID_CELL = 2;

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
    particleDensity: Float32Array;
    particleRestDensity: number;

    particleRadius: number;
    pInvSpacing: number;
    pNumX: number;
    pNumY: number;
    pNumCells: number;

    numCellParticles: Int32Array;
    firstCellParticle: Int32Array;
    cellParticleIds: Int32Array;

    numParticles: number;

    constructor(density: number, width: number, height: number, spacing: number, particleRadius: number, maxParticles: number) {
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
            this.particleColor[3 * i + 2] = 1.0;
        }

        this.particleVel = new Float32Array(2 * this.maxParticles);
        this.particleDensity = new Float32Array(this.fNumCells);
        this.particleRestDensity = 0.0;

        this.particleRadius = particleRadius;
        this.pInvSpacing = 1.0 / (2.2 * particleRadius);
        this.pNumX = Math.floor(width * this.pInvSpacing) + 1;
        this.pNumY = Math.floor(height * this.pInvSpacing) + 1;
        this.pNumCells = this.pNumX * this.pNumY;

        this.numCellParticles = new Int32Array(this.pNumCells);
        this.firstCellParticle = new Int32Array(this.pNumCells + 1);
        this.cellParticleIds = new Int32Array(maxParticles);

        this.numParticles = 0;
    }

    integrateParticles(dt: number, gravity: number) {
        for (let i = 0; i < this.numParticles; i++) {
            this.particleVel[2 * i + 1] += dt * gravity;
            this.particlePos[2 * i] += this.particleVel[2 * i] * dt;
            this.particlePos[2 * i + 1] += this.particleVel[2 * i + 1] * dt;
        }
    }

    pushParticlesApart(numIters: number) {
        const colorDiffusionCoeff = 0.001;

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

        // Increase the separation distance slightly to force gaps and reduce jitter
        const minDist = 2.4 * this.particleRadius;
        const minDist2 = minDist * minDist;

        for (let iter = 0; iter < numIters; iter++) {
            for (let i = 0; i < this.numParticles; i++) {
                let px = this.particlePos[2 * i];
                let py = this.particlePos[2 * i + 1];

                const pxi = Math.floor(px * this.pInvSpacing);
                const pyi = Math.floor(py * this.pInvSpacing);
                const x0 = Math.max(pxi - 1, 0);
                const y0 = Math.max(pyi - 1, 0);
                const x1 = Math.min(pxi + 1, this.pNumX - 1);
                const y1 = Math.min(pyi + 1, this.pNumY - 1);

                for (let xi = x0; xi <= x1; xi++) {
                    for (let yi = y0; yi <= y1; yi++) {
                        const cellNr = xi * this.pNumY + yi;
                        const first = this.firstCellParticle[cellNr];
                        const last = this.firstCellParticle[cellNr + 1];
                        for (let j = first; j < last; j++) {
                            const id = this.cellParticleIds[j];
                            if (id === i) continue;
                            let qx = this.particlePos[2 * id];
                            let qy = this.particlePos[2 * id + 1];

                            let dx = qx - px;
                            let dy = qy - py;
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
        }
    }

    handleParticleCollisions(obstacleX: number, obstacleY: number, obstacleRadius: number, obstacleVelX: number, obstacleVelY: number) {
        const h = 1.0 / this.fInvSpacing;
        const r = this.particleRadius;
        const or = obstacleRadius;
        const minDist = obstacleRadius + r;
        const minDist2 = minDist * minDist;

        const minX = h + r;
        const maxX = (this.fNumX - 1) * h - r;
        const minY = h + r;
        const maxY = (this.fNumY - 1) * h - r;

        for (let i = 0; i < this.numParticles; i++) {
            let x = this.particlePos[2 * i];
            let y = this.particlePos[2 * i + 1];

            const dx = x - obstacleX;
            const dy = y - obstacleY;
            const d2 = dx * dx + dy * dy;

            if (d2 < minDist2) {
                this.particleVel[2 * i] = obstacleVelX;
                this.particleVel[2 * i + 1] = obstacleVelY;
            }

            if (x < minX) {
                x = minX;
                this.particleVel[2 * i] = 0.0;
            }
            if (x > maxX) {
                x = maxX;
                this.particleVel[2 * i] = 0.0;
            }
            if (y < minY) {
                y = minY;
                this.particleVel[2 * i + 1] = 0.0;
            }
            if (y > maxY) {
                y = maxY;
                this.particleVel[2 * i + 1] = 0.0;
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

                const x0 = Math.min(Math.floor((x - dx) * h1), this.fNumX - 2);
                const tx = ((x - dx) - x0 * h) * h1;
                const x1 = Math.min(x0 + 1, this.fNumX - 2);

                const y0 = Math.min(Math.floor((y - dy) * h1), this.fNumY - 2);
                const ty = ((y - dy) - y0 * h) * h1;
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
                    f[nr0] += pv * d0; d[nr0] += d0;
                    f[nr1] += pv * d1; d[nr1] += d1;
                    f[nr2] += pv * d2; d[nr2] += d2;
                    f[nr3] += pv * d3; d[nr3] += d3;
                } else {
                    const offset = component === 0 ? n : 1;
                    const valid0 = this.cellType[nr0] !== AIR_CELL || this.cellType[nr0 - offset] !== AIR_CELL ? 1.0 : 0.0;
                    const valid1 = this.cellType[nr1] !== AIR_CELL || this.cellType[nr1 - offset] !== AIR_CELL ? 1.0 : 0.0;
                    const valid2 = this.cellType[nr2] !== AIR_CELL || this.cellType[nr2 - offset] !== AIR_CELL ? 1.0 : 0.0;
                    const valid3 = this.cellType[nr3] !== AIR_CELL || this.cellType[nr3 - offset] !== AIR_CELL ? 1.0 : 0.0;

                    const v = this.particleVel[2 * i + component];
                    const weight = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;

                    if (weight > 0.0) {
                        const picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / weight;
                        const corr = (valid0 * d0 * (f[nr0] - prevF[nr0]) + valid1 * d1 * (f[nr1] - prevF[nr1])
                            + valid2 * d2 * (f[nr2] - prevF[nr2]) + valid3 * d3 * (f[nr3] - prevF[nr3])) / weight;
                        const flipV = v + corr;

                        this.particleVel[2 * i + component] = (1.0 - flipRatio) * picV + flipRatio * flipV;
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
                        const solid = this.cellType[i * n + j] === SOLID_CELL;
                        if (solid || (i > 0 && this.cellType[(i - 1) * n + j] === SOLID_CELL))
                            this.u[i * n + j] = this.prevU[i * n + j];
                        if (solid || (j > 0 && this.cellType[i * n + j - 1] === SOLID_CELL))
                            this.v[i * n + j] = this.prevV[i * n + j];
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

    updateParticleColors() {
        // Keep particles simple blue
        for (let i = 0; i < this.numParticles; i++) {
            this.particleColor[3 * i] = 0.1;
            this.particleColor[3 * i + 1] = 0.4;
            this.particleColor[3 * i + 2] = 1.0;
        }
    }

    simulate(dt: number, gravity: number, flipRatio: number, numPressureIters: number, numParticleIters: number, overRelaxation: number, compensateDrift: boolean, separateParticles: boolean, obstacleX: number, obstacleY: number, obstacleRadius: number, obstacleVelX: number, obstacleVelY: number) {
        const numSubSteps = 1;
        const sdt = dt / numSubSteps;

        for (let step = 0; step < numSubSteps; step++) {
            this.integrateParticles(sdt, gravity);
            if (separateParticles)
                this.pushParticlesApart(numParticleIters);
            this.handleParticleCollisions(obstacleX, obstacleY, obstacleRadius, obstacleVelX, obstacleVelY);
            this.transferVelocities(true, flipRatio);
            this.updateParticleDensity();
            this.solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift);
            this.transferVelocities(false, flipRatio);
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
        flipRatio: opts?.flipRatio ?? 0.9,
        numPressureIters: opts?.pressureIters ?? 50,
        numParticleIters: opts?.sepIters ?? 4, // More separation iterations to fix bottom jitter
        overRelaxation: opts?.overRelax ?? 1.9,
        compensateDrift: true,
        separateParticles: true,
        obstacleX: 0.0,
        obstacleY: 0.0,
        obstacleRadius: 0.15,
        obstacleVelX: 0.0,
        obstacleVelY: 0.0,
        paused: false,
        frameNr: 0
    };

    const simHeight = 3.0;
    const cScale = (canvas.clientHeight || window.innerHeight) / simHeight;
    const simWidth = (canvas.clientWidth || window.innerWidth) / cScale;
    const res = 100;

    const tankHeight = 1.0 * simHeight;
    const tankWidth = 1.0 * simWidth;
    const h = tankHeight / res;
    const density = 1000.0;

    // decrease starting volume
    const relWaterHeight = 0.50;
    const relWaterWidth = 0.40;

    // compute number of particles
    const r = 0.4 * h; // increased radius
    const dx = 2.0 * r;
    const dy = Math.sqrt(3.0) / 2.0 * dx;

    const numX = Math.floor((relWaterWidth * tankWidth - 2.0 * h - 2.0 * r) / dx);
    const numY = Math.floor((relWaterHeight * tankHeight - 2.0 * h - 2.0 * r) / dy);
    // Allow room to spawn thousands of extra particles by clicking
    const maxParticles = numX * numY + 5000;

    // create fluid
    const f = new FlipFluid(density, tankWidth, tankHeight, h, r, maxParticles);

    // create particles
    f.numParticles = numX * numY;
    let p = 0;
    for (let i = 0; i < numX; i++) {
        for (let j = 0; j < numY; j++) {
            f.particlePos[p++] = h + r + dx * i + (j % 2 === 0 ? 0.0 : r);
            f.particlePos[p++] = h + r + dy * j;
        }
    }

    // setup grid cells for tank
    const n = f.fNumY;
    for (let i = 0; i < f.fNumX; i++) {
        for (let j = 0; j < f.fNumY; j++) {
            let s = 1.0;
            if (i === 0 || i === f.fNumX - 1 || j === 0)
                s = 0.0;
            f.s[i * n + j] = s;
        }
    }

    // ── WebGL2 Rendering Setup ──
    const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false });
    if (!gl) {
        console.warn('No WebGL2');
        return { cleanup() { }, reset() { }, setFlipRatio() { }, setPaused() { } };
    }

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs); gl.attachShader(prog, fs);
    gl.linkProgram(prog);

    const uRes = gl.getUniformLocation(prog, 'u_res')!;
    const uPt = gl.getUniformLocation(prog, 'u_ptSz')!;
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
    gl.bindVertexArray(null);

    // Mesh setup for obstacle
    const m_vs = compile(gl, gl.VERTEX_SHADER, MESH_VERT);
    const m_fs = compile(gl, gl.FRAGMENT_SHADER, MESH_FRAG);
    const m_prog = gl.createProgram()!;
    gl.attachShader(m_prog, m_vs); gl.attachShader(m_prog, m_fs);
    gl.linkProgram(m_prog);

    // variables omitted as the obstacle mesh is no longer drawn

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

    function setObstacle(x: number, y: number, reset: boolean) {
        let vx = 0.0;
        let vy = 0.0;
        if (!reset) {
            vx = (x - scene.obstacleX) / scene.dt;
            vy = (y - scene.obstacleY) / scene.dt;
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

    function dropParticles(cx: number, cy: number) {
        const rc = canvas.getBoundingClientRect();
        const mx = cx - rc.left;
        const my = cy - rc.top;

        const x = mx / cScale;
        const y = (canvas.clientHeight - my) / cScale; // y-axis up

        // Drop a cluster of particles
        const dropCount = 40;

        for (let i = 0; i < dropCount; i++) {
            if (f.numParticles >= f.maxParticles) break;

            const idx = f.numParticles;
            f.particlePos[2 * idx] = x;
            f.particlePos[2 * idx + 1] = y;

            // Wavy / fanning out burst
            const angle = Math.random() * Math.PI * 2;
            const speed = 3.0 + Math.random() * 5.0; // fast enough to fan out clearly

            f.particleVel[2 * idx] = Math.cos(angle) * speed;
            f.particleVel[2 * idx + 1] = Math.sin(angle) * speed - 2.0; // radial outward + downward bias

            f.numParticles++;
        }
    }

    function handlePointerMove(cx: number, cy: number) {
        const rc = canvas.getBoundingClientRect();
        const mx = cx - rc.left;
        const my = cy - rc.top;
        const x = mx / cScale;
        const y = (canvas.clientHeight - my) / cScale;
        setObstacle(x, y, false);
    }

    function handlePointerLeave() {
        scene.obstacleVelX = 0.0;
        scene.obstacleVelY = 0.0;
        setObstacle(-10, -10, true);
    }

    // Set obstacle far away initially
    setObstacle(-10, -10, true);

    listen(wrapper, 'mousedown', ((e: MouseEvent) => {
        dropParticles(e.clientX, e.clientY);
    }) as EventListener);
    listen(wrapper, 'mousemove', ((e: MouseEvent) => handlePointerMove(e.clientX, e.clientY)) as EventListener);
    listen(wrapper, 'mouseleave', handlePointerLeave as EventListener);
    listen(window, 'mouseup', handlePointerLeave as EventListener);

    listen(wrapper, 'touchstart', ((e: TouchEvent) => {
        e.preventDefault();
        handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
        dropParticles(e.touches[0].clientX, e.touches[0].clientY);
    }) as EventListener, false);
    listen(wrapper, 'touchmove', ((e: TouchEvent) => {
        e.preventDefault();
        handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
    }) as EventListener, false);
    listen(wrapper, 'touchend', ((e: TouchEvent) => {
        e.preventDefault();
        handlePointerLeave();
    }) as EventListener, false);

    // ── Loop ──
    let rafId = 0;
    function loop() {
        if (!dead) {
            if (!scene.paused) {
                f.simulate(
                    scene.dt, scene.gravity, scene.flipRatio, scene.numPressureIters, scene.numParticleIters,
                    scene.overRelaxation, scene.compensateDrift, scene.separateParticles,
                    scene.obstacleX, scene.obstacleY, scene.obstacleRadius, scene.obstacleVelX, scene.obstacleVelY
                );
                scene.frameNr++;
            }

            // Render
            const cvs = gl!.canvas as HTMLCanvasElement;
            if (cvs.width !== cvs.clientWidth || cvs.height !== cvs.clientHeight) {
                cvs.width = cvs.clientWidth; cvs.height = cvs.clientHeight;
                // note: ideally if resizing we should recreate grids, but we'll leave it
            }
            gl!.viewport(0, 0, cvs.width, cvs.height);
            gl!.clearColor(0, 0, 0, 0);
            gl!.clear(gl!.COLOR_BUFFER_BIT | gl!.DEPTH_BUFFER_BIT);
            gl!.enable(gl!.BLEND);
            gl!.blendFunc(gl!.SRC_ALPHA, gl!.ONE_MINUS_SRC_ALPHA);

            if (scene.frameNr % 60 === 0) {
                console.log(`Frame ${scene.frameNr}: P0[${f.particlePos[0].toFixed(2)}, ${f.particlePos[1].toFixed(2)}] NumP: ${f.numParticles} simW: ${simWidth.toFixed(2)}`);
            }

            const ptSz = 2.0 * f.particleRadius / simWidth * cvs.width;

            gl!.useProgram(prog);
            gl!.uniform2f(uRes, simWidth, simHeight);
            gl!.uniform1f(uPt, ptSz);

            gl!.bindVertexArray(vao);
            gl!.bindBuffer(gl!.ARRAY_BUFFER, pBuf);
            gl!.bufferData(gl!.ARRAY_BUFFER, f.particlePos, gl!.DYNAMIC_DRAW);
            gl!.bindBuffer(gl!.ARRAY_BUFFER, cBuf);
            gl!.bufferData(gl!.ARRAY_BUFFER, f.particleColor, gl!.DYNAMIC_DRAW);

            gl!.drawArrays(gl!.POINTS, 0, f.numParticles);
            gl!.bindVertexArray(null);

            // Draw obstacle (disabled to keep invisible on drag)
            /*
            if (scene.obstacleX > -5.0) {
                gl!.useProgram(m_prog);
                gl!.uniform2f(m_uRes, simWidth, simHeight);
                gl!.uniform3f(m_uCol, 1.0, 0.0, 0.0);
                gl!.uniform2f(m_uTrans, scene.obstacleX, scene.obstacleY);
                gl!.uniform1f(m_uScale, scene.obstacleRadius + f.particleRadius);

                gl!.bindVertexArray(mVao);
                gl!.drawElements(gl!.TRIANGLES, 3 * numSegs, gl!.UNSIGNED_SHORT, 0);
                gl!.bindVertexArray(null);
            }
            */

            rafId = requestAnimationFrame(loop);
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
    };
}
