/**
 * FluidSim — 2D FLIP (Fluid-Implicit Particle) simulation
 *
 * Coordinate system : y-down (0,0 = top-left)
 * Grid              : MAC staggered
 *   U on vertical faces   — size (nx+1) × ny     — position (i·h, (j+0.5)·h)
 *   V on horizontal faces — size nx × (ny+1)      — position ((i+0.5)·h, j·h)
 *   pressure / cellType   — size nx × ny (centers) — position ((i+0.5)·h, (j+0.5)·h)
 * Gravity           : (0, +g)  (positive = downward)
 */

// ─── Cell type enum ─────────────────────────────────────────────
const SOLID = 0;
const AIR = 1;
const FLUID = 2;

// ─── Configuration ──────────────────────────────────────────────
export interface FluidSimConfig {
    width: number;
    height: number;
    spacing: number;
    density: number;
    particleRadius: number;
    numParticles: number;
    dt: number;
    gravity: number;
    flipRatio: number;
    numPressureIters?: number;
    numParticleSepIters?: number;
    overRelaxation?: number;
    wallDamping?: number;
    wallFriction?: number;
    velocityDamping?: number;
    obstacleRadius?: number;
    dampingObstacle?: number;
    densityCorrectionStrength?: number;
    returnForce?: number;       // extra downward accel above pool region
    maxVelocity?: number;       // velocity clamp
}

// ─── FluidSim class ─────────────────────────────────────────────
export class FluidSim {
    width: number;
    height: number;
    h: number;
    density: number;
    particleRadius: number;
    numParticles: number;
    dt: number;
    gravity: number;
    flipRatio: number;
    numPressureIters: number;
    numParticleSepIters: number;
    overRelaxation: number;
    wallDamping: number;
    wallFriction: number;
    velocityDamping: number;
    densityCorrectionStrength: number;
    returnForce: number;
    maxVelocity: number;

    /* soft pool region: top boundary of the "rest" zone (y-down) */
    poolTopY: number;

    /* obstacle (mouse/touch driven) */
    obstacleX: number;
    obstacleY: number;
    obstacleRadius: number;
    dampingObstacle: number;
    obstacleActive: boolean;

    /* grid */
    nx: number;
    ny: number;
    U: Float32Array;
    V: Float32Array;
    prevU: Float32Array;
    prevV: Float32Array;
    weightU: Float32Array;
    weightV: Float32Array;
    pressure: Float32Array;
    divergence: Float32Array;
    cellType: Int32Array;
    gridDensity: Float32Array;

    restDensity: number;
    private restDensitySet: boolean;

    /* particles */
    posX: Float32Array;
    posY: Float32Array;
    velX: Float32Array;
    velY: Float32Array;
    particleColor: Float32Array;

    /* spatial hash */
    private hashCellSize: number;
    private hashW: number;
    private hashH: number;
    private hashCount: Int32Array;
    private hashStart: Int32Array;
    private hashSorted: Int32Array;

    constructor(cfg: FluidSimConfig) {
        this.width = cfg.width;
        this.height = cfg.height;
        this.h = cfg.spacing;
        this.density = cfg.density;
        this.particleRadius = cfg.particleRadius;
        this.numParticles = cfg.numParticles;
        this.dt = cfg.dt;
        this.gravity = cfg.gravity;
        this.flipRatio = cfg.flipRatio;
        this.numPressureIters = cfg.numPressureIters ?? 50;
        this.numParticleSepIters = cfg.numParticleSepIters ?? 2;
        this.overRelaxation = cfg.overRelaxation ?? 1.9;
        this.wallDamping = cfg.wallDamping ?? 0.5;
        this.wallFriction = cfg.wallFriction ?? 0.05;
        this.velocityDamping = cfg.velocityDamping ?? 2.0;
        this.densityCorrectionStrength = cfg.densityCorrectionStrength ?? 0.3;
        this.returnForce = cfg.returnForce ?? 600;
        this.maxVelocity = cfg.maxVelocity ?? 500;

        // pool rest zone: bottom 35% (top of pool at 65% from canvas top)
        this.poolTopY = this.height * 0.65;

        this.obstacleX = this.width * 0.5;
        this.obstacleY = this.height * 0.5;
        this.obstacleRadius = cfg.obstacleRadius ?? 40;
        this.dampingObstacle = cfg.dampingObstacle ?? 0.5;
        this.obstacleActive = false;

        this.nx = Math.floor(this.width / this.h) + 1;
        this.ny = Math.floor(this.height / this.h) + 1;

        const uLen = (this.nx + 1) * this.ny;
        const vLen = this.nx * (this.ny + 1);
        const cLen = this.nx * this.ny;

        this.U = new Float32Array(uLen);
        this.V = new Float32Array(vLen);
        this.prevU = new Float32Array(uLen);
        this.prevV = new Float32Array(vLen);
        this.weightU = new Float32Array(uLen);
        this.weightV = new Float32Array(vLen);
        this.pressure = new Float32Array(cLen);
        this.divergence = new Float32Array(cLen);
        this.cellType = new Int32Array(cLen);
        this.gridDensity = new Float32Array(cLen);

        this.restDensity = 0;
        this.restDensitySet = false;

        this.posX = new Float32Array(this.numParticles);
        this.posY = new Float32Array(this.numParticles);
        this.velX = new Float32Array(this.numParticles);
        this.velY = new Float32Array(this.numParticles);
        this.particleColor = new Float32Array(this.numParticles * 3);

        this.hashCellSize = 2 * this.particleRadius;
        this.hashW = Math.ceil(this.width / this.hashCellSize) + 1;
        this.hashH = Math.ceil(this.height / this.hashCellSize) + 1;
        const numHash = this.hashW * this.hashH;
        this.hashCount = new Int32Array(numHash);
        this.hashStart = new Int32Array(numHash);
        this.hashSorted = new Int32Array(this.numParticles);

        this.initParticles();
    }

    private iU(i: number, j: number) { return i + j * (this.nx + 1); }
    private iV(i: number, j: number) { return i + j * this.nx; }
    private iC(i: number, j: number) { return i + j * this.nx; }

    setObstacle(x: number, y: number, r?: number) {
        this.obstacleX = x;
        this.obstacleY = y;
        if (r !== undefined) this.obstacleRadius = r;
        this.obstacleActive = true;
        this.collideObstacle();
    }

    initParticles() {
        const margin = this.particleRadius;
        const waterHeight = this.height * 0.35;
        const sx = margin;
        const ex = this.width - margin;
        const sy = this.height - waterHeight + margin;
        const ey = this.height - margin;

        const aspect = (ex - sx) / (ey - sy);
        const cols = Math.ceil(Math.sqrt(this.numParticles * aspect));
        const rows = Math.ceil(this.numParticles / cols);
        const dx = (ex - sx) / cols;
        const dy = (ey - sy) / rows;

        let idx = 0;
        for (let r = 0; r < rows && idx < this.numParticles; r++) {
            for (let c = 0; c < cols && idx < this.numParticles; c++) {
                this.posX[idx] = sx + (c + 0.5) * dx + (Math.random() - 0.5) * dx * 0.4;
                this.posY[idx] = sy + (r + 0.5) * dy + (Math.random() - 0.5) * dy * 0.4;
                this.velX[idx] = 0;
                this.velY[idx] = 0;
                this.particleColor[idx * 3] = 0.1;
                this.particleColor[idx * 3 + 1] = 0.3;
                this.particleColor[idx * 3 + 2] = 0.8;
                idx++;
            }
        }

        this.U.fill(0); this.V.fill(0);
        this.prevU.fill(0); this.prevV.fill(0);
        this.pressure.fill(0); this.divergence.fill(0);
        this.gridDensity.fill(0);
        this.restDensitySet = false;
        this.restDensity = 0;
    }

    // ── bilinear sampling ─────────────────────────────────────────
    private sampleU(x: number, y: number): number {
        const h = this.h;
        const gx = x / h;
        const gy = y / h - 0.5;
        let i0 = Math.floor(gx);
        let j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, this.nx - 1));
        j0 = Math.max(0, Math.min(j0, this.ny - 2));
        const fx = Math.max(0, Math.min(1, gx - i0));
        const fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * this.U[this.iU(i0, j0)]
            + fx * (1 - fy) * this.U[this.iU(i0 + 1, j0)]
            + (1 - fx) * fy * this.U[this.iU(i0, j0 + 1)]
            + fx * fy * this.U[this.iU(i0 + 1, j0 + 1)];
    }

    private sampleV(x: number, y: number): number {
        const h = this.h;
        const gx = x / h - 0.5;
        const gy = y / h;
        let i0 = Math.floor(gx);
        let j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, this.nx - 2));
        j0 = Math.max(0, Math.min(j0, this.ny - 1));
        const fx = Math.max(0, Math.min(1, gx - i0));
        const fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * this.V[this.iV(i0, j0)]
            + fx * (1 - fy) * this.V[this.iV(i0 + 1, j0)]
            + (1 - fx) * fy * this.V[this.iV(i0, j0 + 1)]
            + fx * fy * this.V[this.iV(i0 + 1, j0 + 1)];
    }

    private samplePrevU(x: number, y: number): number {
        const h = this.h;
        const gx = x / h;
        const gy = y / h - 0.5;
        let i0 = Math.floor(gx);
        let j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, this.nx - 1));
        j0 = Math.max(0, Math.min(j0, this.ny - 2));
        const fx = Math.max(0, Math.min(1, gx - i0));
        const fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * this.prevU[this.iU(i0, j0)]
            + fx * (1 - fy) * this.prevU[this.iU(i0 + 1, j0)]
            + (1 - fx) * fy * this.prevU[this.iU(i0, j0 + 1)]
            + fx * fy * this.prevU[this.iU(i0 + 1, j0 + 1)];
    }

    private samplePrevV(x: number, y: number): number {
        const h = this.h;
        const gx = x / h - 0.5;
        const gy = y / h;
        let i0 = Math.floor(gx);
        let j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, this.nx - 2));
        j0 = Math.max(0, Math.min(j0, this.ny - 1));
        const fx = Math.max(0, Math.min(1, gx - i0));
        const fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * this.prevV[this.iV(i0, j0)]
            + fx * (1 - fy) * this.prevV[this.iV(i0 + 1, j0)]
            + (1 - fx) * fy * this.prevV[this.iV(i0, j0 + 1)]
            + fx * fy * this.prevV[this.iV(i0 + 1, j0 + 1)];
    }

    private sampleDensity(x: number, y: number): number {
        const h = this.h;
        const { nx, ny } = this;
        const gx = x / h - 0.5;
        const gy = y / h - 0.5;
        let i0 = Math.floor(gx);
        let j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, nx - 2));
        j0 = Math.max(0, Math.min(j0, ny - 2));
        const fx = Math.max(0, Math.min(1, gx - i0));
        const fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * this.gridDensity[this.iC(i0, j0)]
            + fx * (1 - fy) * this.gridDensity[this.iC(i0 + 1, j0)]
            + (1 - fx) * fy * this.gridDensity[this.iC(i0, j0 + 1)]
            + fx * fy * this.gridDensity[this.iC(i0 + 1, j0 + 1)];
    }

    // ═══════════════════════════════════════════════════════════════
    //  MAIN SIMULATION STEP
    // ═══════════════════════════════════════════════════════════════
    step() {
        this.prevU.set(this.U);
        this.prevV.set(this.V);

        this.applyGravityAndAdvect();
        this.enforceBoundaries();
        this.collideObstacle();
        this.separateParticles();
        this.computeDensity();
        this.classifyCells();
        this.particleToGrid();
        this.pressureProjection();
        this.gridToParticle();
        this.correctDensity();
        this.collideObstacle();
        this.enforceBoundaries();
        this.clampVelocities();
        this.updateColors();
    }

    // ── gravity + advect + soft return force ──────────────────────
    private applyGravityAndAdvect() {
        const dt = this.dt;
        const g = this.gravity;
        const damp = 1 - this.velocityDamping * dt;

        for (let p = 0; p < this.numParticles; p++) {
            // base damping
            this.velX[p] *= damp;
            this.velY[p] *= damp;

            // gravity
            this.velY[p] += g * dt;

            // soft return: if above pool region, apply extra downward pull + extra damping
            if (this.posY[p] < this.poolTopY) {
                this.velY[p] += this.returnForce * dt;
                // extra damping above pool to settle quickly
                this.velX[p] *= 0.98;
                this.velY[p] *= 0.98;
            }

            // advect
            this.posX[p] += this.velX[p] * dt;
            this.posY[p] += this.velY[p] * dt;
        }
    }

    // ── velocity clamp ────────────────────────────────────────────
    private clampVelocities() {
        const max = this.maxVelocity;
        const max2 = max * max;
        for (let p = 0; p < this.numParticles; p++) {
            const vx = this.velX[p];
            const vy = this.velY[p];
            const v2 = vx * vx + vy * vy;
            if (v2 > max2) {
                const s = max / Math.sqrt(v2);
                this.velX[p] *= s;
                this.velY[p] *= s;
            }
        }
    }

    // ── density splat ─────────────────────────────────────────────
    private computeDensity() {
        const { h, nx, ny, numParticles } = this;
        this.gridDensity.fill(0);

        for (let p = 0; p < numParticles; p++) {
            const gx = this.posX[p] / h - 0.5;
            const gy = this.posY[p] / h - 0.5;
            let i0 = Math.floor(gx);
            let j0 = Math.floor(gy);
            i0 = Math.max(0, Math.min(i0, nx - 2));
            j0 = Math.max(0, Math.min(j0, ny - 2));
            const fx = Math.max(0, Math.min(1, gx - i0));
            const fy = Math.max(0, Math.min(1, gy - j0));

            this.gridDensity[this.iC(i0, j0)] += (1 - fx) * (1 - fy);
            this.gridDensity[this.iC(i0 + 1, j0)] += fx * (1 - fy);
            this.gridDensity[this.iC(i0, j0 + 1)] += (1 - fx) * fy;
            this.gridDensity[this.iC(i0 + 1, j0 + 1)] += fx * fy;
        }

        if (!this.restDensitySet) {
            let sum = 0;
            let count = 0;
            for (let j = 1; j < ny - 1; j++) {
                for (let i = 1; i < nx - 1; i++) {
                    const d = this.gridDensity[this.iC(i, j)];
                    if (d > 0.01) { sum += d; count++; }
                }
            }
            this.restDensity = count > 0 ? sum / count : 1;
            this.restDensitySet = true;
        }
    }

    // ── density correction ────────────────────────────────────────
    private correctDensity() {
        if (this.restDensity < 0.01) return;
        const { h } = this;
        const strength = this.densityCorrectionStrength;

        for (let p = 0; p < this.numParticles; p++) {
            const px = this.posX[p];
            const py = this.posY[p];
            const localD = this.sampleDensity(px, py);
            const compression = localD - this.restDensity;

            if (compression > 0.01) {
                const eps = h * 0.5;
                const dR = this.sampleDensity(px + eps, py);
                const dL = this.sampleDensity(px - eps, py);
                const dD = this.sampleDensity(px, py + eps);
                const dU = this.sampleDensity(px, py - eps);

                let gx = (dR - dL) / (2 * eps);
                let gy = (dD - dU) / (2 * eps);
                const gradLen = Math.sqrt(gx * gx + gy * gy);
                if (gradLen > 1e-6) {
                    gx /= gradLen;
                    gy /= gradLen;
                    const push = Math.min(compression * strength * this.dt, this.particleRadius * 0.5);
                    this.posX[p] -= gx * push;
                    this.posY[p] -= gy * push;
                }
            }
        }
    }

    // ── classify cells ────────────────────────────────────────────
    private classifyCells() {
        const { nx, ny, h } = this;
        this.cellType.fill(AIR);

        for (let i = 0; i < nx; i++) {
            this.cellType[this.iC(i, 0)] = SOLID;
            this.cellType[this.iC(i, ny - 1)] = SOLID;
        }
        for (let j = 0; j < ny; j++) {
            this.cellType[this.iC(0, j)] = SOLID;
            this.cellType[this.iC(nx - 1, j)] = SOLID;
        }

        if (this.obstacleActive) {
            const cx = this.obstacleX;
            const cy = this.obstacleY;
            const or2 = this.obstacleRadius * this.obstacleRadius;
            for (let j = 1; j < ny - 1; j++) {
                for (let i = 1; i < nx - 1; i++) {
                    const dx = (i + 0.5) * h - cx;
                    const dy = (j + 0.5) * h - cy;
                    if (dx * dx + dy * dy < or2) {
                        this.cellType[this.iC(i, j)] = SOLID;
                    }
                }
            }
        }

        for (let p = 0; p < this.numParticles; p++) {
            const ci = Math.floor(this.posX[p] / h);
            const cj = Math.floor(this.posY[p] / h);
            if (ci >= 1 && ci < nx - 1 && cj >= 1 && cj < ny - 1) {
                if (this.cellType[this.iC(ci, cj)] !== SOLID) {
                    this.cellType[this.iC(ci, cj)] = FLUID;
                }
            }
        }
    }

    // ── obstacle collision ────────────────────────────────────────
    collideObstacle() {
        if (!this.obstacleActive) return;
        const cx = this.obstacleX;
        const cy = this.obstacleY;
        const R = this.obstacleRadius + this.particleRadius;
        const damp = this.dampingObstacle;

        for (let p = 0; p < this.numParticles; p++) {
            const dx = this.posX[p] - cx;
            const dy = this.posY[p] - cy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < R) {
                const invD = dist > 1e-6 ? 1 / dist : 0;
                const nx = dx * invD;
                const ny = dy * invD;
                this.posX[p] = cx + nx * R;
                this.posY[p] = cy + ny * R;
                const vn = this.velX[p] * nx + this.velY[p] * ny;
                if (vn < 0) {
                    this.velX[p] -= (1 + damp) * vn * nx;
                    this.velY[p] -= (1 + damp) * vn * ny;
                }
                this.velX[p] *= (1 - damp * 0.3);
                this.velY[p] *= (1 - damp * 0.3);
            }
        }
    }

    // ── P2G ───────────────────────────────────────────────────────
    private particleToGrid() {
        const { h, nx, ny, numParticles } = this;
        this.U.fill(0); this.V.fill(0);
        this.weightU.fill(0); this.weightV.fill(0);

        for (let p = 0; p < numParticles; p++) {
            const px = this.posX[p], py = this.posY[p];
            const vx = this.velX[p], vy = this.velY[p];

            // U (vertical faces)
            {
                const gx = px / h, gy = py / h - 0.5;
                let i0 = Math.floor(gx), j0 = Math.floor(gy);
                i0 = Math.max(0, Math.min(i0, nx - 1));
                j0 = Math.max(0, Math.min(j0, ny - 2));
                const fx = Math.max(0, Math.min(1, gx - i0));
                const fy = Math.max(0, Math.min(1, gy - j0));
                const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                const a = this.iU(i0, j0), b = this.iU(i0 + 1, j0), c = this.iU(i0, j0 + 1), d = this.iU(i0 + 1, j0 + 1);
                this.U[a] += w00 * vx; this.weightU[a] += w00;
                this.U[b] += w10 * vx; this.weightU[b] += w10;
                this.U[c] += w01 * vx; this.weightU[c] += w01;
                this.U[d] += w11 * vx; this.weightU[d] += w11;
            }
            // V (horizontal faces)
            {
                const gx = px / h - 0.5, gy = py / h;
                let i0 = Math.floor(gx), j0 = Math.floor(gy);
                i0 = Math.max(0, Math.min(i0, nx - 2));
                j0 = Math.max(0, Math.min(j0, ny - 1));
                const fx = Math.max(0, Math.min(1, gx - i0));
                const fy = Math.max(0, Math.min(1, gy - j0));
                const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                const a = this.iV(i0, j0), b = this.iV(i0 + 1, j0), c = this.iV(i0, j0 + 1), d = this.iV(i0 + 1, j0 + 1);
                this.V[a] += w00 * vy; this.weightV[a] += w00;
                this.V[b] += w10 * vy; this.weightV[b] += w10;
                this.V[c] += w01 * vy; this.weightV[c] += w01;
                this.V[d] += w11 * vy; this.weightV[d] += w11;
            }
        }

        for (let i = 0; i < this.U.length; i++)
            this.U[i] = this.weightU[i] > 1e-6 ? this.U[i] / this.weightU[i] : 0;
        for (let i = 0; i < this.V.length; i++)
            this.V[i] = this.weightV[i] > 1e-6 ? this.V[i] / this.weightV[i] : 0;

        for (let j = 0; j < ny; j++)
            for (let i = 0; i < nx; i++)
                if (this.cellType[this.iC(i, j)] === SOLID) {
                    this.U[this.iU(i, j)] = 0; this.U[this.iU(i + 1, j)] = 0;
                    this.V[this.iV(i, j)] = 0; this.V[this.iV(i, j + 1)] = 0;
                }
    }

    // ── pressure projection ───────────────────────────────────────
    private pressureProjection() {
        const { nx, ny, h } = this;
        const omega = this.overRelaxation;
        this.pressure.fill(0);

        for (let iter = 0; iter < this.numPressureIters; iter++) {
            for (let j = 1; j < ny - 1; j++) {
                for (let i = 1; i < nx - 1; i++) {
                    if (this.cellType[this.iC(i, j)] !== FLUID) continue;
                    const div = (this.U[this.iU(i + 1, j)] - this.U[this.iU(i, j)]
                        + this.V[this.iV(i, j + 1)] - this.V[this.iV(i, j)]) / h;
                    const sl = this.cellType[this.iC(i - 1, j)] !== SOLID ? 1 : 0;
                    const sr = this.cellType[this.iC(i + 1, j)] !== SOLID ? 1 : 0;
                    const sb = this.cellType[this.iC(i, j - 1)] !== SOLID ? 1 : 0;
                    const st = this.cellType[this.iC(i, j + 1)] !== SOLID ? 1 : 0;
                    const s = sl + sr + sb + st;
                    if (s === 0) continue;
                    const pSum = sl * this.pressure[this.iC(i - 1, j)]
                        + sr * this.pressure[this.iC(i + 1, j)]
                        + sb * this.pressure[this.iC(i, j - 1)]
                        + st * this.pressure[this.iC(i, j + 1)];
                    const newP = (pSum - div * h) / s;
                    this.pressure[this.iC(i, j)] += omega * (newP - this.pressure[this.iC(i, j)]);
                }
            }
        }

        for (let j = 1; j < ny - 1; j++)
            for (let i = 1; i < nx; i++) {
                if (this.cellType[this.iC(i - 1, j)] === SOLID || this.cellType[this.iC(i, j)] === SOLID) continue;
                this.U[this.iU(i, j)] -= (this.pressure[this.iC(i, j)] - this.pressure[this.iC(i - 1, j)]) / h;
            }
        for (let j = 1; j < ny; j++)
            for (let i = 1; i < nx - 1; i++) {
                if (this.cellType[this.iC(i, j - 1)] === SOLID || this.cellType[this.iC(i, j)] === SOLID) continue;
                this.V[this.iV(i, j)] -= (this.pressure[this.iC(i, j)] - this.pressure[this.iC(i, j - 1)]) / h;
            }
    }

    // ── G2P (FLIP/PIC) ───────────────────────────────────────────
    private gridToParticle() {
        const fr = this.flipRatio;
        for (let p = 0; p < this.numParticles; p++) {
            const x = this.posX[p], y = this.posY[p];
            const picU = this.sampleU(x, y);
            const picV = this.sampleV(x, y);
            const deltaU = picU - this.samplePrevU(x, y);
            const deltaV = picV - this.samplePrevV(x, y);
            const flipU = this.velX[p] + deltaU;
            const flipV = this.velY[p] + deltaV;
            this.velX[p] = (1 - fr) * picU + fr * flipU;
            this.velY[p] = (1 - fr) * picV + fr * flipV;
        }
    }

    // ── particle separation ───────────────────────────────────────
    private separateParticles() {
        const cs = this.hashCellSize;
        const hw = this.hashW, hh = this.hashH;
        const numHash = hw * hh;
        const minDist = 2 * this.particleRadius;
        const minDist2 = minDist * minDist;

        for (let iter = 0; iter < this.numParticleSepIters; iter++) {
            this.hashCount.fill(0);
            for (let p = 0; p < this.numParticles; p++) {
                const ci = Math.max(0, Math.min(hw - 1, Math.floor(this.posX[p] / cs)));
                const cj = Math.max(0, Math.min(hh - 1, Math.floor(this.posY[p] / cs)));
                this.hashCount[ci + cj * hw]++;
            }
            this.hashStart[0] = 0;
            for (let i = 1; i < numHash; i++)
                this.hashStart[i] = this.hashStart[i - 1] + this.hashCount[i - 1];
            const tmp = new Int32Array(numHash);
            for (let p = 0; p < this.numParticles; p++) {
                const ci = Math.max(0, Math.min(hw - 1, Math.floor(this.posX[p] / cs)));
                const cj = Math.max(0, Math.min(hh - 1, Math.floor(this.posY[p] / cs)));
                const cell = ci + cj * hw;
                this.hashSorted[this.hashStart[cell] + tmp[cell]] = p;
                tmp[cell]++;
            }

            for (let p = 0; p < this.numParticles; p++) {
                const ci = Math.max(0, Math.min(hw - 1, Math.floor(this.posX[p] / cs)));
                const cj = Math.max(0, Math.min(hh - 1, Math.floor(this.posY[p] / cs)));
                for (let dj = -1; dj <= 1; dj++) {
                    const nj = cj + dj;
                    if (nj < 0 || nj >= hh) continue;
                    for (let di = -1; di <= 1; di++) {
                        const ni = ci + di;
                        if (ni < 0 || ni >= hw) continue;
                        const cell = ni + nj * hw;
                        const end = this.hashStart[cell] + this.hashCount[cell];
                        for (let k = this.hashStart[cell]; k < end; k++) {
                            const q = this.hashSorted[k];
                            if (q <= p) continue;
                            const dx = this.posX[q] - this.posX[p];
                            const dy = this.posY[q] - this.posY[p];
                            const d2 = dx * dx + dy * dy;
                            if (d2 < minDist2 && d2 > 1e-8) {
                                const d = Math.sqrt(d2);
                                const overlap = (minDist - d) * 0.5;
                                const nx = dx / d, ny = dy / d;
                                this.posX[p] -= overlap * nx;
                                this.posY[p] -= overlap * ny;
                                this.posX[q] += overlap * nx;
                                this.posY[q] += overlap * ny;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── wall boundaries (no hard pool edge) ───────────────────────
    private enforceBoundaries() {
        const r = this.particleRadius;
        const d = this.wallDamping;
        const f = this.wallFriction;
        const minX = r, maxX = this.width - r;
        const minY = r, maxY = this.height - r;

        for (let p = 0; p < this.numParticles; p++) {
            if (this.posX[p] < minX) { this.posX[p] = minX; this.velX[p] *= -d; this.velY[p] *= (1 - f); }
            if (this.posX[p] > maxX) { this.posX[p] = maxX; this.velX[p] *= -d; this.velY[p] *= (1 - f); }
            if (this.posY[p] < minY) { this.posY[p] = minY; this.velY[p] *= -d; this.velX[p] *= (1 - f); }
            if (this.posY[p] > maxY) { this.posY[p] = maxY; this.velY[p] *= -d; this.velX[p] *= (1 - f); }
        }
    }

    // ── per-particle color ────────────────────────────────────────
    private updateColors() {
        const maxSpeed = 150;
        for (let p = 0; p < this.numParticles; p++) {
            const vx = this.velX[p], vy = this.velY[p];
            const t = Math.min(1, Math.sqrt(vx * vx + vy * vy) / maxSpeed);
            this.particleColor[p * 3] = 0.1 + t * 0.2;
            this.particleColor[p * 3 + 1] = 0.3 + t * 0.5;
            this.particleColor[p * 3 + 2] = 0.8 + t * 0.2;
        }
    }
}
