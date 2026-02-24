/**
 * createFluidPool — self-contained 2D particle-fluid pool mini-program.
 *
 * Receives wrapper + canvas DOM elements.
 * Returns { cleanup, reset, setFlipRatio, setPaused }.
 *
 * Coord system: y-down (0,0 = top-left). Gravity = +y.
 * Grid: MAC staggered. Rendering: WebGL2 point-sprite SDF circles.
 */

// ═══ Shaders ════════════════════════════════════════════════════
const VERT = `#version 300 es
precision highp float;
in vec2 a_pos;
in vec3 a_color;
uniform vec2 u_res;
uniform float u_ptSz;
out vec3 v_col;
void main(){
  gl_Position=vec4(a_pos.x/u_res.x*2.0-1.0,1.0-a_pos.y/u_res.y*2.0,0,1);
  gl_PointSize=u_ptSz;
  v_col=a_color;
}`;
const FRAG = `#version 300 es
precision mediump float;
in vec3 v_col;
out vec4 o;
void main(){
  float d=length(gl_PointCoord-0.5)*2.0;
  if(d>1.0)discard;
  o=vec4(v_col,(1.0-smoothstep(0.8,1.0,d))*0.9);
}`;

// ═══ Constants ══════════════════════════════════════════════════
const SOLID = 0, AIR = 1, FLUID = 2;

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
    // ── Config ──
    const r = opts?.particleRadius ?? 12;
    const N = opts?.numParticles ?? 1800;
    const grav = opts?.gravity ?? 100;
    let fRat = opts?.flipRatio ?? 0.9;
    const DT = opts?.dt ?? 1 / 120;
    const pIt = opts?.pressureIters ?? 30;
    const sIt = opts?.sepIters ?? 2;
    const omega = opts?.overRelax ?? 1.9;
    const obsR = opts?.obstacleRadius ?? 100;

    const SUB = 2;
    const WFRIC = 0.7;     // wall friction
    const RETF = 200;     // gentle gravity pull-back above pool
    const SLEEPV = 3;       // sleep threshold (settled particles)
    const SETTLE_DAMP = 0.90;    // extra damp for very slow in-pool particles
    const OBS_REST = 0.3;     // obstacle restitution

    // ── Air vs Water & Settle ────────────────────────────────
    const AIR_GRAV_MULT = 16.5;      // falling in air is faster
    const DAMP_AIR = 0.998;    // preserve momentum in air
    const DAMP_WATER = 0.985;    // settling damping in water
    const COOLDOWN_MS = 1200;     // time after mouse move to trigger settle mode
    const MAXV_ACTIVE = 1400;     // higher speed allowed while juggling
    const MAXV_REST = 600;      // lower speed for stability at rest

    // ── Jitter Reduction ─────────────────────────────────────
    const CONTACT_REST = 0.05;     // inelastic contacts (0.0=stick, 1.0=bounce)
    const VISCOSITY = 0.02;     // XSPH smoothing factor
    const MAX_POS_CORR = r * 0.15; // limit positional push per iteration
    const COHESION = 0.005;    // subtle background attraction at rest
    const PARTICLE_GAP = r * 0.2;  // extra space between circles

    // ── Obstacle / interaction ────────────────────────────────
    const MIN_RADIUS = obsR * 0.8;  // resting hover radius
    const MAX_RADIUS = obsR;        // max radius at full swipe speed
    const VEL_SENSITIVITY = 0.02;       // how much mouse speed grows the radius


    // ── Cleanup tracking ──
    const rms: (() => void)[] = [];
    let dead = false;
    const listen = (el: EventTarget, t: string, fn: EventListener, p?: boolean) => {
        const o = p ? { passive: true } as AddEventListenerOptions : undefined;
        el.addEventListener(t, fn, o);
        rms.push(() => el.removeEventListener(t, fn, o));
    };

    // ── WebGL ──
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
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
        throw new Error(`Link: ${gl.getProgramInfoLog(prog)}`);

    const uRes = gl.getUniformLocation(prog, 'u_res')!;
    const uPt = gl.getUniformLocation(prog, 'u_ptSz')!;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const pBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, pBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
    const aP = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(aP);
    gl.vertexAttribPointer(aP, 2, gl.FLOAT, false, 0, 0);

    const cBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
    const aC = gl.getAttribLocation(prog, 'a_color');
    gl.enableVertexAttribArray(aC);
    gl.vertexAttribPointer(aC, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    rms.push(() => {
        gl.deleteProgram(prog); gl.deleteShader(vs); gl.deleteShader(fs);
        gl.deleteBuffer(pBuf); gl.deleteBuffer(cBuf); gl.deleteVertexArray(vao);
    });

    // ── Simulation state ──
    const simW = canvas.clientWidth;
    const simH = canvas.clientHeight;
    const h = Math.max(10, 2 * r);
    const nx = Math.floor(simW / h) + 1;
    const ny = Math.floor(simH / h) + 1;
    const poolTopY = simH * 0.50;

    const px = new Float32Array(N);
    const py = new Float32Array(N);
    const vx = new Float32Array(N);
    const vy = new Float32Array(N);
    const col = new Float32Array(N * 3);

    const uLen = (nx + 1) * ny, vLen = nx * (ny + 1), cLen = nx * ny;
    const U = new Float32Array(uLen), V = new Float32Array(vLen);
    // preU/preV: post-P2G, pre-projection snapshot — required for correct FLIP delta.
    // Preallocated once; filled with .set() each frame (zero allocations).
    const preU = new Float32Array(uLen), preV = new Float32Array(vLen);
    // prevU/prevV: previous-frame snapshot (kept for optional future use only).
    const pU = new Float32Array(uLen), pV = new Float32Array(vLen);
    const wU = new Float32Array(uLen), wV = new Float32Array(vLen);
    const P = new Float32Array(cLen);
    const CT = new Int32Array(cLen);

    // spatial hash
    const hCS = 2 * r;
    const hW = Math.ceil(simW / hCS) + 1, hH = Math.ceil(simH / hCS) + 1;
    const hN = hW * hH;
    const hCnt = new Int32Array(hN), hSt = new Int32Array(hN), hSo = new Int32Array(N);

    // ── Index helpers ──
    const iU = (i: number, j: number) => i + j * (nx + 1);
    const iV = (i: number, j: number) => i + j * nx;
    const iC = (i: number, j: number) => i + j * nx;

    // ── Init particles (hex-packed at bottom) ──
    function initParticles() {
        const sp = 2 * r, rh = sp * 0.866;
        let idx = 0, row = 0;
        const bot = simH - r;
        while (idx < N) {
            const y = bot - row * rh;
            if (y < r) break;
            const xo = (row % 2) * r;
            for (let x = r + xo; x <= simW - r && idx < N; x += sp) {
                px[idx] = x; py[idx] = y; vx[idx] = 0; vy[idx] = 0;
                col[idx * 3] = 1.0; col[idx * 3 + 1] = 1.0; col[idx * 3 + 2] = 1.0;
                idx++;
            }
            row++;
        }
        while (idx < N) {
            px[idx] = r + Math.random() * (simW - 2 * r);
            py[idx] = simH * 0.6 + Math.random() * simH * 0.35;
            vx[idx] = 0; vy[idx] = 0;
            col[idx * 3] = 1.0; col[idx * 3 + 1] = 1.0; col[idx * 3 + 2] = 1.0;
            idx++;
        }
        U.fill(0); V.fill(0); preU.fill(0); preV.fill(0); pU.fill(0); pV.fill(0); P.fill(0);
    }
    initParticles();

    // ── Bilinear samplers ──
    function smpU(arr: Float32Array, x: number, y: number) {
        const gx = x / h, gy = y / h - 0.5;
        let i0 = Math.floor(gx), j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, nx - 1));
        j0 = Math.max(0, Math.min(j0, ny - 2));
        const fx = Math.max(0, Math.min(1, gx - i0)), fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * arr[iU(i0, j0)] + fx * (1 - fy) * arr[iU(i0 + 1, j0)]
            + (1 - fx) * fy * arr[iU(i0, j0 + 1)] + fx * fy * arr[iU(i0 + 1, j0 + 1)];
    }
    function smpV(arr: Float32Array, x: number, y: number) {
        const gx = x / h - 0.5, gy = y / h;
        let i0 = Math.floor(gx), j0 = Math.floor(gy);
        i0 = Math.max(0, Math.min(i0, nx - 2));
        j0 = Math.max(0, Math.min(j0, ny - 1));
        const fx = Math.max(0, Math.min(1, gx - i0)), fy = Math.max(0, Math.min(1, gy - j0));
        return (1 - fx) * (1 - fy) * arr[iV(i0, j0)] + fx * (1 - fy) * arr[iV(i0 + 1, j0)]
            + (1 - fx) * fy * arr[iV(i0, j0 + 1)] + fx * fy * arr[iV(i0 + 1, j0 + 1)];
    }

    // ── Obstacle / pointer state ───────────────────────────────
    let obsX = simW * 0.5;   // obstacle centre (sim units)
    let obsY = simH * 0.5;
    let curObsR = 0;            // current radius (0 = inactive)
    let obsVelX = 0;            // velocity stamped onto the grid
    let obsVelY = 0;
    let mouseDown = false;
    let prevMouseX: number | null = null;
    let prevMouseY: number | null = null;
    let prevTime: number | null = null;
    let mouseVelocity = 0;
    let lastMoveTime = 0;           // performance.now() of last drag/move
    let idleTimer = 0;            // setTimeout id for 100ms idle
    let shrinkRaf = 0;            // rAF id for radius-shrink animation

    /** Convert a client-space point to simulation space. */
    function toSim(cx: number, cy: number) {
        const rc = canvas.getBoundingClientRect();
        return {
            x: (cx - rc.left) / rc.width * simW,
            y: (cy - rc.top) / rc.height * simH,
        };
    }

    /**
     * Place / move the invisible circular obstacle at sim-space (sx, sy).
     * Computes obstacle velocity from displacement and stamps it onto every
     * grid face inside the circle — this is the engine of the juggle.
     * reset=true: snap with zero velocity (first contact).
     */
    function setObstacle(sx: number, sy: number, reset: boolean) {
        if (reset) {
            obsVelX = 0; obsVelY = 0;
        } else {
            obsVelX = (sx - obsX) / DT;
            obsVelY = (sy - obsY) / DT;
        }
        obsX = sx; obsY = sy;
        lastMoveTime = performance.now();
    }

    /**
     * Called inside step(), AFTER p2g() normalises U/V and BEFORE the
     * pre-projection snapshot.  Marks obstacle cells SOLID and stamps
     * obsVelX/obsVelY onto every grid face inside the obstacle circle.
     * This is equivalent to the reference setObstacle ‘f.u[i*n+j] = vx’ writes.
     */
    function stampObstacle() {
        if (curObsR <= 0) return;
        const r2 = curObsR * curObsR;
        for (let j = 1; j < ny - 1; j++) {
            for (let i = 1; i < nx - 1; i++) {
                const dx = (i + 0.5) * h - obsX;
                const dy = (j + 0.5) * h - obsY;
                if (dx * dx + dy * dy < r2) {
                    CT[iC(i, j)] = SOLID;
                    U[iU(i, j)] = obsVelX;
                    U[iU(i + 1, j)] = obsVelX;
                    V[iV(i, j)] = obsVelY;
                    V[iV(i, j + 1)] = obsVelY;
                }
            }
        }
    }

    // ── Step ──
    function step() {
        pU.set(U); pV.set(V);

        integrate(); bounds(); separate();
        classify();
        p2g();

        // Stamp obstacle velocity onto the grid AFTER p2g normalization,
        // BEFORE pre-projection snapshot — this is how the juggle energy enters the fluid.
        stampObstacle();

        preU.set(U); preV.set(V);
        pressure();
        g2p();

        clampV(); sleep(); colors();
    }

    function integrate() {
        const now = performance.now();
        const isSettling = (now - lastMoveTime > COOLDOWN_MS) && !mouseDown;

        for (let i = 0; i < N; i++) {
            const isAir = py[i] < poolTopY - r;
            const damp = isAir ? DAMP_AIR : DAMP_WATER;

            vx[i] *= damp; vy[i] *= damp;

            // Gravity: boosted in air or during settling to return faster
            let gMult = 1.0;
            if (isAir) gMult = AIR_GRAV_MULT;
            if (isSettling && py[i] < poolTopY + 100) gMult *= 1.5; // push down faster when settling

            vy[i] += grav * gMult * DT;

            if (isAir) {
                // Return force: slightly stronger if we are very far up
                vy[i] += RETF * DT * (py[i] < 0 ? 2.0 : 1.0);
            } else {
                // In water: damp only very slow particles
                const spd = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
                if (spd < 40) { vx[i] *= SETTLE_DAMP; vy[i] *= SETTLE_DAMP; }
            }
            px[i] += vx[i] * DT; py[i] += vy[i] * DT;
        }
    }

    function bounds() {
        const mn = r, mxX = simW - r, mxY = simH - r;
        for (let i = 0; i < N; i++) {

            // ── Circular obstacle collision ──────────────────────────
            // Push particle to the surface of the invisible obstacle circle,
            // then reflect its inward velocity component with 70% energy loss.
            if (curObsR > 0) {
                const ddx = px[i] - obsX, ddy = py[i] - obsY;
                const d2 = ddx * ddx + ddy * ddy;
                const minD = curObsR + r;            // centre-to-centre minimum
                if (d2 < minD * minD && d2 > 1e-8) {
                    const d = Math.sqrt(d2);
                    const nx = ddx / d, ny = ddy / d;
                    // push particle to surface
                    px[i] = obsX + nx * minD;
                    py[i] = obsY + ny * minD;
                    // reflect inward normal component, absorb 70% energy
                    const vn = vx[i] * nx + vy[i] * ny;
                    if (vn < 0) {
                        vx[i] -= (1 + OBS_REST) * vn * nx;
                        vy[i] -= (1 + OBS_REST) * vn * ny;
                    }
                    // inherit obstacle velocity (adds the juggle impulse)
                    vx[i] += obsVelX * OBS_REST;
                    vy[i] += obsVelY * OBS_REST;
                }
            }

            // ── Tank walls ──────────────────────────────────────────
            if (px[i] < mn) { px[i] = mn; vx[i] = -vx[i] * 0.1; vy[i] *= (1 - WFRIC); }
            if (px[i] > mxX) { px[i] = mxX; vx[i] = -vx[i] * 0.1; vy[i] *= (1 - WFRIC); }
            if (py[i] < mn) { py[i] = mn; vy[i] = Math.abs(vy[i]) * 0.15; vx[i] *= (1 - WFRIC); }
            if (py[i] > mxY) { py[i] = mxY; vy[i] = 0; vx[i] *= (1 - WFRIC); }
        }
    }



    function separate() {
        const md = 2 * r + PARTICLE_GAP;
        const md2 = md * md;

        const now = performance.now();
        const isSettling = (now - lastMoveTime > COOLDOWN_MS) && !mouseDown;

        for (let it = 0; it < sIt; it++) {
            hCnt.fill(0);
            for (let i = 0; i < N; i++) {
                const ci = Math.max(0, Math.min(hW - 1, Math.floor(px[i] / hCS)));
                const cj = Math.max(0, Math.min(hH - 1, Math.floor(py[i] / hCS)));
                hCnt[ci + cj * hW]++;
            }
            hSt[0] = 0;
            for (let i = 1; i < hN; i++) hSt[i] = hSt[i - 1] + hCnt[i - 1];
            const tmp = new Int32Array(hN);
            for (let i = 0; i < N; i++) {
                const ci = Math.max(0, Math.min(hW - 1, Math.floor(px[i] / hCS)));
                const cj = Math.max(0, Math.min(hH - 1, Math.floor(py[i] / hCS)));
                const c = ci + cj * hW;
                hSo[hSt[c] + tmp[c]] = i; tmp[c]++;
            }

            for (let p = 0; p < N; p++) {
                const ci = Math.max(0, Math.min(hW - 1, Math.floor(px[p] / hCS)));
                const cj = Math.max(0, Math.min(hH - 1, Math.floor(py[p] / hCS)));

                let avgVX = 0, avgVY = 0, count = 0;

                for (let dj = -1; dj <= 1; dj++) {
                    const nj = cj + dj; if (nj < 0 || nj >= hH) continue;
                    for (let di = -1; di <= 1; di++) {
                        const ni = ci + di; if (ni < 0 || ni >= hW) continue;
                        const c = ni + nj * hW, end = hSt[c] + hCnt[c];
                        for (let k = hSt[c]; k < end; k++) {
                            const q = hSo[k];
                            if (q === p) continue;

                            const dx = px[q] - px[p], dy = py[q] - py[p], d2 = dx * dx + dy * dy;
                            if (d2 < md2 && d2 > 1e-8) {
                                const d = Math.sqrt(d2);

                                // 1) Position Correction (clamped)
                                let ov = (md - d) * 0.5;
                                ov = Math.min(ov, MAX_POS_CORR);
                                const nx = dx / d, ny = dy / d;
                                px[p] -= ov * nx; py[p] -= ov * ny;
                                px[q] += ov * nx; py[q] += ov * ny;

                                // 2) Inelastic Contact Impulse
                                // Compute relative normal velocity
                                const rvx = vx[p] - vx[q], rvy = vy[p] - vy[q];
                                const rvn = rvx * nx + rvy * ny;
                                if (rvn > 0) { // Particles are approaching
                                    // Target post-collision relative normal velocity = -rvn * CONTACT_REST
                                    // Impulse J = -(1 + restitution) * rvn / 2
                                    const J = -(1 + CONTACT_REST) * rvn * 0.5;
                                    vx[p] += J * nx; vy[p] += J * ny;
                                    vx[q] -= J * nx; vy[q] -= J * ny;
                                }

                                // Accumulate for XSPH Smoothing (optional pass later)
                                avgVX += vx[q]; avgVY += vy[q]; count++;

                                // Subtle neighbor attraction (cohesion)
                                if (d > md * 0.7) {
                                    const strength = COHESION * (isSettling ? 2 : 1);
                                    vx[p] += dx * strength; vy[p] += dy * strength;
                                    vx[q] -= dx * strength; vy[q] -= dy * strength;
                                }
                            }
                        }
                    }
                }


                // 3) XSPH Smoothing (Velocity Viscosity)
                if (count > 0) {
                    const viscosity = isSettling ? VISCOSITY * 2 : VISCOSITY;
                    vx[p] += viscosity * (avgVX / count - vx[p]);
                    vy[p] += viscosity * (avgVY / count - vy[p]);
                }
            }
        }
    }

    function classify() {
        CT.fill(AIR);
        for (let i = 0; i < nx; i++) { CT[iC(i, 0)] = SOLID; CT[iC(i, ny - 1)] = SOLID; }
        for (let j = 0; j < ny; j++) { CT[iC(0, j)] = SOLID; CT[iC(nx - 1, j)] = SOLID; }
        // Obstacle cells are marked SOLID in stampObstacle() (after p2g), not here,
        // so that p2g zero-clears them and stampObstacle can then write the velocity.
        for (let p = 0; p < N; p++) {
            const ci = Math.floor(px[p] / h), cj = Math.floor(py[p] / h);
            if (ci >= 1 && ci < nx - 1 && cj >= 1 && cj < ny - 1 && CT[iC(ci, cj)] !== SOLID)
                CT[iC(ci, cj)] = FLUID;
        }
    }

    function p2g() {
        U.fill(0); V.fill(0); wU.fill(0); wV.fill(0);
        for (let p = 0; p < N; p++) {
            const ppx = px[p], ppy = py[p], pvx = vx[p], pvy = vy[p];
            { // U
                const gx = ppx / h, gy = ppy / h - 0.5;
                let i0 = Math.floor(gx), j0 = Math.floor(gy);
                i0 = Math.max(0, Math.min(i0, nx - 1)); j0 = Math.max(0, Math.min(j0, ny - 2));
                const fx = Math.max(0, Math.min(1, gx - i0)), fy = Math.max(0, Math.min(1, gy - j0));
                const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                const a = iU(i0, j0), b = iU(i0 + 1, j0), c = iU(i0, j0 + 1), d = iU(i0 + 1, j0 + 1);
                U[a] += w00 * pvx; wU[a] += w00; U[b] += w10 * pvx; wU[b] += w10;
                U[c] += w01 * pvx; wU[c] += w01; U[d] += w11 * pvx; wU[d] += w11;
            }
            { // V
                const gx = ppx / h - 0.5, gy = ppy / h;
                let i0 = Math.floor(gx), j0 = Math.floor(gy);
                i0 = Math.max(0, Math.min(i0, nx - 2)); j0 = Math.max(0, Math.min(j0, ny - 1));
                const fx = Math.max(0, Math.min(1, gx - i0)), fy = Math.max(0, Math.min(1, gy - j0));
                const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                const a = iV(i0, j0), b = iV(i0 + 1, j0), c = iV(i0, j0 + 1), d = iV(i0 + 1, j0 + 1);
                V[a] += w00 * pvy; wV[a] += w00; V[b] += w10 * pvy; wV[b] += w10;
                V[c] += w01 * pvy; wV[c] += w01; V[d] += w11 * pvy; wV[d] += w11;
            }
        }
        for (let i = 0; i < U.length; i++) U[i] = wU[i] > 1e-6 ? U[i] / wU[i] : 0;
        for (let i = 0; i < V.length; i++) V[i] = wV[i] > 1e-6 ? V[i] / wV[i] : 0;
        // Zero wall-cell faces (obstacle faces are zeroed then re-stamped by stampObstacle)
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
            if (CT[iC(i, j)] === SOLID) {
                U[iU(i, j)] = 0; U[iU(i + 1, j)] = 0; V[iV(i, j)] = 0; V[iV(i, j + 1)] = 0;
            }
    }

    function pressure() {
        P.fill(0);
        for (let it = 0; it < pIt; it++)
            for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
                if (CT[iC(i, j)] !== FLUID) continue;
                const div = (U[iU(i + 1, j)] - U[iU(i, j)] + V[iV(i, j + 1)] - V[iV(i, j)]) / h;
                const sl = CT[iC(i - 1, j)] !== SOLID ? 1 : 0, sr = CT[iC(i + 1, j)] !== SOLID ? 1 : 0;
                const sb = CT[iC(i, j - 1)] !== SOLID ? 1 : 0, st = CT[iC(i, j + 1)] !== SOLID ? 1 : 0;
                const s = sl + sr + sb + st; if (!s) continue;
                const ps = sl * P[iC(i - 1, j)] + sr * P[iC(i + 1, j)] + sb * P[iC(i, j - 1)] + st * P[iC(i, j + 1)];
                P[iC(i, j)] += omega * ((ps - div * h) / s - P[iC(i, j)]);
            }
        for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx; i++) {
            if (CT[iC(i - 1, j)] === SOLID || CT[iC(i, j)] === SOLID) continue;
            U[iU(i, j)] -= (P[iC(i, j)] - P[iC(i - 1, j)]) / h;
        }
        for (let j = 1; j < ny; j++) for (let i = 1; i < nx - 1; i++) {
            if (CT[iC(i, j - 1)] === SOLID || CT[iC(i, j)] === SOLID) continue;
            V[iV(i, j)] -= (P[iC(i, j)] - P[iC(i, j - 1)]) / h;
        }
    }

    function g2p() {
        // PIC/FLIP blend:
        //   v_pic     = sample(U_post, p)          — fully grid-derived, dissipative
        //   deltaGrid = sample(U_post, p) − sample(U_pre, p)  — what projection added
        //   v_flip    = v_particle_old + deltaGrid  — keeps particle detail
        //   v_new     = (1−fRat)*v_pic + fRat*v_flip
        for (let i = 0; i < N; i++) {
            const x = px[i], y = py[i];
            const picU = smpU(U, x, y), picV = smpV(V, x, y);
            const dU = picU - smpU(preU, x, y);
            const dV = picV - smpV(preV, x, y);
            vx[i] = (1 - fRat) * picU + fRat * (vx[i] + dU);
            vy[i] = (1 - fRat) * picV + fRat * (vy[i] + dV);
        }
    }

    function clampV() {
        const maxV = mouseDown ? MAXV_ACTIVE : MAXV_REST;
        const m2 = maxV * maxV;
        for (let i = 0; i < N; i++) {
            const v2 = vx[i] * vx[i] + vy[i] * vy[i];
            if (v2 > m2) { const s = maxV / Math.sqrt(v2); vx[i] *= s; vy[i] *= s; }
        }
    }

    function sleep() {
        const s2 = SLEEPV * SLEEPV;
        for (let i = 0; i < N; i++)
            if (vx[i] * vx[i] + vy[i] * vy[i] < s2 && py[i] > poolTopY) { vx[i] = 0; vy[i] = 0; }
    }

    function colors() {
        // Static colour — obstacle is invisible; velocity-based tints would reveal it.
        for (let i = 0; i < N; i++) {
            col[i * 3] = 1.0; col[i * 3 + 1] = 1.0; col[i * 3 + 2] = 1.0;
        }
    }

    // ── Interaction event handlers ────────────────────────────────

    function startDrag(cx: number, cy: number) {
        cancelAnimationFrame(shrinkRaf);
        mouseDown = true;
        prevMouseX = cx; prevMouseY = cy; prevTime = performance.now();
        const s = toSim(cx, cy);
        curObsR = MIN_RADIUS;
        setObstacle(s.x, s.y, true);
    }

    function endDrag() {
        mouseDown = false;
        prevMouseX = null; prevMouseY = null; prevTime = null;
        obsVelX = 0; obsVelY = 0;
        // Smoothly shrink radius to zero so the obstacle fades out gracefully.
        function shrink() {
            if (dead) return;
            curObsR *= 0.9;
            if (curObsR > 0.5) { shrinkRaf = requestAnimationFrame(shrink); }
            else { curObsR = 0; }
        }
        shrinkRaf = requestAnimationFrame(shrink);
    }

    function drag(cx: number, cy: number) {
        // Velocity-based dynamic radius: fast swipe = bigger obstacle.
        const now = performance.now();
        if (prevMouseX !== null && prevMouseY !== null && prevTime !== null) {
            const dt = (now - prevTime) / 1000;
            if (dt > 0) {
                const rc = canvas.getBoundingClientRect();
                const scX = simW / rc.width;
                const scY = simH / rc.height;
                const ddx = (cx - prevMouseX) * scX;
                const ddy = (cy - prevMouseY) * scY;
                mouseVelocity = Math.sqrt(ddx * ddx + ddy * ddy) / dt;
                const target = Math.min(MAX_RADIUS, MIN_RADIUS + mouseVelocity * VEL_SENSITIVITY);
                curObsR = curObsR * 0.7 + target * 0.3;   // smooth blend
            }
        }
        prevMouseX = cx; prevMouseY = cy; prevTime = now;

        if (mouseDown) {
            const s = toSim(cx, cy);
            setObstacle(s.x, s.y, false);
        } else {
            startDrag(cx, cy);
        }
    }

    // Idle timer: stop stamping velocity 100ms after last move
    function resetIdle() {
        clearTimeout(idleTimer);
        idleTimer = window.setTimeout(() => { obsVelX = 0; obsVelY = 0; }, 100);
    }

    listen(wrapper, 'mousedown', ((e: MouseEvent) => startDrag(e.clientX, e.clientY)) as EventListener);
    listen(window, 'mouseup', (endDrag) as EventListener);
    listen(window, 'mousemove', ((e: MouseEvent) => { drag(e.clientX, e.clientY); resetIdle(); }) as EventListener);
    listen(wrapper, 'mouseenter', ((e: MouseEvent) => startDrag(e.clientX, e.clientY)) as EventListener);
    listen(window, 'blur', (endDrag) as EventListener);
    listen(wrapper, 'touchstart', ((e: TouchEvent) => { e.preventDefault(); startDrag(e.touches[0].clientX, e.touches[0].clientY); }) as EventListener);
    listen(wrapper, 'touchmove', ((e: TouchEvent) => { e.preventDefault(); drag(e.touches[0].clientX, e.touches[0].clientY); }) as EventListener);
    listen(wrapper, 'touchend', ((e: TouchEvent) => { e.preventDefault(); endDrag(); }) as EventListener);


    // ── Render ──
    const posD = new Float32Array(N * 2);
    function render() {
        const cvs = gl!.canvas as HTMLCanvasElement;
        if (cvs.width !== cvs.clientWidth || cvs.height !== cvs.clientHeight) {
            cvs.width = cvs.clientWidth; cvs.height = cvs.clientHeight;
        }
        gl!.viewport(0, 0, cvs.width, cvs.height);
        gl!.clearColor(0, 0, 0, 0);
        gl!.clear(gl!.COLOR_BUFFER_BIT);
        gl!.enable(gl!.BLEND);
        gl!.blendFunc(gl!.SRC_ALPHA, gl!.ONE_MINUS_SRC_ALPHA);

        for (let i = 0; i < N; i++) { posD[i * 2] = px[i]; posD[i * 2 + 1] = py[i]; }
        gl!.bindBuffer(gl!.ARRAY_BUFFER, pBuf);
        gl!.bufferData(gl!.ARRAY_BUFFER, posD, gl!.DYNAMIC_DRAW);
        gl!.bindBuffer(gl!.ARRAY_BUFFER, cBuf);
        gl!.bufferData(gl!.ARRAY_BUFFER, col.subarray(0, N * 3), gl!.DYNAMIC_DRAW);

        const ptSz = r * 2 * (cvs.width / simW);
        gl!.useProgram(prog);
        gl!.uniform2f(uRes, simW, simH);
        gl!.uniform1f(uPt, ptSz);
        gl!.bindVertexArray(vao);
        gl!.drawArrays(gl!.POINTS, 0, N);
        gl!.bindVertexArray(null);
    }

    // ── Loop ──
    let paused = false;
    let rafId = 0;
    function loop() {
        if (!dead) {
            if (!paused) { for (let s = 0; s < SUB; s++) step(); }
            render();
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
    }

    return {
        cleanup,
        reset: initParticles,
        setFlipRatio: (v: number) => { fRat = v; },
        setPaused: (v: boolean) => { paused = v; },
    };
}
