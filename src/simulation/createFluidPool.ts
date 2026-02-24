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
in float a_shape;      // 0=circle 1=square 2=plus 3=cross 4=triangle
uniform vec2 u_res;
uniform float u_ptSz;
out vec3 v_col;
flat out int v_shape;
void main(){
  gl_Position=vec4(a_pos.x/u_res.x*2.0-1.0,1.0-a_pos.y/u_res.y*2.0,0,1);
  // Decrease size of circle (0) and square (1) by 4px as requested
  float finalSz = u_ptSz;
  if(a_shape < 1.5) finalSz = max(2.0, u_ptSz - 4.0);
  gl_PointSize=finalSz;
  v_col=a_color;
  v_shape=int(a_shape);
}`;
const FRAG = `#version 300 es
precision mediump float;
in vec3 v_col;
flat in int v_shape;
out vec4 o;

float sdEquilateralTriangle(in vec2 p, in float r ) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x+k*p.y>0.0 ) p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p)*sign(p.y);
}

void main(){
  vec2 uv=(gl_PointCoord-0.5)*2.0;
  float alpha=0.0;
  vec3 finalCol=v_col;

  if(v_shape==0){
    // circle
    float d=length(uv);
    alpha=1.0-smoothstep(0.82,1.0,d);
  } else if(v_shape==1){
    // square
    float d=max(abs(uv.x),abs(uv.y));
    alpha=1.0-smoothstep(0.82,0.98,d);
  } else if(v_shape==2){
    // plus (+): thin arms
    float arm=0.08;
    float dH=max(abs(uv.y)-arm, max(abs(uv.x)-1.0,0.0));
    float dV=max(abs(uv.x)-arm, max(abs(uv.y)-1.0,0.0));
    float d=min(dH,dV);
    d=max(d,length(uv)-1.0);
    alpha=1.0-smoothstep(0.0,0.04,d);
  } else if(v_shape==3){
    // cross (X): thin diagonal arms
    vec2 r45=vec2((uv.x+uv.y)*0.7071,(uv.x-uv.y)*0.7071);
    float arm=0.08;
    float dH=max(abs(r45.y)-arm, max(abs(r45.x)-0.7071,0.0));
    float dV=max(abs(r45.x)-arm, max(abs(r45.y)-0.7071,0.0));
    float d=min(dH,dV);
    d=max(d,length(uv)-1.0);
    alpha=1.0-smoothstep(0.0,0.04,d);
  } else {
    // triangle: white fill
    float d=sdEquilateralTriangle(uv, 0.85);
    alpha=1.0-smoothstep(0.0,0.04,d);
    finalCol=vec3(1.0);
  }
  if(alpha<0.01)discard;
  o=vec4(finalCol,alpha*0.92);
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
    const N = opts?.numParticles ?? 2800;
    const grav = opts?.gravity ?? 200;          // ↑ stronger gravity for big splashes
    let fRat = opts?.flipRatio ?? 0.9;
    const DT = opts?.dt ?? 1 / 120;
    const pIt = opts?.pressureIters ?? 30;
    const sIt = opts?.sepIters ?? 2;
    const omega = opts?.overRelax ?? 1.9;
    const obsR = opts?.obstacleRadius ?? 100;

    const SUB = 2;
    const WFRIC = 0.5;     // wall friction — less grip = more splash bounce
    const RETF = 350;     // pull-back above pool
    const SLEEPV = 3;       // sleep threshold (settled particles)
    const SETTLE_DAMP = 0.90;    // extra damp for very slow in-pool particles
    const OBS_REST = 0.7;     // obstacle restitution — bouncier (was 0.3)

    // ── Air vs Water & Settle ────────────────────────────────
    const AIR_GRAV_MULT = 28.0;      // ↑ fast free-fall for big flip arc
    const DAMP_AIR = 0.999;    // almost no air drag → farther projectiles
    const DAMP_WATER = 0.998;    // less damping in water — waves travel farther
    const COOLDOWN_MS = 500;     // time after mouse move to trigger settle mode
    const MAXV_ACTIVE = 3200;     // very fast during juggle
    const MAXV_REST = 2200;       // higher resting cap so waves propagate wide

    // ── Jitter Reduction ─────────────────────────────────────
    // const CONTACT_REST = 0.05; // Omitted: restitution disabled in separate() for stability
    const VISCOSITY = 0.02;     // XSPH smoothing factor
    const MAX_POS_CORR = r * 0.15; // limit positional push per iteration
    const COHESION = 0.005;    // subtle background attraction at rest
    const PARTICLE_GAP = r * 0.2;  // extra space between circles

    // ── XSPH Viscosity (adaptive) ──────────────────────────────
    const VISC_SPARSE = 0.0;    // near gaps / void edges: let particles flow freely
    const VISC_DENSE = 0.04;   // packed pile: damp jitter
    const VISC_DENSE_THRESH = 7;      // neighbour count above which dense coeff kicks in
    const V_RADIUS = r * 3.0;// smoothing neighbourhood radius
    const V_MAX_FIX = 200;    // max velocity correction per frame

    // ── Heightfield re-leveling (B1) ──────────────────────────
    // Divides pool into columns; pushes fluid from overfull→underfull columns.
    const LEVEL_COLS = 32;    // vertical bins across X
    const LEVEL_K_REST = 8.0;   // always-on leveling — strong so voids close fast
    const LEVEL_K_REFILL = 20.0;  // burst leveling in REFILL state
    const LEVEL_MAX_AX = 600;   // max lateral accel px/s² (was 80)
    const REFILL_MS = 500;  // REFILL lasts 1s after interaction ends


    // ── Obstacle / interaction ────────────────────────────────
    const MIN_RADIUS = obsR * 0.6;  // resting hover radius (smaller = more surgical)
    const MAX_RADIUS = obsR * 1.4;  // ↑ large radius at full speed = big wave wall
    const VEL_SENSITIVITY = 0.04;   // ↑ more radius growth per speed unit


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

    // Shape-type attribute buffer (float per particle: 0=circle,1=square,2=plus,3=cross)
    const sBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, sBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
    const aS = gl.getAttribLocation(prog, 'a_shape');
    gl.enableVertexAttribArray(aS);
    gl.vertexAttribPointer(aS, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    rms.push(() => {
        gl.deleteProgram(prog); gl.deleteShader(vs); gl.deleteShader(fs);
        gl.deleteBuffer(pBuf); gl.deleteBuffer(cBuf); gl.deleteBuffer(sBuf);
        gl.deleteVertexArray(vao);
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
    // Per-particle shape type: 0=circle, 1=square, 2=plus, 3=cross
    const shapeType = new Float32Array(N);
    const shapeD = new Float32Array(N); // upload scratch

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
                shapeType[idx] = idx % 5; // uniform distribution across 5 shapes
                idx++;
            }
            row++;
        }
        while (idx < N) {
            px[idx] = r + Math.random() * (simW - 2 * r);
            py[idx] = simH * 0.6 + Math.random() * simH * 0.35;
            vx[idx] = 0; vy[idx] = 0;
            col[idx * 3] = 1.0; col[idx * 3 + 1] = 1.0; col[idx * 3 + 2] = 1.0;
            shapeType[idx] = idx % 5;
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
    let refillStart = 0;           // when REFILL state began
    let idleTimer = 0;            // setTimeout id for 100ms idle
    let shrinkRaf = 0;            // rAF id for radius-shrink animation
    // Sim interaction state: 'active' while mouse is down, 'refill' right after,
    // 'rest' once the pool has settled back. Controls leveling strength + damping.
    let simState: 'active' | 'refill' | 'rest' = 'rest';

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
        // ── Update sim state machine ──
        const nowMs = performance.now();
        if (mouseDown) {
            simState = 'active';
        } else if (simState === 'active') {
            // Transition: mouse just released → start REFILL
            simState = 'refill';
            refillStart = nowMs;
        } else if (simState === 'refill' && nowMs - refillStart > REFILL_MS) {
            simState = 'rest';
        }

        pU.set(U); pV.set(V);

        integrate();
        applyRelevelingForce(); // lateral pressure to close voids
        bounds(); separate();
        classify();
        p2g();

        // Stamp obstacle velocity onto the grid AFTER p2g normalization,
        // BEFORE pre-projection snapshot — this is how the juggle energy enters the fluid.
        stampObstacle();

        preU.set(U); preV.set(V);
        pressure();
        g2p();
        applyViscosity();

        clampV(); sleep(); colors();
    }

    function integrate() {
        const isRefill = simState === 'refill';
        const isRest = simState === 'rest';
        const isSettling = isRefill || (isRest && !mouseDown && (performance.now() - lastMoveTime > COOLDOWN_MS));

        for (let i = 0; i < N; i++) {
            const isAir = py[i] < poolTopY - r;
            const damp = isAir ? DAMP_AIR : DAMP_WATER;

            vx[i] *= damp; vy[i] *= damp;

            // Gravity: boosted in air; extra during refill so particles fall back fast
            let gMult = 1.0;
            if (isAir) gMult = AIR_GRAV_MULT;
            // During REFILL: pull airborne particles down faster to restore pool level
            if (isRefill && isAir) gMult *= 2.8;
            if (isSettling && py[i] < poolTopY + 100) gMult *= 1.5;

            vy[i] += grav * gMult * DT;

            if (isAir) {
                vy[i] += RETF * DT * (py[i] < 0 ? 2.0 : 1.0);
            } else {
                const spd = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
                // Stronger in-water damping during refill to settle the pile quickly
                const settleMult = isRefill ? 0.85 : SETTLE_DAMP;
                if (spd < 40) { vx[i] *= settleMult; vy[i] *= settleMult; }
            }
            px[i] += vx[i] * DT; py[i] += vy[i] * DT;
        }
    }

    /**
     * Heightfield-based lateral re-leveling (B1).
     * Divides the pool X axis into LEVEL_COLS bins. Columns with more particles
     * than average exert outward pressure; underfull columns pull fluid in.
     * This reproduces the "hand in water" backfill: wherever you push particles
     * away, a pressure gradient drives the rest back into the void.
     */
    function applyRelevelingForce() {
        // Only apply within the pool region (poolTopY → bottom)
        const poolH = simH - poolTopY;
        if (poolH <= 0) return;

        const colW = simW / LEVEL_COLS;
        const fill = new Float32Array(LEVEL_COLS);

        // count particles per column (pool region only)
        for (let i = 0; i < N; i++) {
            if (py[i] < poolTopY) continue;
            const col = Math.max(0, Math.min(LEVEL_COLS - 1, Math.floor(px[i] / colW)));
            fill[col]++;
        }

        // average fill
        let fillSum = 0;
        for (let c = 0; c < LEVEL_COLS; c++) fillSum += fill[c];
        const fillAvg = fillSum / LEVEL_COLS;
        if (fillAvg < 0.5) return; // pool nearly empty, nothing to do

        // pressure = k * (fill[c] - avg)  – positive = overfull, negative = underfull
        // Apply strongly in REFILL, moderately always (REST/ACTIVE) so gaps never linger
        const kLevel = simState === 'refill' ? LEVEL_K_REFILL : LEVEL_K_REST;

        // Pressure gradient per column: dpdx = p[c+1] - p[c-1]
        const grad = new Float32Array(LEVEL_COLS);
        for (let c = 0; c < LEVEL_COLS; c++) {
            const pL = (c > 0) ? kLevel * (fill[c - 1] - fillAvg) : 0;
            const pR = (c < LEVEL_COLS - 1) ? kLevel * (fill[c + 1] - fillAvg) : 0;
            grad[c] = pR - pL; // positive → more on right → push left
        }

        // Apply acceleration to each particle in the pool
        for (let i = 0; i < N; i++) {
            if (py[i] < poolTopY) continue;
            const col = Math.max(0, Math.min(LEVEL_COLS - 1, Math.floor(px[i] / colW)));
            // Acceleration: -∇p  (move from high→low fill)
            let ax = -grad[col];
            // Clamp so this doesn't override strong splash velocities
            ax = Math.max(-LEVEL_MAX_AX, Math.min(LEVEL_MAX_AX, ax));
            vx[i] += ax * DT;
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

            // ── Tank walls ────────────────────────────────────────────
            // Raised restitution so waves bounce energetically off walls (was 0.1)
            if (px[i] < mn) { px[i] = mn; vx[i] = -vx[i] * 0.65; vy[i] *= (1 - WFRIC); }
            if (px[i] > mxX) { px[i] = mxX; vx[i] = -vx[i] * 0.65; vy[i] *= (1 - WFRIC); }
            if (py[i] < mn) { py[i] = mn; vy[i] = Math.abs(vy[i]) * 0.50; vx[i] *= (1 - WFRIC); }
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
                                // Soft correction (0.2 weight) over multiple steps/iterations
                                // prevents high-frequency glittering while still ensuring 
                                // particles stack and repel from their outer surfaces.
                                let ov = (md - d) * 0.2;
                                ov = Math.min(ov, MAX_POS_CORR);
                                const nx = dx / d, ny = dy / d;
                                px[p] -= ov * nx; py[p] -= ov * ny;
                                px[q] += ov * nx; py[q] += ov * ny;

                                // 2) Inelastic Contact Impulse
                                // Compute relative normal velocity
                                // 2) Velocity impulse: DISABLED
                                // Removing the contact impulse entirely eliminates the
                                // high-frequency glittering/sparkling in dense regions.
                                // Position correction (above) still prevents overlap;
                                // the grid pressure solve handles flow forces instead.
                                // const rvn = ...; — intentionally omitted.

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
            const isAir = y < poolTopY - r;

            // Particles in air regime skip the grid projection update to avoid stalling
            if (isAir) continue;

            const picU = smpU(U, x, y), picV = smpV(V, x, y);
            const dU = picU - smpU(preU, x, y);
            const dV = picV - smpV(preV, x, y);
            vx[i] = (1 - fRat) * picU + fRat * (vx[i] + dU);
            vy[i] = (1 - fRat) * picV + fRat * (vy[i] + dV);
        }
    }

    function applyViscosity() {
        // Adaptive XSPH:
        //   Sparse zones (few neighbours) → VISC_SPARSE ≈ 0
        //     Particles near voids stay free to flow back in.
        //   Dense zones (many neighbours) → VISC_DENSE
        //     Packed pile gets damping to stop jitter.
        const vr = V_RADIUS, vr2 = vr * vr;

        for (let i = 0; i < N; i++) {
            if (py[i] < poolTopY - r) continue; // skip clearly airborne

            const ci = Math.max(0, Math.min(hW - 1, Math.floor(px[i] / hCS)));
            const cj = Math.max(0, Math.min(hH - 1, Math.floor(py[i] / hCS)));

            let avgVX = 0, avgVY = 0, wsum = 0, count = 0;

            for (let dj = -1; dj <= 1; dj++) {
                const nj = cj + dj; if (nj < 0 || nj >= hH) continue;
                for (let di = -1; di <= 1; di++) {
                    const ni = ci + di; if (ni < 0 || ni >= hW) continue;
                    const cell = ni + nj * hW;
                    const end = hSt[cell] + hCnt[cell];
                    for (let k = hSt[cell]; k < end; k++) {
                        const j = hSo[k];
                        if (i === j) continue;
                        const dx = px[j] - px[i], dy = py[j] - py[i], d2 = dx * dx + dy * dy;
                        if (d2 < vr2) {
                            const d = Math.sqrt(d2);
                            const q = d / vr;
                            const w = (1 - q) * (1 - q);
                            avgVX += vx[j] * w; avgVY += vy[j] * w;
                            wsum += w; count++;
                        }
                    }
                }
            }

            if (wsum > 0) {
                // density factor: 0 = sparse (void edge), 1 = fully packed pile
                const denseFactor = Math.min(1, count / VISC_DENSE_THRESH);
                // During REFILL boost viscosity in dense regions to settle quickly;
                // keep sparse regions free so the leveling force can move particles.
                const refillBoost = simState === 'refill' ? 1.5 : 1.0;
                const strength = (VISC_SPARSE + (VISC_DENSE - VISC_SPARSE) * denseFactor) * refillBoost;

                let dvx = strength * (avgVX / wsum - vx[i]);
                let dvy = strength * (avgVY / wsum - vy[i]);

                const d2 = dvx * dvx + dvy * dvy;
                if (d2 > V_MAX_FIX * V_MAX_FIX) {
                    const s = V_MAX_FIX / Math.sqrt(d2);
                    dvx *= s; dvy *= s;
                }
                vx[i] += dvx; vy[i] += dvy;
            }
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
    // Pattern follows reference simulation:
    //   mousedown / mouseenter  → startDrag
    //   mousemove on wrapper    → drag  (stops the moment cursor leaves)
    //   mouseleave on wrapper   → instant endDrag (kills obstacle immediately)
    //   mouseup on window       → endDrag (handles button release outside wrapper)
    //   blur on window          → endDrag (tab switch)

    function startDrag(cx: number, cy: number) {
        cancelAnimationFrame(shrinkRaf);
        mouseDown = true;
        prevMouseX = cx; prevMouseY = cy; prevTime = performance.now();
        const s = toSim(cx, cy);
        curObsR = MIN_RADIUS;
        setObstacle(s.x, s.y, true);
    }

    /** Instantly kill the obstacle — called on leave so zero ghost circle remains. */
    function killObstacle() {
        cancelAnimationFrame(shrinkRaf);
        curObsR = 0;
        obsVelX = 0;
        obsVelY = 0;
    }

    function endDrag() {
        mouseDown = false;
        prevMouseX = null; prevMouseY = null; prevTime = null;
        obsVelX = 0; obsVelY = 0;
        // Smoothly shrink radius to zero so the obstacle fades out gracefully
        // when mouse is released INSIDE the canvas.
        function shrink() {
            if (dead) return;
            curObsR *= 0.75;          // faster shrink than before
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

        const s = toSim(cx, cy);
        if (mouseDown) {
            setObstacle(s.x, s.y, false);
        } else {
            // Hover (no button held): treat as light drag so the fluid flows
            curObsR = MIN_RADIUS * 0.8;
            setObstacle(s.x, s.y, false);
        }
    }

    // Idle timer: stop stamping velocity 100ms after last move
    function resetIdle() {
        clearTimeout(idleTimer);
        idleTimer = window.setTimeout(() => { obsVelX = 0; obsVelY = 0; }, 100);
    }

    // -- Desktop events --
    // mousedown inside wrapper: begin drag
    listen(wrapper, 'mousedown', ((e: MouseEvent) => startDrag(e.clientX, e.clientY)) as EventListener);
    // mouseup anywhere: end drag (handles release outside wrapper)
    listen(window, 'mouseup', (endDrag) as EventListener);
    // mousemove ONLY on wrapper: obstacle strictly follows cursor within the zone
    listen(wrapper, 'mousemove', ((e: MouseEvent) => { drag(e.clientX, e.clientY); resetIdle(); }) as EventListener);
    // mouseenter: (re)start when cursor enters the wrapper
    listen(wrapper, 'mouseenter', ((e: MouseEvent) => startDrag(e.clientX, e.clientY)) as EventListener);
    // mouseleave: INSTANTLY zero the obstacle so zero ghost circle is left outside
    listen(wrapper, 'mouseleave', ((_e: MouseEvent) => killObstacle()) as EventListener);
    // window blur: lost focus
    listen(window, 'blur', (endDrag) as EventListener);

    // -- Touch events (on wrapper so they don't fire outside) --
    listen(wrapper, 'touchstart', ((e: TouchEvent) => { e.preventDefault(); startDrag(e.touches[0].clientX, e.touches[0].clientY); }) as EventListener);
    listen(wrapper, 'touchmove', ((e: TouchEvent) => { e.preventDefault(); drag(e.touches[0].clientX, e.touches[0].clientY); }) as EventListener);
    listen(wrapper, 'touchend', ((e: TouchEvent) => { e.preventDefault(); endDrag(); }) as EventListener);

    // ── Click-to-spawn: teleport N/20 particles to the click point ──
    // Attach to WRAPPER (not canvas) so we get the event in the same zone as drag.
    // Use e.buttons===0 to confirm no button is held (pure click, not end-of-drag).
    listen(wrapper, 'click', ((e: MouseEvent) => {
        // Only fire on a real single click with no button held at time of event
        if (e.buttons !== 0) return;
        const s = toSim(e.clientX, e.clientY);
        const spawnCount = Math.max(1, Math.round(N / 20));

        // Build a pool of candidates and shuffle to pick random indices
        const indices = Array.from({ length: N }, (_, i) => i);
        for (let i = N - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        for (let k = 0; k < spawnCount; k++) {
            const i = indices[k];
            // Precise click origin (no jitter) as requested
            px[i] = Math.max(r, Math.min(simW - r, s.x));
            py[i] = Math.max(r, Math.min(simH - r, s.y));
            // Strong outward burst so spawned particles fan out visibly
            const angle = Math.random() * Math.PI * 2;
            const speed = 200 + Math.random() * 600; // fast enough to see clearly
            vx[i] = Math.cos(angle) * speed;
            vy[i] = Math.sin(angle) * speed;
            // Randomise shape on spawn (0-4)
            shapeType[i] = Math.floor(Math.random() * 5);
        }
    }) as EventListener);


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
        // Upload shape types
        for (let i = 0; i < N; i++) shapeD[i] = shapeType[i];
        gl!.bindBuffer(gl!.ARRAY_BUFFER, sBuf);
        gl!.bufferData(gl!.ARRAY_BUFFER, shapeD, gl!.DYNAMIC_DRAW);

        // ptSz: r*2 in sim-space mapped to canvas px, then subtract 6px for size reduction
        const ptSz = Math.max(2, r * 2 * (cvs.width / simW) - 6);
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
