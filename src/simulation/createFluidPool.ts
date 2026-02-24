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
    const r = opts?.particleRadius ?? 8;
    const N = opts?.numParticles ?? 4000;
    const grav = opts?.gravity ?? 100;
    let fRat = opts?.flipRatio ?? 0.75;
    const DT = opts?.dt ?? 1 / 120;
    const pIt = opts?.pressureIters ?? 30;
    const sIt = opts?.sepIters ?? 5;
    const omega = opts?.overRelax ?? 1.9;
    const obsR = opts?.obstacleRadius ?? 100;

    const SUB = 2;
    const DAMP = 0.97;
    const WFRIC = 0.7;
    const ODAMP = 0.3;
    const OBS_IMPULSE = 400;
    const RETF = 300;
    const RETD = 0.85;
    const MAXV = 600;
    const SLEEPV = 3;
    const SETTLE_DAMP = 0.90;

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
                col[idx * 3] = 0.1; col[idx * 3 + 1] = 0.3; col[idx * 3 + 2] = 0.8;
                idx++;
            }
            row++;
        }
        while (idx < N) {
            px[idx] = r + Math.random() * (simW - 2 * r);
            py[idx] = simH * 0.6 + Math.random() * simH * 0.35;
            vx[idx] = 0; vy[idx] = 0;
            col[idx * 3] = 0.1; col[idx * 3 + 1] = 0.3; col[idx * 3 + 2] = 0.8;
            idx++;
        }
        U.fill(0); V.fill(0); pU.fill(0); pV.fill(0); P.fill(0);
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

    // ── Pointer state ──
    let ptrX = 0, ptrY = 0, ptrOn = false;
    function s2s(cx: number, cy: number) {
        const rc = canvas.getBoundingClientRect();
        return { x: (cx - rc.left) / rc.width * simW, y: (cy - rc.top) / rc.height * simH };
    }

    // ── Step ──
    function step() {
        pU.set(U); pV.set(V);
        integrate(); bounds(); obstacle(); separate();
        classify(); p2g(); pressure(); g2p();
        clampV(); sleep(); colors();
    }

    function integrate() {
        for (let i = 0; i < N; i++) {
            vx[i] *= DAMP; vy[i] *= DAMP;
            vy[i] += grav * DT;
            if (py[i] < poolTopY) {
                // above pool: strong pull back down + damping
                vy[i] += RETF * DT;
                vx[i] *= RETD; vy[i] *= RETD;
            } else {
                // inside pool: settle particles to reduce jitter
                const spd = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
                if (spd < 40) { vx[i] *= SETTLE_DAMP; vy[i] *= SETTLE_DAMP; }
            }
            px[i] += vx[i] * DT; py[i] += vy[i] * DT;
        }
    }

    function bounds() {
        const mn = r, mxX = simW - r, mxY = simH - r;
        for (let i = 0; i < N; i++) {
            // left/right walls: kill horizontal vel, damp vertical hard
            if (px[i] < mn)  { px[i] = mn;  vx[i] = -vx[i] * 0.1; vy[i] *= (1 - WFRIC); }
            if (px[i] > mxX) { px[i] = mxX; vx[i] = -vx[i] * 0.1; vy[i] *= (1 - WFRIC); }
            // top wall: bounce weakly downward, kill most speed
            if (py[i] < mn)  { py[i] = mn;  vy[i] = Math.abs(vy[i]) * 0.15; vx[i] *= (1 - WFRIC); }
            // bottom wall: kill bounce
            if (py[i] > mxY) { py[i] = mxY; vy[i] = 0; vx[i] *= (1 - WFRIC); }
        }
    }

    function obstacle() {
        if (!ptrOn) return;
        const R = obsR + r;
        for (let i = 0; i < N; i++) {
            const dx = px[i] - ptrX, dy = py[i] - ptrY;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < R && d > 1e-6) {
                const nx = dx / d, ny2 = dy / d;
                px[i] = ptrX + nx * R; py[i] = ptrY + ny2 * R;
                const vn = vx[i] * nx + vy[i] * ny2;
                if (vn < 0) { vx[i] -= (1 + ODAMP) * vn * nx; vy[i] -= (1 + ODAMP) * vn * ny2; }
                // impulse: closer particles get pushed harder
                const strength = OBS_IMPULSE * (1 - d / R);
                vx[i] += nx * strength * DT;
                vy[i] += ny2 * strength * DT;
            }
        }
    }

    function separate() {
        const md = 2 * r, md2 = md * md;
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
                for (let dj = -1; dj <= 1; dj++) {
                    const nj = cj + dj; if (nj < 0 || nj >= hH) continue;
                    for (let di = -1; di <= 1; di++) {
                        const ni = ci + di; if (ni < 0 || ni >= hW) continue;
                        const c = ni + nj * hW, end = hSt[c] + hCnt[c];
                        for (let k = hSt[c]; k < end; k++) {
                            const q = hSo[k]; if (q <= p) continue;
                            const dx = px[q] - px[p], dy = py[q] - py[p], d2 = dx * dx + dy * dy;
                            if (d2 < md2 && d2 > 1e-8) {
                                const d = Math.sqrt(d2), ov = (md - d) * 0.5;
                                const nnx = dx / d, nny = dy / d;
                                px[p] -= ov * nnx; py[p] -= ov * nny;
                                px[q] += ov * nnx; py[q] += ov * nny;
                            }
                        }
                    }
                }
            }
        }
    }

    function classify() {
        CT.fill(AIR);
        for (let i = 0; i < nx; i++) { CT[iC(i, 0)] = SOLID; CT[iC(i, ny - 1)] = SOLID; }
        for (let j = 0; j < ny; j++) { CT[iC(0, j)] = SOLID; CT[iC(nx - 1, j)] = SOLID; }
        if (ptrOn) {
            const or2 = obsR * obsR;
            for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
                const dx = (i + .5) * h - ptrX, dy = (j + .5) * h - ptrY;
                if (dx * dx + dy * dy < or2) CT[iC(i, j)] = SOLID;
            }
        }
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
        for (let i = 0; i < N; i++) {
            const x = px[i], y = py[i];
            const picU = smpU(U, x, y), picV = smpV(V, x, y);
            const dU = picU - smpU(pU, x, y), dV = picV - smpV(pV, x, y);
            vx[i] = (1 - fRat) * picU + fRat * (vx[i] + dU);
            vy[i] = (1 - fRat) * picV + fRat * (vy[i] + dV);
        }
    }

    function clampV() {
        const m2 = MAXV * MAXV;
        for (let i = 0; i < N; i++) {
            const v2 = vx[i] * vx[i] + vy[i] * vy[i];
            if (v2 > m2) { const s = MAXV / Math.sqrt(v2); vx[i] *= s; vy[i] *= s; }
        }
    }

    function sleep() {
        const s2 = SLEEPV * SLEEPV;
        for (let i = 0; i < N; i++)
            if (vx[i] * vx[i] + vy[i] * vy[i] < s2 && py[i] > poolTopY) { vx[i] = 0; vy[i] = 0; }
    }

    function colors() {
        for (let i = 0; i < N; i++) {
            const t = Math.min(1, Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i]) / 150);
            col[i * 3] = 0.1 + t * 0.2; col[i * 3 + 1] = 0.3 + t * 0.5; col[i * 3 + 2] = 0.8 + t * 0.2;
        }
    }

    // ── Events ──
    listen(wrapper, 'pointermove', ((e: PointerEvent) => {
        const p = s2s(e.clientX, e.clientY); ptrX = p.x; ptrY = p.y; ptrOn = true;
    }) as EventListener);
    listen(wrapper, 'pointerenter', ((e: PointerEvent) => {
        const p = s2s(e.clientX, e.clientY); ptrX = p.x; ptrY = p.y; ptrOn = true;
    }) as EventListener);
    listen(wrapper, 'pointerleave', (() => { ptrOn = false; }) as EventListener);
    listen(wrapper, 'touchmove', ((e: TouchEvent) => {
        if (e.touches.length) { const t = e.touches[0], p = s2s(t.clientX, t.clientY); ptrX = p.x; ptrY = p.y; ptrOn = true; }
    }) as EventListener, true);
    listen(wrapper, 'touchend', (() => { ptrOn = false; }) as EventListener);
    listen(wrapper, 'touchcancel', (() => { ptrOn = false; }) as EventListener);

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
