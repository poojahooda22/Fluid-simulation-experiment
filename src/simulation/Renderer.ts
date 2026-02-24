/**
 * WebGL2 renderer — draws particles as point sprites with SDF circles.
 * No obstacle circle rendering (removed per user request).
 */

// ─── GLSL (particles) ───────────────────────────────────────────
const VERT = `#version 300 es
precision highp float;

in vec2 a_pos;
in vec3 a_color;

uniform vec2  u_res;
uniform float u_pointSize;

out vec3 v_color;

void main() {
  vec2 clip = vec2(
    a_pos.x / u_res.x *  2.0 - 1.0,
    1.0 - a_pos.y / u_res.y * 2.0
  );
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = u_pointSize;
  v_color = a_color;
}
`;

const FRAG = `#version 300 es
precision mediump float;

in vec3 v_color;
out vec4 outColor;

void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  float alpha = 1.0 - smoothstep(0.85, 1.0, d);
  outColor = vec4(v_color, alpha * 0.85);
}
`;

// ─── Helpers ────────────────────────────────────────────────────
function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s);
        gl.deleteShader(s);
        throw new Error(`Shader compile error: ${log}`);
    }
    return s;
}

// ─── Renderer class ─────────────────────────────────────────────
export class Renderer {
    private gl: WebGL2RenderingContext;
    private program: WebGLProgram;
    private posBuf: WebGLBuffer;
    private colorBuf: WebGLBuffer;
    private vao: WebGLVertexArrayObject;
    private uRes: WebGLUniformLocation;
    private uPointSize: WebGLUniformLocation;

    constructor(canvas: HTMLCanvasElement) {
        const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: false })!;
        if (!gl) throw new Error('WebGL2 not available');
        this.gl = gl;

        // compile + link
        const vs = compileShader(gl, gl.VERTEX_SHADER, VERT);
        const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG);
        const prog = gl.createProgram()!;
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error(`Program link error: ${gl.getProgramInfoLog(prog)}`);
        }
        this.program = prog;

        this.uRes = gl.getUniformLocation(prog, 'u_res')!;
        this.uPointSize = gl.getUniformLocation(prog, 'u_pointSize')!;

        // VAO
        this.vao = gl.createVertexArray()!;
        gl.bindVertexArray(this.vao);

        // position buffer
        this.posBuf = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
        const aPos = gl.getAttribLocation(prog, 'a_pos');
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

        // color buffer
        this.colorBuf = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
        const aColor = gl.getAttribLocation(prog, 'a_color');
        gl.enableVertexAttribArray(aColor);
        gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, 0, 0);

        gl.bindVertexArray(null);
    }

    render(
        posX: Float32Array,
        posY: Float32Array,
        colors: Float32Array,
        count: number,
        simWidth: number,
        simHeight: number,
        particleRadius: number,
    ) {
        const gl = this.gl;
        const cvs = gl.canvas as HTMLCanvasElement;

        if (cvs.width !== cvs.clientWidth || cvs.height !== cvs.clientHeight) {
            cvs.width = cvs.clientWidth;
            cvs.height = cvs.clientHeight;
        }
        gl.viewport(0, 0, cvs.width, cvs.height);

        gl.clearColor(0, 0, 0, 0);  // transparent clear (parent bg shows through)
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        // interleave positions
        const posData = new Float32Array(count * 2);
        for (let i = 0; i < count; i++) {
            posData[i * 2] = posX[i];
            posData[i * 2 + 1] = posY[i];
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, posData, gl.DYNAMIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, colors.subarray(0, count * 3), gl.DYNAMIC_DRAW);

        const pixelsPerSimUnit = cvs.width / simWidth;
        const pointSize = particleRadius * 2.0 * pixelsPerSimUnit;

        gl.useProgram(this.program);
        gl.uniform2f(this.uRes, simWidth, simHeight);
        gl.uniform1f(this.uPointSize, pointSize);

        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.POINTS, 0, count);
        gl.bindVertexArray(null);
    }

    dispose() {
        const gl = this.gl;
        gl.deleteProgram(this.program);
        gl.deleteBuffer(this.posBuf);
        gl.deleteBuffer(this.colorBuf);
        gl.deleteVertexArray(this.vao);
    }
}
