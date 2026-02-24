import { useEffect, useRef, useState } from 'react';
import { createFluidPool, type PoolAPI } from '../simulation/createFluidPool';

export default function FluidCanvas() {
    const wrapperRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const apiRef = useRef<PoolAPI | null>(null);

    const [paused, setPaused] = useState(false);
    const [flipRatio, setFlipRatio] = useState(0.75);

    // one-off init + full cleanup
    useEffect(() => {
        const w = wrapperRef.current, c = canvasRef.current;
        if (!w || !c) return;
        const api = createFluidPool(w, c);
        apiRef.current = api;
        return () => { api.cleanup(); apiRef.current = null; };
    }, []);

    // sync controls without re-init
    useEffect(() => { apiRef.current?.setFlipRatio(flipRatio); }, [flipRatio]);
    useEffect(() => { apiRef.current?.setPaused(paused); }, [paused]);

    return (
        <div className="relative w-full h-full">
            {/* 2× interaction wrapper — extends beyond canvas for pointer capture */}
            <div
                ref={wrapperRef}
                className="absolute z-10"
                style={{ width: '200%', height: '200%', top: '-50%', left: '-50%' }}
            />

            {/* simulation canvas */}
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

            {/* controls */}
            {/* <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-5
                bg-white/10 backdrop-blur-md rounded-xl px-6 py-3 z-20
                text-white text-sm select-none">
                <label className="flex items-center gap-2 cursor-pointer">
                    <span className="opacity-70">FLIP</span>
                    <input type="range" min="0" max="1" step="0.01" value={flipRatio}
                        onChange={e => setFlipRatio(parseFloat(e.target.value))}
                        className="w-28 accent-blue-400" />
                    <span className="w-8 text-right tabular-nums">{flipRatio.toFixed(2)}</span>
                </label>
                <button onClick={() => setPaused(p => !p)}
                    className="px-3 py-1 rounded-md bg-white/15 hover:bg-white/25 transition-colors">
                    {paused ? '▶ Play' : '⏸ Pause'}
                </button>
                <button onClick={() => apiRef.current?.reset()}
                    className="px-3 py-1 rounded-md bg-white/15 hover:bg-white/25 transition-colors">
                    ↻ Reset
                </button>
            </div> */}
        </div>
    );
}
