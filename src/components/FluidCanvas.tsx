import { useEffect, useRef, useState } from 'react';
import { createFluidPool, type PoolAPI } from '../simulation/createFluidPool';

interface FluidCanvasProps {
    onApiReady?: (api: PoolAPI) => void;
}

export default function FluidCanvas({ onApiReady }: FluidCanvasProps) {
    const wrapperRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const apiRef = useRef<PoolAPI | null>(null);

    const [paused] = useState(false);
    const [flipRatio] = useState(0.95);

    // one-off init + full cleanup
    useEffect(() => {
        const w = wrapperRef.current, c = canvasRef.current;
        if (!w || !c) return;
        const api = createFluidPool(w, c);
        apiRef.current = api;
        onApiReady?.(api);
        return () => { api.cleanup(); apiRef.current = null; };
    }, []);

    // sync controls without re-init
    useEffect(() => { apiRef.current?.setFlipRatio(flipRatio); }, [flipRatio]);
    useEffect(() => { apiRef.current?.setPaused(paused); }, [paused]);

    return (
        <div className="relative w-full h-full overflow-hidden">
            {/* 2× interaction wrapper — extends beyond canvas for pointer capture */}
            <div
                ref={wrapperRef}
                className="absolute z-10"
                style={{ width: '200%', height: '200%', top: '-50%', left: '-50%' }}
            />

            {/* simulation canvas */}
            <canvas ref={canvasRef} className="absolute top-0 left-0 block w-full h-full bg-[#1a2ffb]" />
        </div>
    );
}
