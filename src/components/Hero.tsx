import { useCallback, useRef } from 'react';
import AnimatedHeading from './AnimatedHeading';
import FluidCanvas from './FluidCanvas';
import Navbar from './Navbar';
import TiltButton from './TiltButton';
import { useTiltControl } from '../hooks/useTiltControl';
import type { PoolAPI } from '../simulation/createFluidPool';

const Hero = () => {
    const apiRef = useRef<PoolAPI | null>(null);
    const tilt = useTiltControl(apiRef);

    const handleApiReady = useCallback((api: PoolAPI) => {
        apiRef.current = api;
    }, []);

    return (
        <section className="relative w-full h-full">
            {/* Fluid simulation background */}
            <FluidCanvas onApiReady={handleApiReady} />

            {/* UI overlay — above the simulation interaction layer */}
            <div className="absolute top-10 left-10 right-10 inset-0 z-20 pointer-events-none">
                <Navbar />
                <div className="flex flex-col justify-center items-center gap-6 w-full h-full -translate-y-[13vh]">
                    <p className="text-[0.9rem] sm:text-[1.3rem] font-light">IS YOUR BIG IDEA READY TO GO WILD?</p>
                    <AnimatedHeading />
                    <TiltButton
                        supported={tilt.supported}
                        enabled={tilt.enabled}
                        permissionDenied={tilt.permissionDenied}
                        onRequest={tilt.requestEnable}
                    />
                </div>
            </div>
        </section>
    );
};

export default Hero;
