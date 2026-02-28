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
        <section className="fixed inset-0 w-full h-full">
            {/* Fluid simulation background */}
            <FluidCanvas onApiReady={handleApiReady} />

            {/* UI overlay — above the simulation interaction layer */}
            <div className="absolute top-4 left-4 right-4 sm:top-10 sm:left-10 sm:right-10 inset-0 z-20 pointer-events-none">
                <Navbar />
                <div className="flex flex-col justify-center items-center gap-6 w-full h-full -translate-y-[13vh]">
                    <p className="text-[0.75rem] sm:text-[1rem] font-light">IS YOUR BIG IDEA READY TO GO WILD?</p>
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
