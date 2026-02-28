import AnimatedHeading from './AnimatedHeading';
import FluidCanvas from './FluidCanvas';
import Navbar from './Navbar';

const Hero = () => {
    return (
        <section className="relative w-full h-full">
            {/* Fluid simulation background */}
            <FluidCanvas />

            {/* UI overlay — above the simulation interaction layer */}
            <div className="absolute top-10 left-10 right-10 inset-0 z-20 pointer-events-none">
                <Navbar />
                <div className="flex flex-col justify-center items-center gap-6 w-full h-full -translate-y-[13vh]">
                    <p className="text-[1.3rem] font-light">IS YOUR BIG IDEA READY TO GO WILD?</p>
                    <AnimatedHeading />
                </div>
            </div>
        </section>
    );
};

export default Hero;
