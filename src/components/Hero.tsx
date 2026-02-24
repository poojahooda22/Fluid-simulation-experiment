import FluidCanvas from './FluidCanvas';

const Hero = () => {
    return (
        <section className="relative w-full h-screen bg-black overflow-hidden">
            {/* Fluid simulation background */}
            <FluidCanvas />

            {/* Centred title overlay */}
            <h1 className="absolute inset-0 flex items-center justify-center
                     text-white text-4xl md:text-6xl lg:text-8xl
                     font-light uppercase select-none
                     z-10 pointer-events-none">
            </h1>
        </section>
    );
};

export default Hero;
