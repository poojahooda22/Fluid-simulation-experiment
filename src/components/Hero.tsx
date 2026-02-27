import FluidCanvas from './FluidCanvas';

const Hero = () => {
    return (
        <section className="relative w-full h-full bg-black">
            {/* Fluid simulation background */}
            <FluidCanvas />  
        </section>
    );
};

export default Hero;
