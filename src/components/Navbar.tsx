const Navbar = () => (
    <nav className="pointer-events-auto w-full flex justify-between items-center py-4 px-8 z-10">
        <span className="text-white text-[1rem] sm:text-[2rem] font-medium cursor-pointer">Fusion</span>
        <button className="bg-black text-white px-[10rem] py-4 sm:px-2 sm:py-2 text-[.9rem] 
        sm:w-[6.5vw] sm:h-[5vh] rounded-full sm:text-[1.2rem]
        font-light cursor-pointer">
            Let's talk
        </button>
    </nav>
);

export default Navbar;
