interface TiltButtonProps {
    supported: boolean;
    enabled: boolean;
    permissionDenied: boolean;
    onRequest: () => void;
}

export default function TiltButton({ supported, enabled, permissionDenied, onRequest }: TiltButtonProps) {
    const isTouchDevice = 'ontouchstart' in window
        || window.matchMedia('(pointer: coarse)').matches;

    if (!isTouchDevice || !supported || enabled) return null;

    return (
        <button
            onClick={onRequest}
            disabled={permissionDenied}
            className="pointer-events-auto backdrop-blur-sm text-black bg-white
                       px-[2vh] py-[2vh] rounded-full text-sm font-light border border-white/30
                       disabled:cursor-not-allowed"
        >
            {permissionDenied ? 'Tilt Permission Denied' : 'Enable Tilt Controls'}
        </button>
    );
}
