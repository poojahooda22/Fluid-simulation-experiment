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
            className="pointer-events-auto bg-white/20 backdrop-blur-sm text-white
                       px-12 py-2.5 rounded-full text-sm font-light border border-white/30
                       hover:bg-white/30 transition-colors disabled:opacity-50
                       disabled:cursor-not-allowed"
        >
            {permissionDenied ? 'Tilt Permission Denied' : 'Enable Tilt Controls'}
        </button>
    );
}
