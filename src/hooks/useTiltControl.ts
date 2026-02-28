import { useCallback, useEffect, useRef, useState } from 'react';
import type { PoolAPI } from '../simulation/createFluidPool';

export interface TiltState {
    supported: boolean;
    enabled: boolean;
    permissionDenied: boolean;
}

// Tuning constants
const DEADZONE_DEG = 3;
const MAX_TILT_DEG = 30;
const SMOOTHING = 0.15;     // low-pass filter coefficient (higher = more smoothing)
const FORCE_SCALE = 5.0;    // max horizontal acceleration in sim units/s^2

export function useTiltControl(apiRef: React.RefObject<PoolAPI | null>) {
    const [state, setState] = useState<TiltState>({
        supported: false,
        enabled: false,
        permissionDenied: false,
    });

    const calibrationGamma = useRef(0);
    const smoothedGamma = useRef(0);
    const orientationRef = useRef<'portrait' | 'landscape'>('portrait');

    // Detect support on mount
    useEffect(() => {
        setState(s => ({ ...s, supported: 'DeviceOrientationEvent' in window }));
    }, []);

    // Track screen orientation changes for axis remapping
    useEffect(() => {
        const update = () => {
            const angle = screen.orientation?.angle ?? 0;
            orientationRef.current = (angle === 90 || angle === 270) ? 'landscape' : 'portrait';
        };
        update();
        screen.orientation?.addEventListener('change', update);
        window.addEventListener('orientationchange', update);
        return () => {
            screen.orientation?.removeEventListener('change', update);
            window.removeEventListener('orientationchange', update);
        };
    }, []);

    // Stable orientation event handler
    const handleOrientation = useCallback((e: DeviceOrientationEvent) => {
        const api = apiRef.current;
        if (!api) return;

        let gamma = (e.gamma ?? 0) - calibrationGamma.current;

        // In landscape, beta acts as the left-right axis
        if (orientationRef.current === 'landscape') {
            gamma = (e.beta ?? 0) - calibrationGamma.current;
        }

        // Low-pass filter
        smoothedGamma.current += (gamma - smoothedGamma.current) * (1 - SMOOTHING);
        let tilt = smoothedGamma.current;

        // Deadzone
        if (Math.abs(tilt) < DEADZONE_DEG) {
            tilt = 0;
        } else {
            tilt -= Math.sign(tilt) * DEADZONE_DEG;
        }

        // Clamp
        const effectiveMax = MAX_TILT_DEG - DEADZONE_DEG;
        tilt = Math.max(-effectiveMax, Math.min(effectiveMax, tilt));

        // Normalize to [-1, 1] and scale
        const forceX = (tilt / effectiveMax) * FORCE_SCALE;
        api.setTiltForce(forceX, 0);
    }, [apiRef]);

    const requestEnable = useCallback(async () => {
        // iOS 13+ requires explicit permission request from a user gesture
        const DOE = DeviceOrientationEvent as unknown as {
            requestPermission?: () => Promise<string>;
        };
        if (typeof DOE.requestPermission === 'function') {
            try {
                const permission = await DOE.requestPermission();
                if (permission !== 'granted') {
                    setState(s => ({ ...s, permissionDenied: true }));
                    return;
                }
            } catch {
                setState(s => ({ ...s, permissionDenied: true }));
                return;
            }
        }

        // Calibrate with the first reading, then start the real listener
        const calibrate = (e: DeviceOrientationEvent) => {
            calibrationGamma.current = orientationRef.current === 'landscape'
                ? (e.beta ?? 0)
                : (e.gamma ?? 0);
            smoothedGamma.current = 0;
            window.removeEventListener('deviceorientation', calibrate);
            window.addEventListener('deviceorientation', handleOrientation);
            setState(s => ({ ...s, enabled: true }));
        };

        window.addEventListener('deviceorientation', calibrate);
    }, [handleOrientation]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            window.removeEventListener('deviceorientation', handleOrientation);
            apiRef.current?.setTiltForce(0, 0);
        };
    }, [handleOrientation, apiRef]);

    return { ...state, requestEnable };
}
