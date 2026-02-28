import { useEffect, useRef } from 'react';
import gsap from 'gsap';

const LINE1 = "Let's work";
const LINE2 = "together!";
const FULL_TEXT = `${LINE1} ${LINE2}`;

function renderLine(
    line: string,
    startIndex: number,
    charRefs: React.RefObject<(HTMLSpanElement | null)[]>,
) {
    return line.split('').map((char, i) => {
        const idx = startIndex + i;
        if (char === ' ') {
            return (
                <span key={idx} className="inline-block" aria-hidden="true">
                    &nbsp;
                </span>
            );
        }
        return (
            <span
                key={idx}
                className="inline-block overflow-hidden h-[1.05em]"
                aria-hidden="true"
            >
                <span
                    ref={(el) => { charRefs.current[idx] = el; }}
                    className="inline-block will-change-transform"
                >
                    {char}
                </span>
            </span>
        );
    });
}

export default function AnimatedHeading() {
    const charRefs = useRef<(HTMLSpanElement | null)[]>([]);
    const containerRef = useRef<HTMLHeadingElement>(null);
    useEffect(() => {
        const allChars = LINE1 + ' ' + LINE2;

        const ctx = gsap.context(() => {
            // Collect indices of non-space characters that can be animated
            const animatableIndices: number[] = [];
            allChars.split('').forEach((char, i) => {
                if (char !== ' ') animatableIndices.push(i);
            });

            const runCycle = () => {
                // Pick 3 unique random indices via partial Fisher-Yates shuffle
                const shuffled = [...animatableIndices];
                for (let i = shuffled.length - 1; i > shuffled.length - 4 && i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
                }
                const picked = shuffled.slice(-3);

                const targets = picked
                    .map(i => charRefs.current[i])
                    .filter((el): el is HTMLSpanElement => el !== null);

                if (targets.length === 0) return;

                const tl = gsap.timeline({
                    onComplete: () => {
                        // 2–3 second pause between cycles
                        gsap.delayedCall(2.0 + Math.random() * 1.0, runCycle);
                    },
                });

                targets.forEach((el, i) => {
                    const staggerDelay = i * 0.12;
                    const exitDur = 0.3;
                    const enterDur = 0.3;

                    // 1) Slide UP out of clip (current position → above)
                    tl.to(el, {
                        yPercent: -100,
                        duration: exitDur,
                        ease: 'power2.in',
                    }, staggerDelay);

                    // 2) Teleport below while still clipped
                    tl.set(el, { yPercent: 100 }, staggerDelay + exitDur);

                    // 3) Slide UP into place (below → normal position)
                    tl.to(el, {
                        yPercent: 0,
                        duration: enterDur,
                        ease: 'power2.out',
                    }, staggerDelay + exitDur);
                });
            };

            // Start after a short initial delay
            gsap.delayedCall(3.0, runCycle);

        }, containerRef);

        return () => ctx.revert();
    }, []);

    // LINE2 starts after LINE1 + the space between them
    const line2Start = LINE1.length + 1;

    return (
        <h1
            ref={containerRef}
            aria-label={FULL_TEXT}
            className="heading-block pointer-events-auto text-white
            font-light text-[4.7rem] sm:text-[6rem] md:text-[8rem] lg:text-[12rem] leading-[2.8rem] sm:leading-[5.5rem] md:leading-[7rem] lg:leading-[10rem] text-center cursor-pointer "
        >
            <span className="heading-line">
                {renderLine(LINE1, 0, charRefs)}
            </span>
            <span className="heading-line text-[4.7rem] sm:text-[7rem] md:text-[9.5rem] lg:text-[14rem] leading-[3.2rem] sm:leading-[6.5rem] md:leading-[8.5rem] lg:leading-[12rem]">
                {renderLine(LINE2, line2Start, charRefs)}
            </span>
        </h1>
    );
}
