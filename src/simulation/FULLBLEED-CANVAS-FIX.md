# Full-Bleed Canvas Viewport Fix

## Problem

The simulation canvas had a recurring regression on mobile browsers (iOS Safari, Android Chrome): fixing a gap at the top would create a gap at the bottom, and vice versa. The canvas edges were not perfectly attached to the viewport edges on all devices.

### Root Cause

The canvas relied on CSS percentage inheritance (`w-full h-full` / `width: 100%; height: 100%`) to determine its size, then read `clientWidth`/`clientHeight` each frame to set the WebGL buffer. On mobile browsers, the dynamic browser chrome (address bar that shows/hides) changes the actual visible viewport height, but CSS `100%` and `100vh` do not always reflect this accurately. This caused:

1. Canvas dimensions to be wrong (too tall or too short)
2. View bounds (`viewBottom`) to shift, misaligning the simulation camera with the physics walls
3. A "seesaw" effect: adjusting for one edge broke the other

## Solution

Make JavaScript the sole authority for canvas dimensions using the `visualViewport` API, removing CSS percentage sizing from the canvas entirely.

## Changes

### 1. `index.html`

Added `viewport-fit=cover` to the viewport meta tag. This tells notched devices (iPhone X+) to extend web content into the safe-area insets instead of leaving automatic padding.

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
```

### 2. `src/index.css`

Replaced `height: 100%` with `100dvh` (dynamic viewport height) as a progressive enhancement. `dvh` tracks the actual visible viewport on mobile, unlike `vh` which may include the area behind browser chrome. The `100vh` line is a fallback for older browsers.

```css
html, body, #root {
  width: 100%;
  height: 100vh;    /* fallback */
  height: 100dvh;   /* dynamic viewport height for mobile */
  margin: 0;
  padding: 0;
  overflow: hidden;
}
```

### 3. `src/components/FluidCanvas.tsx`

Removed `w-full h-full` from the canvas element's className. The canvas no longer gets its size from CSS — JS sets `canvas.style.width` and `canvas.style.height` explicitly.

```tsx
<canvas ref={canvasRef} className="absolute top-0 left-0 block bg-[#1a2ffb]" />
```

### 4. `src/simulation/createFluidPool.ts`

Four sub-changes:

#### `getViewportSize()` helper

Returns the most accurate viewport dimensions available. On mobile, `window.visualViewport` accounts for dynamic browser chrome (address bar show/hide). Falls back to `window.innerWidth/innerHeight` on desktop or older browsers.

```ts
function getViewportSize() {
    const vv = window.visualViewport;
    return { w: vv?.width ?? window.innerWidth, h: vv?.height ?? window.innerHeight };
}
```

#### Initial grid sizing

The simulation grid dimensions (`cScale`, `simWidth`, `res`) are now computed from `getViewportSize()` instead of `canvas.clientWidth/clientHeight`, which could be 0 or wrong before the first layout pass.

#### `resizeCanvas()` function

Imperatively sets canvas CSS size and buffer dimensions from the viewport, then recalculates simulation view bounds. Early-returns when size is unchanged (zero cost per frame).

Called:
- Once at init (before first frame)
- Per-frame as a safety net
- On viewport resize events

#### Resize event listeners

Listens to all viewport change signals, rAF-throttled to avoid intermediate values during transitions:

- `window.resize` — desktop window resizing
- `window.orientationchange` — mobile rotation
- `window.visualViewport.resize` — mobile address bar resize
- `window.visualViewport.scroll` — iOS Safari fires this (not resize) when the address bar hides/shows

All listeners auto-clean via the existing `listen()`/`rms[]` mechanism.

## Key Principle

**Before:** CSS cascade (`100%` height through 6 levels of DOM) determined canvas size, JS read it back via `clientWidth/clientHeight`.

**After:** JS reads the viewport size directly from `visualViewport` API, sets canvas dimensions imperatively. CSS is only a fallback for the initial layout before JS runs.

## Browser Compatibility

| Feature | Support |
|---------|---------|
| `visualViewport` API | Safari 13+, Chrome 61+, Firefox 91+ |
| `100dvh` CSS unit | Safari 15.4+, Chrome 108+, Firefox 101+ |
| `viewport-fit=cover` | Safari 11+, Chrome 64+ |

All features have graceful fallbacks — older browsers get the previous behavior (`innerWidth/innerHeight`, `100vh`).