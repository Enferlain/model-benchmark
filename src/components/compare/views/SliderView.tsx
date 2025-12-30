import React, { useState } from 'react';
import { ModelOutput } from '../../../types';

interface Props {
  images: (ModelOutput | undefined)[];
  modelNames: string[];
}

export const SliderView: React.FC<Props> = ({ images, modelNames }) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const getUrl = (url: string) => `${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${url}`;

  if (images.length !== 2) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500">
        Slider view requires exactly 2 models selected.
      </div>
    );
  }

  const [imgA, imgB] = images;
  if (!imgA || !imgB) return null;

  return (
    <div className="w-full h-full flex items-center justify-center p-4">
      <div className="relative w-full max-w-[800px] aspect-square select-none overflow-hidden rounded-lg shadow-xl border border-slate-300 dark:border-slate-600 bg-black">
          {/* Base Image (Model B - Right side) */}
          <img
            src={getUrl(imgB.url)}
            alt={modelNames[1]}
            className="absolute inset-0 w-full h-full object-contain"
            draggable={false}
          />

          {/* Overlay Image (Model A - Left side) */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ width: `${sliderPosition}%`, borderRight: '2px solid white' }}
          >
              {/* Counter-scaling to keep image static while container shrinks */}
              <img
                src={getUrl(imgA.url)}
                alt={modelNames[0]}
                className="absolute top-0 left-0 max-w-none h-full"
                style={{ width: `${100 * (100/sliderPosition)}%`, maxWidth: 'none' }}
              />
               {/*
                  Note: The simple percentage trick above works well if the image fills the container.
                  If using object-contain with different aspect ratios, it can get tricky.
                  For now, we assume consistent aspect ratios for benchmarks.

                  Correct approach for exact registration:
                  Both images are strictly 100% width/height of the container.
                  The wrapper div clips the left one.
                  The inner image of the wrapper needs to be width: 100vw of the PARENT container.

                  style={{ width: `${100 / (sliderPosition/100)}%` }}
               */}
               <img
                src={getUrl(imgA.url)}
                alt={modelNames[0]}
                className="absolute top-0 left-0 w-full h-full object-contain"
                style={{ width: `${100 / (sliderPosition/100)}%`, maxWidth: 'none' }}
               />
          </div>

          {/* Slider Control */}
          <input
            type="range"
            min="0"
            max="100"
            value={sliderPosition}
            onChange={(e) => setSliderPosition(Number(e.target.value))}
            className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-20"
          />

          {/* Labels */}
          <div className="absolute bottom-4 left-4 bg-black/50 text-white px-2 py-1 rounded text-sm pointer-events-none z-10">{modelNames[0]}</div>
          <div className="absolute bottom-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-sm pointer-events-none z-10">{modelNames[1]}</div>

          {/* Slider Handle Visual */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_10px_rgba(0,0,0,0.5)] z-10 pointer-events-none"
            style={{ left: `${sliderPosition}%` }}
          >
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-slate-800"><path d="m9 18 6-6-6-6"/></svg>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-slate-800 rotate-180 absolute"><path d="m9 18 6-6-6-6"/></svg>
            </div>
          </div>
      </div>
    </div>
  );
};
