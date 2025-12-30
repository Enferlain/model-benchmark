import React, { useState } from 'react';
import { ModelOutput } from '../../../types';
import { getImageUrl } from '../utils';

interface Props {
  images: (ModelOutput | undefined)[];
  modelNames: string[];
}

export const SliderView: React.FC<Props> = ({ images, modelNames }) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50);

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
            src={getImageUrl(imgB.url)}
            alt={modelNames[1]}
            className="absolute inset-0 w-full h-full object-contain"
            draggable={false}
          />

          {/* Overlay Image (Model A - Left side) */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ width: `${sliderPosition}%`, borderRight: '2px solid white' }}
          >
               <img
                src={getImageUrl(imgA.url)}
                alt={modelNames[0]}
                className="absolute top-0 left-0 w-full h-full object-contain"
                style={{ width: `${100 / (Math.max(sliderPosition, 1)/100)}%`, maxWidth: 'none' }}
               />
          </div>

          {/* Slider Control */}
          <input
            type="range"
            min="1"
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
