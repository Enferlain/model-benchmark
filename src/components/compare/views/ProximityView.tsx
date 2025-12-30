import React, { useState, useRef, useEffect } from 'react';
import { ModelOutput } from '../../../types';

interface Props {
  images: (ModelOutput | undefined)[];
  modelNames: string[];
}

export const ProximityView: React.FC<Props> = ({ images, modelNames }) => {
  const [activeIndex, setActiveIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const getUrl = (url: string) => `${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${url}`;

  // This view works by dividing the container width into N segments.
  // Hovering a segment shows that model's image.

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;

    // Calculate index based on X position
    const segmentWidth = width / images.length;
    const index = Math.min(Math.floor(x / segmentWidth), images.length - 1);

    if (index !== activeIndex && index >= 0) {
      setActiveIndex(index);
    }
  };

  const activeImage = images[activeIndex];

  return (
    <div className="w-full h-full flex flex-col items-center justify-center p-4">
       <div className="mb-4 text-center">
         <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            Interactive Lens
         </h3>
         <p className="text-sm text-slate-500">Move your mouse horizontally to switch models</p>
       </div>

       <div
         ref={containerRef}
         onMouseMove={handleMouseMove}
         onMouseLeave={() => setActiveIndex(0)} // Reset? Or keep last?
         className="relative w-full max-w-[800px] aspect-square bg-black rounded-lg overflow-hidden shadow-xl cursor-crosshair border border-slate-300 dark:border-slate-700"
       >
          {activeImage ? (
            <img
              src={getUrl(activeImage.url)}
              alt={modelNames[activeIndex]}
              className="w-full h-full object-contain pointer-events-none"
            />
          ) : (
             <div className="w-full h-full flex items-center justify-center text-slate-500">
               No image for {modelNames[activeIndex]}
             </div>
          )}

          {/* Overlay Label */}
          <div className="absolute top-4 left-4 bg-black/70 text-white px-4 py-2 rounded-full text-lg font-bold backdrop-blur-md pointer-events-none transition-all">
             {modelNames[activeIndex]}
          </div>

          {/* Visual Guides (optional) */}
          <div className="absolute inset-0 flex pointer-events-none opacity-20">
             {images.map((_, i) => (
               <div key={i} className="flex-1 border-r border-white/50 last:border-0" />
             ))}
          </div>
       </div>
    </div>
  );
};
