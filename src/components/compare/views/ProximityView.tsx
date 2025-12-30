import React, { useState, useRef, useMemo } from 'react';
import { ModelOutput } from '../../../types';
import { getImageUrl } from '../utils';

interface Props {
  images: (ModelOutput | undefined)[];
  modelNames: string[];
}

interface ScatteredImage extends ModelOutput {
  modelName: string;
  scatterX: number;
  scatterY: number;
  rotation: number;
  originalIndex: number;
}

export const ProximityView: React.FC<Props> = ({ images, modelNames }) => {
  const [radius, setRadius] = useState(450);
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 });
  const workspaceRef = useRef<HTMLDivElement>(null);

  // Generate stable scatter coordinates for the images
  const scatteredImages = useMemo(() => {
    return images.map((img, idx) => {
      if (!img) return null;
      return {
        ...img,
        modelName: modelNames[idx],
        originalIndex: idx,
        // Percentage-based coordinates for better responsiveness
        scatterX: 5 + Math.random() * 80,
        scatterY: 10 + Math.random() * 70,
        rotation: (Math.random() - 0.5) * 12
      };
    }).filter((item): item is ScatteredImage => item !== null);
  }, [images, modelNames]);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!workspaceRef.current) return;
    const rect = workspaceRef.current.getBoundingClientRect();
    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleMouseLeave = () => {
    setMousePos({ x: -1000, y: -1000 });
  };

  return (
    <div className="flex flex-col h-full w-full bg-zinc-100 dark:bg-zinc-900/20 rounded-xl overflow-hidden">
      {/* Main Workspace */}
      <div
        ref={workspaceRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="relative flex-grow min-h-[600px] overflow-hidden shadow-inner cursor-crosshair"
      >
        {scatteredImages.map((out) => {
          const w = workspaceRef.current?.clientWidth || 1000;
          const h = workspaceRef.current?.clientHeight || 600;

          const centerX = (out.scatterX / 100) * w;
          const centerY = (out.scatterY / 100) * h;

          const dx = mousePos.x - centerX;
          const dy = mousePos.y - centerY;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // IMPROVED MATH: S-Curve influence
          let influence = Math.max(0, 1 - (distance / radius));
          influence = Math.pow(influence, 3); // Steeper curve for more "snap"

          const scale = 1 + (influence * 1.6); // Massive scale-up
          const zIndex = Math.floor(out.originalIndex + (influence * 2000));

          // Move slightly toward cursor to "look at you"
          const moveX = dx * influence * 0.12;
          const moveY = dy * influence * 0.12;

          return (
            <div
              key={out.originalIndex}
              className="absolute w-44 aspect-square bg-white dark:bg-zinc-800 rounded-2xl overflow-hidden border-2 border-zinc-200 dark:border-zinc-700 transition-transform duration-150 ease-out pointer-events-none"
              style={{
                left: `${out.scatterX}%`,
                top: `${out.scatterY}%`,
                zIndex: zIndex,
                // Solid opacity, varying shadow depth instead
                boxShadow: influence > 0.1
                  ? `0 ${20 * influence}px ${50 * influence}px -10px rgba(0,0,0,0.5), 0 0 ${15 * influence}px rgba(99,102,241, 0.4)`
                  : '0 4px 12px rgba(0,0,0,0.1)',
                transform: `translate(${moveX}px, ${moveY}px) scale(${scale}) rotate(${out.rotation * (1 - influence)}deg)`,
                borderColor: influence > 0.2 ? '#6366f1' : '',
                borderWidth: influence > 0.2 ? '3px' : '2px'
              }}
            >
              <img
                src={getImageUrl(out.url)}
                className="w-full h-full object-cover"
                alt={out.modelName}
              />
              <div className={`absolute bottom-0 w-full bg-zinc-900/90 p-2 text-[10px] text-white font-bold transition-opacity ${influence > 0.3 ? 'opacity-100' : 'opacity-0'}`}>
                <div className="truncate">{out.modelName}</div>
                <div className="text-zinc-500 text-[8px] uppercase tracking-widest">Seed: {out.seed}</div>
              </div>
            </div>
          );
        })}

        {scatteredImages.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-400 gap-3">
             <div className="w-12 h-12 border-4 border-zinc-200 dark:border-zinc-800 border-t-indigo-500 rounded-full animate-spin"></div>
             <p className="text-sm font-bold uppercase tracking-widest">Gathering Latents...</p>
          </div>
        )}
      </div>

      {/* Footer Controls */}
      <footer className="mt-auto flex flex-wrap items-center gap-8 p-5 bg-white dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800">
        <div className="flex flex-col min-w-[200px]">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-black text-indigo-500 uppercase tracking-widest">Magnet Strength</span>
            <span className="text-[10px] font-mono text-zinc-400">{radius}px</span>
          </div>
          <input
            type="range" min="200" max="1200" step="10" value={radius}
            onChange={(e) => setRadius(Number(e.target.value))}
            className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-full appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
          />
        </div>

        <div className="hidden lg:block h-10 w-[1px] bg-zinc-200 dark:bg-zinc-800"></div>

        <div className="flex flex-grow justify-end gap-4">
             <div className="text-right">
                <p className="text-[10px] font-black text-zinc-400 uppercase leading-none">Status</p>
                <p className="text-xs font-black text-green-500 leading-none mt-1">Ready</p>
            </div>
            <div className="bg-zinc-100 dark:bg-zinc-800 h-8 w-8 rounded-full flex items-center justify-center font-black text-[10px] text-indigo-500">
                {scatteredImages.length}
            </div>
        </div>
      </footer>
    </div>
  );
};
