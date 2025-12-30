import React, { useState, useEffect, useRef } from 'react';
import { ModelOutput } from '../../../types';
import { getImageUrl } from '../utils';

interface Props {
  images: (ModelOutput | undefined)[];
  modelNames: string[];
}

interface ScatterItem extends ModelOutput {
  modelName: string;
  scatterX: number;
  scatterY: number;
  rotation: number;
  localId: string;
}

export const ProximityView: React.FC<Props> = ({ images, modelNames }) => {
  const [items, setItems] = useState<ScatterItem[]>([]);
  const [radius, setRadius] = useState(450);
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 });
  const workspaceRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // When images prop changes, re-scatter them or update them.
    // Since images length = modelNames length, we map them.
    const newItems: ScatterItem[] = [];
    images.forEach((img, idx) => {
      if (!img) return;
      newItems.push({
        ...img,
        modelName: modelNames[idx],
        localId: `${idx}-${img.seed}`, // Unique ID for this view
        scatterX: 5 + Math.random() * 80, // 5% to 85% width
        scatterY: 10 + Math.random() * 70, // 10% to 80% height
        rotation: (Math.random() - 0.5) * 12,
      });
    });
    setItems(newItems);
  }, [images, modelNames]);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!workspaceRef.current) return;
    const rect = workspaceRef.current.getBoundingClientRect();
    setMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseLeave = () => {
    setMousePos({ x: -1000, y: -1000 });
  };

  return (
    <div className="w-full h-full flex flex-col relative font-sans text-zinc-900 dark:text-zinc-100 min-h-[600px]">
      {/* Header / Instructions */}
      <div className="flex justify-between items-center mb-4 px-2">
         <div className="flex flex-col">
             <h3 className="text-xl font-black tracking-tight uppercase leading-none">
                <span className="text-indigo-600">Magnetic</span> Cluster
             </h3>
             <p className="text-[11px] text-zinc-500 font-bold uppercase tracking-widest mt-1">Multi-Model Proximity Compare</p>
         </div>
         {/* Radius Control embedded here */}
         <div className="flex flex-col w-[200px]">
              <div className="flex justify-between items-center mb-1">
                 <span className="text-[10px] font-black text-indigo-500 uppercase tracking-widest">Magnet Strength</span>
                 <span className="text-[10px] font-mono text-zinc-400">{radius}px</span>
              </div>
              <input
                type="range" min="200" max="1200" step="10" value={radius}
                onChange={(e) => setRadius(Number(e.target.value))}
                className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
              />
         </div>
      </div>

      {/* Main Workspace */}
      <div
        ref={workspaceRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="relative flex-grow rounded-3xl border-2 border-dashed border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900/20 overflow-hidden shadow-inner transition-all"
      >
        {items.map((out, idx) => {
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
            const zIndex = Math.floor(idx + (influence * 2000));

            // Move slightly toward cursor to "look at you"
            const moveX = dx * influence * 0.12;
            const moveY = dy * influence * 0.12;

            return (
              <div
                key={out.localId}
                className="absolute w-44 aspect-square bg-white dark:bg-zinc-800 rounded-2xl overflow-hidden border-2 border-zinc-200 dark:border-zinc-700 transition-transform duration-150 ease-out pointer-events-none"
                style={{
                  left: `${out.scatterX}%`,
                  top: `${out.scatterY}%`,
                  zIndex: zIndex,
                  // Solid opacity, varying shadow depth instead
                  boxShadow: influence > 0.1
                    ? `0 ${20 * influence}px ${50 * influence}px -10px rgba(0,0,0,0.5), 0 0 ${15 * influence}px rgba(99,102,241, 0.4)`
                    : '0 4px 12px rgba(0,0,0,0.1)',
                  transform: `translate(${moveX}px, ${moveY}px) scale(${scale}) rotate(${out.rotation * (1-influence)}deg)`,
                  borderColor: influence > 0.2 ? '#6366f1' : '',
                  borderWidth: influence > 0.2 ? '3px' : '2px'
                }}
              >
                <img src={getImageUrl(out.url)} className="w-full h-full object-cover" alt={out.modelName} />
                <div className={`absolute bottom-0 w-full bg-zinc-900/90 p-2 text-[10px] text-white font-bold transition-opacity ${influence > 0.3 ? 'opacity-100' : 'opacity-0'}`}>
                    <div className="truncate">{out.modelName}</div>
                    <div className="text-zinc-500 text-[8px] uppercase tracking-widest">Seed: {out.seed}</div>
                </div>
              </div>
            );
        })}

        {items.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-400 gap-3">
             <p className="text-sm font-bold uppercase tracking-widest">No images to display</p>
          </div>
        )}
      </div>

      <div className="mt-2 text-right">
            <div className="inline-flex items-center gap-2 bg-zinc-100 dark:bg-zinc-800 px-3 py-1 rounded-full">
                <span className="text-[10px] font-black text-zinc-400 uppercase leading-none">Items</span>
                <span className="font-black text-xs text-indigo-500">{items.length}</span>
            </div>
      </div>
    </div>
  );
};
