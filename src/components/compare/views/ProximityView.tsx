import React, { useState, useRef, useMemo, useEffect } from 'react';
import { ModelOutput } from '../../../types';
import { getImageUrl } from '../utils';

interface GroupedImages {
  modelName: string;
  images: ModelOutput[];
}

interface Props {
  groups: GroupedImages[];
}

interface LayoutItem extends ModelOutput {
  modelName: string;
  x: number;
  y: number;
  rotation: number;
  scale: number;
  opacity: number;
  zIndex: number;
  isHovered: boolean;
  isRelated: boolean;
  groupId: number;
}

export const ProximityView: React.FC<Props> = ({ groups }) => {
  const [workspaceSize, setWorkspaceSize] = useState({ width: 0, height: 0 });
  const [hoveredSeed, setHoveredSeed] = useState<number | null>(null);
  const workspaceRef = useRef<HTMLDivElement>(null);

  // Resize Observer
  useEffect(() => {
    if (!workspaceRef.current) return;
    const updateSize = () => {
      if (workspaceRef.current) {
        setWorkspaceSize({
            width: workspaceRef.current.clientWidth,
            height: workspaceRef.current.clientHeight
        });
      }
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(workspaceRef.current);
    return () => observer.disconnect();
  }, []);

  // Compute Layout
  const layoutItems = useMemo(() => {
     if (workspaceSize.width === 0 || workspaceSize.height === 0) return [];
     if (groups.length === 0) return [];

     const items: LayoutItem[] = [];
     const groupCount = groups.length;
     
     // Determine Group Centers
     // If 1 group: Center
     // If 2 groups: Left, Right
     // If 3+ groups: Circle
     const centers = groups.map((_, i) => {
         if (groupCount === 1) return { x: 0.5, y: 0.5 };
         if (groupCount === 2) return { x: 0.25 + (i * 0.5), y: 0.5 };
         
         const angle = (i / groupCount) * Math.PI * 2 - Math.PI / 2;
         const radius = 0.25; // % of workspace
         return {
             x: 0.5 + Math.cos(angle) * radius,
             y: 0.5 + Math.sin(angle) * (radius * (workspaceSize.width / workspaceSize.height)) // Adjust for aspect ratio
         };
     });

     groups.forEach((group, groupIdx) => {
         const center = centers[groupIdx];
         
         group.images.forEach((img) => {
             // Position based on Seed
             // Consistent hashing for angle
             const seedAngle = (img.seed * 137.508) * (Math.PI / 180); // Golden angle distribution
             // Radius based on sort order or just seed value variation?
             // Let's use a spiral approach based on seed value to keep "Seed X" consistently placed
             // But we probably want all seeds visible. 
             // Let's put them in a local cloud around the center.
             
             // Deterministic Local Position
             // Normalize seed to 0-1 for layout "slots" or just use random-ish based on seed
             // Using seed directly ensures same seed = same relative pos
             
             // Pseudo-random from seed
             const rnd = (s: number) => {
                 const x = Math.sin(s * 12.9898) * 43758.5453;
                 return x - Math.floor(x);
             };
             
             const localRadius = 80 + (rnd(img.seed) * 120); // 80px to 200px spread
             const localAngle = rnd(img.seed + 1) * Math.PI * 2;

             // Absolute pixel coords
             const centerX = center.x * workspaceSize.width;
             const centerY = center.y * workspaceSize.height;

             // Collision-ish: push apart?
             // For now, simple static.

             const x = centerX + Math.cos(localAngle) * localRadius;
             const y = centerY + Math.sin(localAngle) * localRadius;
             
             const isHovered = hoveredSeed === img.seed && hoveredSeed !== null; // Specific logic? No, hoveredSeed check
             // Actually, if we hover an image, we want that specific image hovered.
             // We also want to highlight images with same seed.
             
             items.push({
                 ...img,
                 modelName: group.modelName,
                 x,
                 y,
                 rotation: (rnd(img.seed + 2) - 0.5) * 10,
                 scale: 1,
                 opacity: 1,
                 zIndex: 1,
                 isHovered: false, // Set during render map based on state
                 isRelated: false,
                 groupId: groupIdx
             });
         });
     });

     return items;
  }, [groups, workspaceSize.width, workspaceSize.height]);

  // Adjust for hover state
  const finalItems = useMemo(() => {
      // Find the specific item being hovered if any (we track seed, but we need to know WHICH item triggered it to scale it most)
      // Actually simpler: just use hoveredSeed.
      
      return layoutItems.map(item => {
          const isSeedMatch = hoveredSeed === item.seed;
          
          let scale = 1;
          let zIndex = 10;
          let opacity = 1;
          
          if (hoveredSeed !== null) {
              if (isSeedMatch) {
                  scale = 1.5;
                  zIndex = 100;
                  opacity = 1;
              } else {
                  opacity = 0.3;
                  scale = 0.8;
                  zIndex = 1;
              }
          }

          // Safety bounds clamping
          // If item is too close to edge, push it in? 
          // For now, let's just let it be, but the container has overflow hidden.
          // We can clamp x/y.
          const buffer = 80; // half size roughly
          const safeX = Math.max(buffer, Math.min(workspaceSize.width - buffer, item.x));
          const safeY = Math.max(buffer, Math.min(workspaceSize.height - buffer, item.y));

          return {
              ...item,
              x: safeX,
              y: safeY,
              scale,
              zIndex,
              opacity,
              isRelated: isSeedMatch
          };
      });
  }, [layoutItems, hoveredSeed, workspaceSize]);


  return (
    <div className="flex flex-col h-full w-full bg-zinc-100 dark:bg-zinc-950 rounded-xl overflow-hidden relative">
      <div 
        ref={workspaceRef}
        className="flex-grow relative overflow-hidden cursor-default"
        onMouseLeave={() => setHoveredSeed(null)}
      >
        {/* Model Labels / Group Centers */}
         {workspaceSize.width > 0 && groups.map((g, i) => {
             // Re-calc center logic strictly for label
             // Duplicated logic from useMemo, but lightweight
             const groupCount = groups.length;
             let cx = 0.5, cy = 0.5;
             if (groupCount === 2) cx = 0.25 + (i * 0.5);
             else if (groupCount > 2) {
                 const angle = (i / groupCount) * Math.PI * 2 - Math.PI / 2;
                 const radius = 0.25;
                 cx = 0.5 + Math.cos(angle) * radius;
                 cy = 0.5 + Math.sin(angle) * (radius * (workspaceSize.width / workspaceSize.height));
             }
             
             return (
                 <div 
                    key={i} 
                    className="absolute transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center pointer-events-none"
                    style={{ left: cx * workspaceSize.width, top: cy * workspaceSize.height }}
                 >
                    <div className="w-64 h-64 rounded-full bg-zinc-200/50 dark:bg-zinc-800/20 blur-3xl absolute" />
                    <h3 className="relative text-2xl font-black text-zinc-300 dark:text-zinc-800 uppercase tracking-widest select-none z-0">
                        {g.modelName}
                    </h3>
                 </div>
             );
         })}

        {finalItems.map((item) => (
            <div
                key={`${item.modelName}-${item.seed}`}
                onMouseEnter={() => setHoveredSeed(item.seed)}
                className="absolute w-32 aspect-square bg-white dark:bg-zinc-800 rounded-lg shadow-lg border-2 border-white dark:border-zinc-700 transition-all duration-300 ease-out origin-center group"
                style={{
                    left: item.x,
                    top: item.y,
                    transform: `translate(-50%, -50%) rotate(${item.rotation}deg) scale(${item.scale})`,
                    zIndex: item.zIndex,
                    opacity: item.opacity,
                    borderColor: item.isRelated ? '#6366f1' : undefined
                }}
            >
                <img 
                    src={getImageUrl(item.url, item.mtime)} 
                    alt={item.modelName}
                    className="w-full h-full object-cover rounded-md"
                    loading="lazy"
                />
                
                {/* Tooltip / Info - Shows on hover or related */}
                <div 
                    className={`absolute -bottom-8 left-1/2 -translate-x-1/2 bg-black/80 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap transition-opacity pointer-events-none ${item.isRelated ? 'opacity-100' : 'opacity-0'}`}
                >
                    Seed {item.seed}
                </div>
            </div>
        ))}

        {finalItems.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-zinc-400">
                No data to display
            </div>
        )}
      </div>
      
      <div className="h-8 bg-white dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800 flex items-center px-4 justify-between text-[10px] text-zinc-400 uppercase tracking-widest">
         <span>Proximity View</span>
         <span>Hover to isolate seeds</span>
      </div>
    </div>
  );
};
