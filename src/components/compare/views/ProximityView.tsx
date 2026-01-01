import React, { useState, useRef, useMemo, useEffect } from 'react';
import { ModelOutput } from '../../../types';
import { getImageUrl } from '../utils';
import { stringToColor } from '../../../utils/colorUtils';

interface GroupedImages {
  id?: string;
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
  color: string;
}

export const ProximityView: React.FC<Props> = ({ groups }) => {
  const [workspaceSize, setWorkspaceSize] = useState({ width: 0, height: 0 });
  const [hoveredSeed, setHoveredSeed] = useState<number | null>(null);
  const [quicklookIndex, setQuicklookIndex] = useState<number | null>(null);
  const [layoutMode, setLayoutMode] = useState<'cascade' | 'row' | 'column' | 'scatter' | 'bloom'>('cascade');
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

  // Lock body scroll when quicklook is open
  useEffect(() => {
    if (quicklookIndex !== null) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [quicklookIndex]);

  // Compute Layout
  const layoutItems = useMemo(() => {
     if (workspaceSize.width === 0 || workspaceSize.height === 0) return [];
     if (groups.length === 0) return [];

     const CARD_WIDTH = 128;
     const CARD_HEIGHT = 192;
     const PADDING = 40; // Space between clusters

     // 1. Calculate LOCAL positions for each group (relative to 0,0)
     //    and determine their bounding boxes.
     const groupLayouts = groups.map((group) => {
         const countInGroup = group.images.length;
         const localItems: { x: number, y: number, rot: number, img: ModelOutput, idx: number }[] = [];
         
         const halfCount = (countInGroup - 1) / 2;
         let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

         group.images.forEach((img, imgIdx) => {
             let offsetX = 0, offsetY = 0, rot = 0;

             switch (layoutMode) {
                case 'cascade': {
                  const stepX = 25, stepY = 30;
                  offsetX = (imgIdx - halfCount) * stepX;
                  offsetY = (imgIdx - halfCount) * stepY;
                  rot = (imgIdx - halfCount) * 3;
                  break;
                }
                case 'row': {
                  const stepX = 50; 
                  offsetX = (imgIdx - halfCount) * stepX;
                  offsetY = 0;
                  rot = 0;
                  break;
                }
                case 'column': {
                  const stepY = 40;
                  offsetX = 0;
                  offsetY = (imgIdx - halfCount) * stepY;
                  rot = 0;
                  break;
                }
                case 'scatter': {
                  const pseudoRandom = (seed: number) => {
                    const x = Math.sin(seed) * 10000;
                    return x - Math.floor(x);
                  };
                  const posSeed = (img.seed * 9973) + (imgIdx * 991) + 12345;
                  const spread = 60;
                  offsetX = (pseudoRandom(posSeed) - 0.5) * spread * 2;
                  offsetY = (pseudoRandom(posSeed + 1) - 0.5) * spread * 2;
                  rot = 0;
                  break;
                }
                case 'bloom': {
                  const angle = (imgIdx / countInGroup) * Math.PI * 2 - Math.PI / 2;
                  const radius = 35 + countInGroup * 5;
                  offsetX = Math.cos(angle) * radius;
                    offsetY = Math.sin(angle) * radius;
                  rot = 0;
                  break;
                }
             }

             localItems.push({ x: offsetX, y: offsetY, rot, img, idx: imgIdx });

             // Update bounds (considering card size approximations)
             // We use a simplified box for the rotated card
             const hw = CARD_WIDTH / 2;
             const hh = CARD_HEIGHT / 2;
             minX = Math.min(minX, offsetX - hw);
             maxX = Math.max(maxX, offsetX + hw);
             minY = Math.min(minY, offsetY - hh);
             maxY = Math.max(maxY, offsetY + hh);
         });

         const width = maxX - minX;
         const height = maxY - minY;
         
         // Center of the bounding box relative to the group origin (0,0)
         const centerOffsetX = (minX + maxX) / 2;
         const centerOffsetY = (minY + maxY) / 2;

         return {
             localItems,
             width,
             height,
             halfWidth: width / 2,
             halfHeight: height / 2,
             centerOffsetX,
             centerOffsetY
         };
     });


     // 2. Initial Global Placement (Target Centers)
     const groupCount = groups.length;
     const targets = groups.map((_, i) => {
         let tx = 0, ty = 0;
         if (groupCount === 1) { tx = 0.5; ty = 0.5; }
         else if (groupCount === 2) { tx = 0.25 + (i * 0.5); ty = 0.5; }
         else if (groupCount <= 4) {
             const row = Math.floor(i / 2);
             const col = i % 2;
             tx = 0.25 + (col * 0.5); 
             ty = 0.25 + (row * 0.5);
         } else {
             const angle = (i / groupCount) * Math.PI * 2 - Math.PI / 2;
             const radius = 0.3;
             tx = 0.5 + Math.cos(angle) * radius;
             ty = 0.5 + Math.sin(angle) * (radius * (workspaceSize.width / workspaceSize.height));
         }
         return { 
             x: tx * workspaceSize.width, 
             y: ty * workspaceSize.height,
             vx: 0, vy: 0 
         };
     });

     // Initialize current positions at targets
     const currentPositions = targets.map(t => ({ x: t.x, y: t.y }));


     // 3. Collision Resolution (Relaxation Loop)
     const iterations = 50;
     for (let iter = 0; iter < iterations; iter++) {
        // Move towards targets (Attraction)
        for(let i=0; i<groupCount; i++) {
            const curr = currentPositions[i];
            const target = targets[i];
            const pullStrength = 0.05; 
            curr.x += (target.x - curr.x) * pullStrength;
            curr.y += (target.y - curr.y) * pullStrength;
        }

        // Solve Collisions (Repulsion)
        for (let i = 0; i < groupCount; i++) {
             for (let j = i + 1; j < groupCount; j++) {
                 const g1 = groupLayouts[i];
                 const g2 = groupLayouts[j];
                 const p1 = currentPositions[i];
                 const p2 = currentPositions[j];

                 // Determine effective bounds in world space
                 // We use the center of the bounding box for distance checks more accurately
                 const c1x = p1.x + g1.centerOffsetX;
                 const c1y = p1.y + g1.centerOffsetY;
                 const c2x = p2.x + g2.centerOffsetX;
                 const c2y = p2.y + g2.centerOffsetY;

                 const dx = c2x - c1x;
                 const dy = c2y - c1y;
                 
                 // Simple box collision check?
                 // Let's use separation of axes or just simple distance vs radii for smoothness
                 // But valid overlap implies: 
                 // |dx| < (w1/2 + w2/2) AND |dy| < (h1/2 + h2/2)
                 
                 const minDistX = g1.halfWidth + g2.halfWidth + PADDING;
                 const minDistY = g1.halfHeight + g2.halfHeight + PADDING;

                 if (Math.abs(dx) < minDistX && Math.abs(dy) < minDistY) {
                     // Overlap detected!
                     // Calculate overlap amounts
                     const overlapX = minDistX - Math.abs(dx);
                     const overlapY = minDistY - Math.abs(dy);

                     // Push along the axis of least overlap (usually easiest way out)
                     if (overlapX < overlapY) {
                         const pushDir = dx > 0 ? -1 : 1; 
                         const pushAmt = overlapX * 0.5; // Split the move
                         // P1 moves left/right
                         currentPositions[i].x += pushDir * pushAmt;
                         currentPositions[j].x -= pushDir * pushAmt;
                     } else {
                         const pushDir = dy > 0 ? -1 : 1;
                         const pushAmt = overlapY * 0.5;
                         // P1 moves up/down
                         currentPositions[i].y += pushDir * pushAmt;
                         currentPositions[j].y -= pushDir * pushAmt;
                     }
                 }
             }
        }
        
        // Wall Constraints
        for (let i = 0; i < groupCount; i++) {
             const g = groupLayouts[i];
             const p = currentPositions[i];
             
             // Min/Max for the center p
             // Bound left: p.x + g.centerOffsetX - g.halfWidth >= 0
             // => p.x >= g.halfWidth - g.centerOffsetX
             
             const minX = g.halfWidth - g.centerOffsetX + 20;
             const maxX = workspaceSize.width - (g.halfWidth + g.centerOffsetX) - 20;
             const minY = g.halfHeight - g.centerOffsetY + 20;
             const maxY = workspaceSize.height - (g.halfHeight + g.centerOffsetY) - 20;

             p.x = Math.max(minX, Math.min(maxX, p.x));
             p.y = Math.max(minY, Math.min(maxY, p.y));
        }
     }


     // 4. Finalize
     const finalItems: LayoutItem[] = [];
     groups.forEach((group, groupIdx) => {
          const layout = groupLayouts[groupIdx];
          const pos = currentPositions[groupIdx];
          const color = stringToColor(group.id || group.modelName);

          layout.localItems.forEach(item => {
              finalItems.push({
                  ...item.img,
                  modelName: group.modelName,
                  x: pos.x + item.x,
                  y: pos.y + item.y,
                  rotation: item.rot,
                  scale: 1,
                  opacity: 1,
                  zIndex: item.idx + 1,
                  isHovered: false,
                  isRelated: false,
                  groupId: groupIdx,
                  color
              });
          });
     });

     return finalItems;
  }, [groups, workspaceSize.width, workspaceSize.height, layoutMode]);

  // Adjust for hover state
  const finalItems = useMemo(() => {
      return layoutItems.map(item => {
          const isSeedMatch = hoveredSeed === item.seed;
          
          let scale = 1;
          let zIndex = 10;
          let opacity = 1;
          
          if (hoveredSeed !== null) {
              if (isSeedMatch) {
                  scale = 1.6;
                  zIndex = 100;
                  opacity = 1;
              } else {
                  opacity = 0.2;
                  scale = 0.8;
                  zIndex = 1;
              }
          }

          // Positional clamping is now done in layoutItems
          return {
              ...item,
              scale,
              zIndex,
              opacity,
              isRelated: isSeedMatch
          };
      });
  }, [layoutItems, hoveredSeed]);


  return (
    <div className="flex flex-col h-full w-full min-h-[800px] bg-zinc-100 dark:bg-zinc-950 rounded-xl overflow-hidden relative">
      <div 
        ref={workspaceRef}
        className="flex-grow relative overflow-hidden"
        onMouseLeave={() => setHoveredSeed(null)}
        onMouseMove={(e) => {
            if (!workspaceRef.current || layoutItems.length === 0) return;
            const rect = workspaceRef.current.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            // Find closest item based on BASE positions (layoutItems, not scaled finalItems)
            // This ensures scaled-up cards don't block detection of neighbors
            let minDist = Infinity;
            let closestSeed: number | null = null;

            layoutItems.forEach(item => {
                const dx = mouseX - item.x;
                const dy = mouseY - item.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < minDist) {
                    minDist = dist;
                    closestSeed = item.seed;
                }
            });

            // Only focus if within reasonable distance (prevents focus when far from all cards)
            if (minDist < 150) {
                setHoveredSeed(closestSeed);
            } else {
                setHoveredSeed(null);
            }
        }}
      >
        {/* Dotted Background Pattern */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.15] [background-image:radial-gradient(#000_1px,transparent_1px)] dark:[background-image:radial-gradient(#fff_1px,transparent_1px)] [background-size:20px_20px]" />

        {/* Legend */}
        <div className="absolute top-4 right-4 bg-white/90 dark:bg-slate-900/90 backdrop-blur-sm p-3 rounded-lg shadow-md border border-slate-200 dark:border-slate-800 z-50 max-w-[200px] pointer-events-none select-none">
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">Models</h4>
            <div className="flex flex-col gap-1.5">
                {groups.map((g, i) => (
                    <div key={i} className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: stringToColor(g.id || g.modelName) }} />
                        <span className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate min-w-0 flex-1" title={g.modelName}>
                            {g.modelName}
                        </span>
                    </div>
                ))}
            </div>
        </div>

        {finalItems.map((item, idx) => (
            <div
                key={`${item.modelName}-${item.seed}`}
                onClick={() => setQuicklookIndex(idx)}
                className="absolute w-32 aspect-[2/3] bg-white dark:bg-zinc-800 rounded-lg shadow-lg transition-all duration-300 ease-out origin-center group overflow-hidden cursor-pointer"
                style={{
                    left: item.x,
                    top: item.y,
                    transform: `translate(-50%, -50%) rotate(${item.rotation}deg) scale(${item.scale})`,
                    zIndex: item.zIndex,
                    opacity: item.opacity,
                    // Subtler "aura" glow instead of hard border
                    boxShadow: `0 0 15px -1px ${(item as any).color}, 0 4px 6px -1px rgba(0, 0, 0, 0.1)`
                }}
            >
                {/* Thin border for definition against similar backgrounds */}
                <div className="absolute inset-0 rounded-lg border border-white/20 dark:border-black/20 pointer-events-none z-20" />

                <img 
                    src={getImageUrl(item.url, item.mtime)} 
                    alt={item.modelName}
                    className="w-full h-full object-cover rounded-lg" // Match parent radius
                    loading="lazy"
                />
                
                {/* Tooltip / Info - Shows on hover or related */}
                <div 
                    className={`absolute top-1.5 left-1.5 bg-black/60 backdrop-blur-md text-white text-[9px] px-1.5 py-0.5 rounded shadow-sm font-mono border border-white/10 ${item.isRelated ? 'opacity-100' : 'opacity-0'}`}
                >
                    {item.seed}
                </div>
            </div>
        ))}

        {finalItems.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-400 p-4 text-center pointer-events-none">
                <p>No data to display in Proximity View</p>
                <div className="text-xs font-mono mt-2 opacity-50">
                    Workspace: {workspaceSize.width}x{workspaceSize.height}<br/>
                    Groups: {groups.length}<br/>
                    Images: {groups.reduce((acc, g) => acc + g.images.length, 0)}
                </div>
            </div>
        )}
      </div>
      
       <div className="h-10 bg-white dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800 flex items-center px-4 justify-between text-[10px] text-zinc-400">
          <div className="flex items-center gap-2">
            <span className="uppercase tracking-widest">Layout:</span>
            {(['cascade', 'row', 'column', 'scatter', 'bloom'] as const).map(mode => (
              <button
                key={mode}
                onClick={() => setLayoutMode(mode)}
                className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                  layoutMode === mode 
                    ? 'bg-indigo-500 text-white' 
                    : 'bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
          <span className="uppercase tracking-widest">Click to enlarge • Hover to isolate</span>
       </div>

       {/* Quicklook Modal */}
       {quicklookIndex !== null && (
         <div 
           className="fixed inset-0 z-[200] bg-black/80 backdrop-blur-sm flex items-center justify-center"
           onClick={() => setQuicklookIndex(null)}
           onKeyDown={(e) => {
             if (e.key === 'Escape') setQuicklookIndex(null);
             if (e.key === 'ArrowLeft') setQuicklookIndex(prev => prev !== null ? Math.max(0, prev - 1) : null);
             if (e.key === 'ArrowRight') setQuicklookIndex(prev => prev !== null ? Math.min(finalItems.length - 1, prev + 1) : null);
           }}
           onWheel={(e) => {
             e.preventDefault();
             if (e.deltaY > 0) {
               setQuicklookIndex(prev => prev !== null ? Math.min(finalItems.length - 1, prev + 1) : null);
             } else {
               setQuicklookIndex(prev => prev !== null ? Math.max(0, prev - 1) : null);
             }
           }}
           tabIndex={0}
           ref={(el) => el?.focus()}
         >
           {/* Top Info Bar */}
           <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-3 text-white text-sm">
             <span className="font-semibold px-3 py-1 rounded-full" style={{ backgroundColor: finalItems[quicklookIndex].color + '40', color: finalItems[quicklookIndex].color }}>
               {finalItems[quicklookIndex].modelName}
             </span>
             <span className="opacity-60 font-mono">#{finalItems[quicklookIndex].seed}</span>
             <span className="opacity-40 text-xs">{quicklookIndex + 1}/{finalItems.length}</span>
           </div>

           {/* Navigation */}
           <button
             className="absolute left-4 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full bg-white/20 hover:bg-white/40 text-white text-2xl flex items-center justify-center transition-colors disabled:opacity-30"
             onClick={(e) => { e.stopPropagation(); setQuicklookIndex(prev => prev !== null ? Math.max(0, prev - 1) : null); }}
             disabled={quicklookIndex === 0}
           >
             ←
           </button>
           
           <button
             className="absolute right-4 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full bg-white/20 hover:bg-white/40 text-white text-2xl flex items-center justify-center transition-colors disabled:opacity-30"
             onClick={(e) => { e.stopPropagation(); setQuicklookIndex(prev => prev !== null ? Math.min(finalItems.length - 1, prev + 1) : null); }}
             disabled={quicklookIndex === finalItems.length - 1}
           >
             →
           </button>

           {/* Image */}
           <div className="max-w-[80vw] max-h-[80vh] relative" onClick={(e) => e.stopPropagation()}>
             <img 
               src={getImageUrl(finalItems[quicklookIndex].url, finalItems[quicklookIndex].mtime)}
               alt={finalItems[quicklookIndex].modelName}
               className="max-w-full max-h-[80vh] rounded-lg shadow-2xl"
             />
           </div>

           {/* Bottom hint */}
           <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/40 text-xs">
             Scroll or ← → to navigate • Click or ESC to close
           </div>
         </div>
       )}
    </div>
  );
};
