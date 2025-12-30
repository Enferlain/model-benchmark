import React from 'react';
import { ModelOutput } from '../../../types';
import { getImageUrl } from '../utils';

interface Props {
  images: (ModelOutput | undefined)[];
  modelNames: string[];
}

export const SideBySideView: React.FC<Props> = ({ images, modelNames }) => {
  // Dynamic grid cols based on count
  const count = images.length;
  let gridCols = 'grid-cols-2';
  if (count === 1) gridCols = 'grid-cols-1';
  if (count >= 3) gridCols = 'grid-cols-3';
  if (count >= 5) gridCols = 'grid-cols-4';

  return (
    <div className={`grid ${gridCols} gap-4 w-full h-full p-4`}>
      {images.map((img, idx) => (
        <div key={idx} className="flex flex-col gap-2 min-h-[300px]">
           <div className="relative w-full h-full bg-white dark:bg-black rounded-lg overflow-hidden border-2 border-slate-200 dark:border-slate-700 shadow-md">
             {img ? (
               <img
                 src={getImageUrl(img.url)}
                 alt={modelNames[idx]}
                 className="w-full h-full object-contain"
               />
             ) : (
               <div className="w-full h-full flex items-center justify-center text-slate-400">
                 No Image
               </div>
             )}
             <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded shadow-sm backdrop-blur-sm">
                {modelNames[idx]}
             </div>
           </div>
        </div>
      ))}
    </div>
  );
};
