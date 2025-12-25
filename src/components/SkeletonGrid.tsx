import React from 'react';

export const SkeletonGrid = () => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-6">
      {Array.from({ length: 12 }).map((_, i) => (
        <div 
          key={i} 
          className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden animate-pulse"
        >
          <div className="aspect-[2/3] bg-slate-200 dark:bg-slate-700"></div>
          <div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
             <div className="flex justify-between items-center text-xs">
               <div className="h-4 w-12 bg-slate-200 dark:bg-slate-700 rounded"></div>
               <div className="h-4 w-24 bg-slate-200 dark:bg-slate-700 rounded"></div>
             </div>
          </div>
        </div>
      ))}
    </div>
  );
};
