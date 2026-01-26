import React from 'react';
import { useDroppable } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { Search } from 'lucide-react';
import { ListItem, SortableListItem } from './ListItem';
import { ModelData } from '../../types';

interface ListContainerProps {
  id: string;
  title: string;
  items: ModelData[];
  selectedIds: Set<string>;
  onSelect: (id: string, multi: boolean) => void;
  search: string;
  onSearchChange: (value: string) => void;
  placeholder?: string;
  icon?: React.ReactNode;
  isDropTarget?: boolean;
  headerAction?: React.ReactNode;
}

export function ListContainer({ 
  id, 
  title, 
  items, 
  selectedIds, 
  onSelect,
  search,
  onSearchChange,
  placeholder = "Search models...",
  icon,
  isDropTarget,
  headerAction
}: ListContainerProps) {
  const { setNodeRef, isOver } = useDroppable({ id });

  return (
    <div 
      className={`
        flex-1 flex flex-col h-[500px] rounded-2xl border-2 backdrop-blur-sm overflow-hidden transition-all duration-200
        ${isOver 
           ? 'border-blue-500 bg-blue-50/80 dark:bg-blue-900/40' 
           : isDropTarget 
             ? 'border-blue-300 dark:border-blue-700 border-dashed bg-blue-50/30 dark:bg-blue-900/10'
             : 'border-slate-200/50 dark:border-slate-700/50 bg-slate-50/50 dark:bg-slate-900/30'
        }
      `}
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-200/50 dark:border-slate-700/50 bg-white/50 dark:bg-slate-800/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-slate-800 dark:text-slate-200 font-medium">
             {icon}
             <span>{title}</span>
             <span className="px-2 py-0.5 rounded-full bg-slate-200 dark:bg-slate-700 text-xs text-slate-600 dark:text-slate-300">
               {items.length}
             </span>
          </div>
        </div>
        
        {/* Search & Action Row */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={14} />
            <input
              type="text"
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder={placeholder}
              className="w-full pl-9 pr-3 py-1.5 text-sm bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500/50 transition-all placeholder:text-slate-400"
            />
          </div>
          {headerAction}
        </div>
      </div>

      {/* List Area */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-2 custom-scrollbar">
        <SortableContext id={id} items={items.map(m => m.id)} strategy={verticalListSortingStrategy}>
          <div ref={setNodeRef} className="min-h-full space-y-2">
            {items.length === 0 ? (
               <div className="h-full flex flex-col items-center justify-center text-slate-400 text-sm italic p-8 text-center border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl">
                 No models found
               </div>
            ) : (
                items.map((model) => (
                  <SortableListItem
                    key={model.id}
                    model={model}
                    isSelected={selectedIds.has(model.id)}
                    onClick={() => onSelect(model.id, false)}
                  />
                ))
            )}
          </div>
        </SortableContext>
      </div>
    </div>
  );
}
