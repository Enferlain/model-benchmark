import React, { useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import {
  DndContext,
  DragOverlay,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragStartEvent,
  DragEndEvent,
  DragOverEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  sortableKeyboardCoordinates,
} from '@dnd-kit/sortable';
import { Library, ListChecks, ArrowRightLeft, Filter } from 'lucide-react';
import { ModelData } from '../../types';
import { ListContainer } from './ListContainer';
import { Controls } from './Controls';
import { ListItem } from './ListItem';
import { FilterMenu, FilterOptions } from './FilterMenu';
import { PresetMenu } from './PresetMenu';

interface TransferListProps {
  models: ModelData[];
  selectedModelIds: string[];
  onChange: (ids: string[]) => void;
}

export function TransferList({ models, selectedModelIds, onChange }: TransferListProps) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [librarySearch, setLibrarySearch] = useState('');
  const [queueSearch, setQueueSearch] = useState('');
  
  // Selection state for button transfers
  const [checkedItems, setCheckedItems] = useState<Set<string>>(new Set());

  const [filters, setFilters] = useState<FilterOptions>({
    modelTypes: new Set(),
    predictionTypes: new Set(),
    sources: new Set(),
  });

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Derived state
  const libraryModels = useMemo(() => 
    models.filter(m => !selectedModelIds.includes(m.id)), 
  [models, selectedModelIds]);

  const queueModels = useMemo(() => 
    models.filter(m => selectedModelIds.includes(m.id)), 
  [models, selectedModelIds]);

  // Extract available model types for filter
  const availableModelTypes = useMemo(() => {
    const types = new Set<string>();
    models.forEach(m => {
      if (m.model_type) types.add(m.model_type);
    });
    return Array.from(types).sort();
  }, [models]);

  // Filtered lists for display
  const filteredLibrary = useMemo(() => 
    libraryModels.filter(m => {
      // 1. Search Query
      const matchesSearch = 
        m.name.toLowerCase().includes(librarySearch.toLowerCase()) ||
        m.model_type?.toLowerCase().includes(librarySearch.toLowerCase());
      
      if (!matchesSearch) return false;

      // 2. Model Type Filter
      if (filters.modelTypes.size > 0) {
        if (!m.model_type || !filters.modelTypes.has(m.model_type)) return false;
      }

      // 3. Source Filter
      if (filters.sources.size > 0) {
        if (!filters.sources.has(m.source)) return false;
      }

      // 4. Prediction Type Filter
      if (filters.predictionTypes.size > 0) {
        let matchesPred = false;
        
        // Check for Epsilon
        if (filters.predictionTypes.has('epsilon')) {
          if (m.prediction_type === 'epsilon') matchesPred = true;
        }
        
        // Check for V-Pred (exclude ztsnr explicitly if separate category desired, OR include if generic v-pred desired?)
        // User asked for "vpred" and "vpred+ztsnr" as separate.
        if (!matchesPred && filters.predictionTypes.has('v_prediction')) {
          if (m.prediction_type === 'v_prediction' && !m.ztsnr) matchesPred = true;
        }

        // Check for V-Pred + ZTSNR
        if (!matchesPred && filters.predictionTypes.has('v_prediction_ztsnr')) {
          if (m.prediction_type === 'v_prediction' && m.ztsnr) matchesPred = true;
        }

        if (!matchesPred) return false;
      }

      return true;
    }),
  [libraryModels, librarySearch, filters]);

  const filteredQueue = useMemo(() => 
    queueModels.filter(m => 
      m.name.toLowerCase().includes(queueSearch.toLowerCase())
    ),
  [queueModels, queueSearch]);

  // Handle Drag Start
  const handleDragStart = (event: DragStartEvent) => {
    setActiveId(event.active.id as string);
  };

  // Handle Drag Over (Moving between containers)
  const handleDragOver = (event: DragOverEvent) => {
     // We define logic in dragEnd mostly for this simple transfer, 
     // but dragOver is needed if we want to preview insertion.
     // For now, simpler implementation: Drop triggers transfer.
  };

  // Handle Drag End
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveId(null);

    if (!over) return;

    const activeModelId = active.id as string;
    const overId = over.id as string; // Could be a container ID ('library', 'queue') or an item ID

    const isLibraryContainer = overId === 'library';
    const isQueueContainer = overId === 'queue';
    
    // Check if dropping onto an item
    const droppedOnLibraryItem = libraryModels.some(m => m.id === overId);
    const droppedOnQueueItem = queueModels.some(m => m.id === overId);
    
    const isMovingToQueue = isQueueContainer || droppedOnQueueItem;
    const isMovingToLibrary = isLibraryContainer || droppedOnLibraryItem;
    
    const alreadyInQueue = selectedModelIds.includes(activeModelId);

    if (alreadyInQueue && isMovingToLibrary) {
         // Remove from Queue
         onChange(selectedModelIds.filter(id => id !== activeModelId));
    } else if (!alreadyInQueue && isMovingToQueue) {
         // Add to Queue
         onChange([...selectedModelIds, activeModelId]);
    }
  };

  // Button Transfer Logic
  const handleSelect = (id: string) => {
    const newChecked = new Set(checkedItems);
    if (newChecked.has(id)) {
      newChecked.delete(id);
    } else {
      newChecked.add(id);
    }
    setCheckedItems(newChecked);
  };

  const handleMoveRight = () => {
    // Move all checked items that are in Library -> Queue
    const itemsToMove = libraryModels.filter(m => checkedItems.has(m.id)).map(m => m.id);
    onChange([...selectedModelIds, ...itemsToMove]);
    setCheckedItems(new Set()); // Clear selection
  };

  const handleMoveLeft = () => {
    // Move all checked items that are in Queue -> Library (Remove from selected)
    const itemsToRemove = queueModels.filter(m => checkedItems.has(m.id)).map(m => m.id);
    onChange(selectedModelIds.filter(id => !itemsToRemove.includes(id)));
    setCheckedItems(new Set()); // Clear selection
  };
  
  // Determine if buttons should be enabled
  const canMoveRight = libraryModels.some(m => checkedItems.has(m.id));
  const canMoveLeft = queueModels.some(m => checkedItems.has(m.id));

  const activeModel = useMemo(() => models.find(m => m.id === activeId), [activeId, models]);
  
  // Determine highlight targets
  const isDraggingFromLibrary = activeModel && !selectedModelIds.includes(activeModel.id);
  const isDraggingFromQueue = activeModel && selectedModelIds.includes(activeModel.id);

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragEnd={handleDragEnd}
    >
      <div className="flex flex-col md:flex-row gap-4 h-[600px] w-full items-stretch">
        
        {/* Left: Library */}
        <ListContainer
          id="library"
          title="Library"
          icon={<Library size={18} />}
          items={filteredLibrary}
          selectedIds={checkedItems}
          onSelect={handleSelect}
          search={librarySearch}
          onSearchChange={setLibrarySearch}
          placeholder="Filter available models..."
          isDropTarget={!!isDraggingFromQueue}
          headerAction={
            <FilterMenu 
               filters={filters} 
               onChange={setFilters} 
               availableModelTypes={availableModelTypes} 
            />
          }
        />

        {/* Middle: Controls */}
        <Controls
          onMoveRight={handleMoveRight}
          onMoveLeft={handleMoveLeft}
          canMoveRight={canMoveRight}
          canMoveLeft={canMoveLeft}
        />

        {/* Right: Queue */}
        <ListContainer
          id="queue"
          title="Benchmark Queue"
          icon={<ListChecks size={18} />}
          items={filteredQueue}
          selectedIds={checkedItems}
          onSelect={handleSelect}
          search={queueSearch}
          onSearchChange={setQueueSearch}
          placeholder="Search queued models..."
          isDropTarget={!!isDraggingFromLibrary}
          headerAction={
            <PresetMenu 
              currentIds={selectedModelIds}
              onLoad={onChange}
            />
          }
        />

      </div>

      {typeof document !== 'undefined' && createPortal(
        <DragOverlay>
          {activeModel ? <ListItem model={activeModel} isDragging /> : null}
        </DragOverlay>,
        document.body
      )}
    </DndContext>
  );
}
