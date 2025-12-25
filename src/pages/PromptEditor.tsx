import React, { useState, useEffect, useMemo } from 'react';
import { Plus, Search, Trash2, Save, X, Image as ImageIcon, FileText, AlertCircle, Loader2 } from 'lucide-react';
import { fetchPrompts, createPrompt, updatePromptText, deletePrompt } from '../services/api';
import { PromptData } from '../types';
import { API_BASE } from '../services/api'; 
import {
  DndContext, 
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

export default function PromptEditor() {
  const [prompts, setPrompts] = useState<PromptData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Editor State
  const [editText, setEditText] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  
  // Creation State
  const [isCreating, setIsCreating] = useState(false);
  const [newPromptText, setNewPromptText] = useState('');
  const [newPromptImage, setNewPromptImage] = useState<File | null>(null);
  
  // Fetch on mount
  useEffect(() => {
    loadPrompts();
  }, []);
  
  const loadPrompts = async () => {
    setIsLoading(true);
    try {
      const data = await fetchPrompts();
      if (Array.isArray(data)) {
        setPrompts(data);
      } else {
        console.error("Fetched prompts is not an array:", data);
        setPrompts([]);
      }
    } catch (err) {
      console.error("Failed to load prompts", err);
      setPrompts([]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Selection Logic
  const selectedPrompt = useMemo(() => 
    prompts.find(p => p.id === selectedId), 
  [prompts, selectedId]);
  
  useEffect(() => {
    if (selectedPrompt) {
      setEditText(selectedPrompt.text);
      setIsDirty(false);
    }
  }, [selectedPrompt]);
  
  // Handlers
  const handleSave = async () => {
    if (!selectedPrompt) return;
    setIsSaving(true);
    try {
      await updatePromptText(selectedPrompt.filename, editText);
      setIsDirty(false);
      // Update local state without refetching for speed
      setPrompts(prev => prev.map(p => 
        p.id === selectedId ? { ...p, text: editText } : p
      ));
    } catch (err) {
      alert("Failed to save prompt");
    } finally {
      setIsSaving(false);
    }
  };
  
  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this prompt?")) return;
    
    try {
      // Find filename
      const p = prompts.find(x => x.id === id);
      if (p) {
        await deletePrompt(p.filename);
        setPrompts(prev => prev.filter(x => x.id !== id));
        if (selectedId === id) setSelectedId(null);
      }
    } catch (err) {
      alert("Failed to delete prompt");
    }
  };
  
  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPromptText) return;
    
    setIsSaving(true);
    try {
      const formData = new FormData();
      formData.append('text', newPromptText);
      if (newPromptImage) {
        formData.append('image', newPromptImage);
      }
      
      const res = await createPrompt(formData);
      if (res.status === 'success') {
        setIsCreating(false);
        setNewPromptText('');
        setNewPromptImage(null);
        loadPrompts(); // Refresh list
      }
    } catch (err) {
      alert("Failed to create prompt");
    } finally {
      setIsSaving(false);
    }
  };
  
  // Handlers
  const handleToggle = async (e: React.MouseEvent, prompt: PromptData) => {
    e.stopPropagation();
    try {
        const newStatus = !prompt.enabled;
        // Optimistic update
        setPrompts(prev => prev.map(p => 
            p.id === prompt.id ? { ...p, enabled: newStatus } : p
        ));
        
        // Pass object directly, do NOT stringify, otherwise api.ts treats it as text update
        await updatePromptText(prompt.filename, { enabled: newStatus });
    } catch (err) {
        alert("Failed to toggle prompt");
        loadPrompts(); // Revert
    }
  };

  const filteredPrompts = useMemo(() => {
    if (!Array.isArray(prompts)) return [];
    
    const q = searchQuery.toLowerCase();
    return prompts.filter(p => {
       if (!p) return false;
       const text = p.text || '';
       const id = p.id || '';
       return text.toLowerCase().includes(q) || id.toLowerCase().includes(q);
    });
  }, [prompts, searchQuery]);

  // DnD Sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
        activationConstraint: {
            distance: 8,
        },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = async (event: DragEndEvent) => {
    const {active, over} = event;
    
    if (over && active.id !== over.id) {
      setPrompts((items) => {
        const oldIndex = items.findIndex(i => i.id === active.id);
        const newIndex = items.findIndex(i => i.id === over.id);
        
        const newItems = arrayMove(items, oldIndex, newIndex);
        
        // Persist order
        const order = newItems.map(p => p.filename);
        fetch(`${API_BASE}/prompts/reorder`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({order})
        }).catch(err => console.error("Failed to save order", err));
        
        return newItems;
      });
    }
  };

  return (
    <div className="max-w-[1800px] mx-auto h-[calc(100vh-100px)] pt-6 px-6 flex gap-6">
      
      {/* Left Sidebar: List */}
      <div className="w-1/3 min-w-[320px] max-w-[450px] flex flex-col bg-white dark:bg-slate-800/50 rounded-2xl shadow-lg border border-slate-200 dark:border-white/5 overflow-hidden backdrop-blur-sm">
        <div className="p-4 border-b border-slate-200 dark:border-white/5 flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
            <input 
              type="text"
              placeholder="Search prompts..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-100 dark:bg-slate-900/50 border-none rounded-xl text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all"
            />
          </div>
          <button 
            onClick={() => setIsCreating(true)}
            className="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl transition-colors shadow-lg shadow-blue-500/20"
            title="Add New Prompt"
          >
            <Plus size={20} />
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2 space-y-2">
          {isLoading ? (
            <div className="flex justify-center p-8 text-slate-400"><Loader2 className="animate-spin" /></div>
          ) : filteredPrompts.length === 0 ? (
             <div className="text-center p-8 text-slate-500 text-sm">No prompts found.</div>
          ) : (
            <DndContext 
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext 
                items={filteredPrompts.map(p => p.id)}
                strategy={verticalListSortingStrategy}
              >
                {filteredPrompts.map((prompt, idx) => (
                  <SortableItem
                    key={prompt.id}
                    prompt={prompt}
                    idx={idx}
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                    onToggle={handleToggle}
                    onDelete={handleDelete}
                  />
                ))}
              </SortableContext>
            </DndContext>
          )}
        </div>
      </div>
      
      {/* Right Sidebar: Editor */}
      <div className="flex-1 bg-white dark:bg-slate-800/50 rounded-2xl shadow-lg border border-slate-200 dark:border-white/5 overflow-hidden flex flex-col backdrop-blur-sm relative">
        {selectedPrompt ? (
          <>
            <div className="p-4 border-b border-slate-200 dark:border-white/5 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/20">
              <div className="flex items-center gap-3">
                 {selectedPrompt.image ? <ImageIcon size={18} className="text-blue-500"/> : <FileText size={18} className="text-slate-500"/>}
                 <span className="font-mono text-sm text-slate-500">{selectedPrompt.filename}</span>
              </div>
              <button
                onClick={handleSave}
                disabled={!isDirty || isSaving}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isDirty 
                    ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20' 
                    : 'bg-slate-100 dark:bg-white/5 text-slate-400 cursor-not-allowed'
                }`}
              >
                {isSaving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                Save Changes
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
               {selectedPrompt.image && (
                 <div className="shrink-0">
                   <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Reference Image</h3>
                   <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-white/10 shadow-lg inline-block md:max-w-md lg:max-w-lg bg-black/5 dark:bg-black/20">
                     <img src={selectedPrompt.image} alt="Reference" className="max-h-[400px] w-auto object-contain" />
                   </div>
                 </div>
               )}
               
               <div className="flex-1 flex flex-col min-h-[200px]">
                 <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Prompt Text</h3>
                 <textarea
                   value={editText}
                   onChange={e => {
                     setEditText(e.target.value);
                     setIsDirty(true);
                   }}
                   className="flex-1 w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-white/10 rounded-xl p-4 font-mono text-sm leading-relaxed text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-blue-500 outline-none resize-none"
                   placeholder="Enter prompt text here..."
                 />
               </div>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
             <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-white/5 flex items-center justify-center mb-4">
               <FileText size={32} className="opacity-50"/>
             </div>
             <p>Select a prompt to edit</p>
          </div>
        )}
        
         {/* Creation Modal Overlay */}
         {isCreating && (
           <div className="absolute inset-0 z-50 bg-white/90 dark:bg-slate-900/95 backdrop-blur-md flex items-center justify-center p-8">
             <div className="w-full max-w-2xl flex flex-col h-full max-h-[600px] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl border border-slate-200 dark:border-white/10 animation-fade-in-up">
                <div className="p-6 border-b border-slate-200 dark:border-white/5 flex justify-between items-center">
                  <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100">Create New Prompt</h2>
                  <button onClick={() => setIsCreating(false)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"><X size={24}/></button>
                </div>
                
                <form onSubmit={handleCreate} className="flex-1 p-6 flex flex-col gap-6 overflow-y-auto">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Prompt Text</label>
                      <textarea
                        required
                        value={newPromptText}
                        onChange={e => setNewPromptText(e.target.value)}
                        className="w-full h-40 bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-white/10 rounded-xl p-4 font-mono text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                        placeholder="Enter your new prompt here..."
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Reference Image (Optional)</label>
                      <div className="border-2 border-dashed border-slate-200 dark:border-white/10 rounded-xl p-8 text-center cursor-pointer hover:bg-slate-50 dark:hover:bg-white/5 transition-colors relative">
                        <input 
                          type="file" 
                          accept="image/*"
                          onChange={e => setNewPromptImage(e.target.files?.[0] || null)}
                          className="absolute inset-0 opacity-0 cursor-pointer"
                        />
                         {newPromptImage ? (
                           <div className="flex flex-col items-center text-blue-500">
                             <ImageIcon size={32} className="mb-2"/>
                             <span className="font-medium">{newPromptImage.name}</span>
                           </div>
                         ) : (
                           <div className="flex flex-col items-center text-slate-400">
                             <ImageIcon size={32} className="mb-2"/>
                             <span>Click or drag to upload an image</span>
                           </div>
                         )}
                      </div>
                    </div>
                </form>
                
                <div className="p-6 border-t border-slate-200 dark:border-white/5 flex justify-end gap-3">
                   <button 
                     type="button" 
                     onClick={() => setIsCreating(false)}
                     className="px-4 py-2 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-white/5 rounded-lg text-sm font-medium"
                   >
                     Cancel
                   </button>
                   <button 
                     onClick={handleCreate}
                     className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium shadow-lg shadow-blue-500/20"
                   >
                     Create Prompt
                   </button>
                </div>
             </div>
           </div>
         )}
      </div>
    </div>
  );
}

// Sortable Item Component
function SortableItem({ prompt, idx, selectedId, onSelect, onToggle, onDelete }: any) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({id: prompt.id});

  const style = {
    transform: transform ? `translate3d(0, ${transform.y}px, 0)` : undefined,
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 50 : 'auto',
  };

  return (
    <div 
      ref={setNodeRef} 
      style={style} 
      {...attributes} 
      {...listeners}
      onClick={() => onSelect(prompt.id)}
      className={`group p-3 rounded-xl cursor-pointer transition-colors border border-transparent relative select-none ${
        selectedId === prompt.id 
          ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800/50 shadow-sm' 
          : 'hover:bg-white dark:hover:bg-white/5 hover:border-slate-200 dark:hover:border-white/10'
      } ${!prompt.enabled ? 'opacity-50 grayscale' : ''}`}
    >
      {/* Index Badge */}
      <div className="absolute top-2 left-2 z-10 w-6 h-6 rounded-full bg-black/50 text-white text-[10px] font-mono flex items-center justify-center backdrop-blur-sm">
          {idx + 1}
      </div>

      <div className="flex gap-3">
        <div className="w-16 h-16 shrink-0 bg-slate-100 dark:bg-slate-900 rounded-lg overflow-hidden border border-slate-200 dark:border-white/5 flex items-center justify-center text-slate-300">
          {prompt.image ? (
            <img src={prompt.image} alt="ref" className="w-full h-full object-cover" />
          ) : (
            <FileText size={24} />
          )}
        </div>
        <div className="flex-1 min-w-0 flex flex-col justify-center">
          <div className="flex justify-between items-center mb-0.5">
            <span className="text-xs font-bold text-slate-700 dark:text-slate-200 truncate pl-6">
               {prompt.id}
            </span>
            <div className="flex items-center gap-1" onMouseDown={e => e.stopPropagation()}>
                {/* Toggle Switch */}
                <div 
                   onClick={(e) => onToggle(e, prompt)}
                   className={`w-8 h-4 rounded-full p-0.5 cursor-pointer transition-colors ${prompt.enabled ? 'bg-green-500' : 'bg-slate-300 dark:bg-slate-600'}`}
                   title={prompt.enabled ? "Enabled" : "Disabled"}
                >
                   <div className={`w-3 h-3 bg-white rounded-full shadow-sm transition-transform ${prompt.enabled ? 'translate-x-4' : 'translate-x-0'}`} />
                </div>
                
                <button 
                  onClick={(e) => onDelete(e, prompt.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 hover:text-red-500 dark:hover:bg-red-900/30 rounded transition-all ml-1"
                  title="Delete Prompt"
                >
                  <Trash2 size={14} />
                </button>
            </div>
          </div>
          {/* Filename sub-label */}
          <div className="text-[10px] font-mono text-slate-400 mb-1 truncate">
              {prompt.filename}
          </div>
          
          {/* Content Preview */}
          <p className={`text-xs line-clamp-1 leading-relaxed ${prompt.text ? 'text-slate-600 dark:text-slate-400' : 'text-slate-400/50 italic'}`}>
            {prompt.text || "(No text content)"}
          </p>
        </div>
      </div>
    </div>
  );
}
