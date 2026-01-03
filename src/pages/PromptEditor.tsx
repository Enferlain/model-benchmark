import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Image as ImageIcon } from 'lucide-react';
import { fetchPrompts, createPrompt, updatePromptText, deletePrompt, shufflePrompts, setAllPromptsEnabled } from '../services/api';
import { PromptData } from '../types';
import { API_BASE } from '../services/api'; 
import { arrayMove } from '@dnd-kit/sortable';
import { DragEndEvent } from '@dnd-kit/core';

import { PromptList } from '../components/prompts/PromptList';
import { PromptDetailEditor } from '../components/prompts/PromptDetailEditor';
import { CreatePromptModal } from '../components/prompts/CreatePromptModal';

export default function PromptEditor() {
  const [prompts, setPrompts] = useState<PromptData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Editor State
  const [editText, setEditText] = useState('');
  const [editAlias, setEditAlias] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  
  // Creation State
  const [isCreating, setIsCreating] = useState(false);
  const [newPromptText, setNewPromptText] = useState('');
  const [newPromptImage, setNewPromptImage] = useState<File | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  
  // Global Drag and Drop
  useEffect(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer?.types.includes('Files')) {
        setIsDraggingOver(true);
      }
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const relatedTarget = e.relatedTarget as (EventTarget | null);
      if (!relatedTarget || !document.contains(relatedTarget as Node)) {
        setIsDraggingOver(false);
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.dataTransfer?.types.includes('Files')) {
        setIsDraggingOver(true);
      }
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDraggingOver(false);

      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith('image/')) {
           setNewPromptImage(file);
           setIsCreating(true);
        }
      }
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('drop', handleDrop);

    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('drop', handleDrop);
    };
  }, []);

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
      setEditAlias(selectedPrompt.alias || '');
      setIsDirty(false);
    }
  }, [selectedPrompt]);
  
  // Handlers
  const handleSave = async () => {
    if (!selectedPrompt) return;
    setIsSaving(true);
    try {
      await updatePromptText(selectedPrompt.filename, {
          text: editText,
          alias: editAlias
      });
      setIsDirty(false);
      setPrompts(prev => prev.map(p => 
        p.id === selectedId ? { ...p, text: editText, alias: editAlias } : p
      ));
    } catch (err) {
      alert("Failed to save prompt");
    } finally {
      setIsSaving(false);
    }
  };

  const handleShuffle = async () => {
    setIsLoading(true);
    try {
        await shufflePrompts();
        await loadPrompts();
    } catch (err) {
        alert("Failed to shuffle prompts");
        setIsLoading(false);
    }
  };

  const handleEnableAll = async (enabled: boolean) => {
    setIsLoading(true);
    try {
        await setAllPromptsEnabled(enabled);
        setPrompts(prev => prev.map(p => ({ ...p, enabled })));
    } catch (err) {
        alert("Failed to update prompts");
        await loadPrompts();
    } finally {
        setIsLoading(false);
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
        loadPrompts();
      }
    } catch (err) {
      alert("Failed to create prompt");
    } finally {
      setIsSaving(false);
    }
  };
  
  const handleToggle = useCallback(async (e: React.MouseEvent, prompt: PromptData) => {
    e.stopPropagation();
    const previousEnabled = prompt.enabled;
    try {
        const newStatus = !prompt.enabled;
        setPrompts(prev => prev.map(p => 
            p.id === prompt.id ? { ...p, enabled: newStatus } : p
        ));
        await updatePromptText(prompt.filename, { enabled: newStatus });
    } catch (err) {
        // Rollback on failure
        setPrompts(prev => prev.map(p =>
            p.id === prompt.id ? { ...p, enabled: previousEnabled } : p
        ));
        alert("Failed to toggle prompt");
    }
  }, []);

  const selectedIdRef = React.useRef(selectedId);
  useEffect(() => { selectedIdRef.current = selectedId; }, [selectedId]);

  const handleDelete = useCallback(async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this prompt?")) return;
    
    const promptToDelete = prompts.find(x => x.id === id);
    if (!promptToDelete) return;

    // Optimistic update
    setPrompts(prev => prev.filter(x => x.id !== id));
    if (selectedIdRef.current === id) setSelectedId(null);

    try {
      await deletePrompt(promptToDelete.filename);
    } catch (err) {
      // Rollback on failure (simplified, append at end)
      setPrompts(prev => [...prev, promptToDelete]);
      alert("Failed to delete prompt");
    }
  }, [prompts]);

  const handleDragEnd = async (event: DragEndEvent) => {
    const {active, over} = event;
    
    if (over && active.id !== over.id) {
      setPrompts((items) => {
        const oldIndex = items.findIndex(i => i.id === active.id);
        const newIndex = items.findIndex(i => i.id === over.id);
        
        const newItems = arrayMove(items, oldIndex, newIndex);
        
        const order = newItems.map(p => p.filename);
        fetch(`${API_BASE}/prompts/reorder`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({order})
        }).catch(err => {
            console.error("Failed to save order", err);
        });
        
        return newItems;
      });
    }
  };

  return (
    <div className="max-w-[1800px] mx-auto h-[calc(100vh-100px)] pt-6 px-6 flex gap-6">
      
      <PromptList
         prompts={prompts}
         isLoading={isLoading}
         searchQuery={searchQuery}
         selectedId={selectedId}
         onSearchChange={setSearchQuery}
         onSelect={setSelectedId}
         onCreate={() => setIsCreating(true)}
         onShuffle={handleShuffle}
         onEnableAll={handleEnableAll}
         onToggle={handleToggle}
         onDelete={handleDelete}
         onDragEnd={handleDragEnd}
      />
      
      <PromptDetailEditor
         prompt={selectedPrompt}
         editText={editText}
         editAlias={editAlias}
         isSaving={isSaving}
         isDirty={isDirty}
         onTextChange={(val) => { setEditText(val); setIsDirty(true); }}
         onAliasChange={(val) => { setEditAlias(val); setIsDirty(true); }}
         onSave={handleSave}
      />
        
      {/* Drag Overlay */}
      {isDraggingOver && (
        <div className="fixed inset-0 z-[60] bg-blue-500/20 backdrop-blur-sm flex items-center justify-center pointer-events-none border-4 border-blue-500 border-dashed m-4 rounded-3xl">
            <div className="bg-white/90 dark:bg-slate-900/90 p-8 rounded-2xl shadow-2xl text-center">
                <ImageIcon size={64} className="mx-auto text-blue-500 mb-4" />
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Drop Image to Interrogate</h3>
                <p className="text-slate-500 dark:text-slate-400 mt-2">Release to create a new prompt from this image</p>
            </div>
        </div>
      )}

      <CreatePromptModal
         isOpen={isCreating}
         onClose={() => setIsCreating(false)}
         onCreate={handleCreate}
         newPromptText={newPromptText}
         setNewPromptText={setNewPromptText}
         newPromptImage={newPromptImage}
         setNewPromptImage={setNewPromptImage}
      />
    </div>
  );
}
