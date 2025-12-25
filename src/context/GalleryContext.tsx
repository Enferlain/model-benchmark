import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ModelOutput, PromptData } from '../types';

import { fetchPrompts } from '../services/api';

interface GalleryContextType {
  // Cache: modelId -> outputs
  outputCache: Record<string, ModelOutput[]>;
  setOutputCache: React.Dispatch<React.SetStateAction<Record<string, ModelOutput[]>>>;
  
  // Prompts Data
  allPrompts: PromptData[];
  
  // Selection State
  selectedModel: string;
  setSelectedModel: (id: string) => void;
  
  // Filter State
  selectedPrompt: string;
  setSelectedPrompt: (prompt: string) => void;
  selectedSeed: string;
  setSelectedSeed: (seed: string) => void;
}

const GalleryContext = createContext<GalleryContextType | undefined>(undefined);

export function GalleryProvider({ children }: { children: ReactNode }) {
  // Initialize from localStorage or defaults
  const [outputCache, setOutputCache] = useState<Record<string, ModelOutput[]>>({});
  const [allPrompts, setAllPrompts] = useState<PromptData[]>([]);
  
  const [selectedModel, setSelectedModel] = useState<string>(() => 
    localStorage.getItem('gallery_selectedModel') || ""
  );
  const [selectedPrompt, setSelectedPrompt] = useState<string>(() => 
    localStorage.getItem('gallery_selectedPrompt') || "All"
  );
  const [selectedSeed, setSelectedSeed] = useState<string>(() => 
    localStorage.getItem('gallery_selectedSeed') || "All"
  );

  // Persistence effects
  useEffect(() => {
    localStorage.setItem('gallery_selectedModel', selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    localStorage.setItem('gallery_selectedPrompt', selectedPrompt);
  }, [selectedPrompt]);

  useEffect(() => {
    localStorage.setItem('gallery_selectedSeed', selectedSeed);
  }, [selectedSeed]);

  useEffect(() => {
    // Fetch/cache prompts on startup
    fetchPrompts().then(data => {
        if (Array.isArray(data)) {
            setAllPrompts(data);
        } else {
            console.error("Prompts is not array", data);
            setAllPrompts([]);
        }
    }).catch(err => console.error("Failed to fetch prompts", err));
  }, []);

  return (
    <GalleryContext.Provider value={{
      outputCache,
      setOutputCache,
      allPrompts,
      selectedModel,
      setSelectedModel,
      selectedPrompt,
      setSelectedPrompt,
      selectedSeed,
      setSelectedSeed
    }}>
      {children}
    </GalleryContext.Provider>
  );
}

export function useGalleryContext() {
  const context = useContext(GalleryContext);
  if (context === undefined) {
    throw new Error('useGalleryContext must be used within a GalleryProvider');
  }
  return context;
}
