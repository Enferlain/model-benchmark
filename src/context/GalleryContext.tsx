import React, { createContext, useContext, useState, ReactNode } from 'react';
import { ModelOutput } from '../types';

interface GalleryContextType {
  // Cache: modelId -> outputs
  outputCache: Record<string, ModelOutput[]>;
  setOutputCache: React.Dispatch<React.SetStateAction<Record<string, ModelOutput[]>>>;
  
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
  const [outputCache, setOutputCache] = useState<Record<string, ModelOutput[]>>({});
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedPrompt, setSelectedPrompt] = useState<string>("All");
  const [selectedSeed, setSelectedSeed] = useState<string>("All");

  return (
    <GalleryContext.Provider value={{
      outputCache,
      setOutputCache,
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
