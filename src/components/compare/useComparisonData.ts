import { useState, useEffect, useMemo, useRef } from 'react';
import { fetchModelOutputs } from '../../services/api';
import { ModelData, ModelOutput } from '../../types';

export function useComparisonData(models: ModelData[], selectedModelIds: string[]) {
  const [outputsMap, setOutputsMap] = useState<Record<string, ModelOutput[]>>({});
  const [loadingMap, setLoadingMap] = useState<Record<string, boolean>>({});

  // Track in-flight requests to avoid stale closures
  const inFlightIds = useRef<Set<string>>(new Set());

  // Fetch outputs for selected models
  useEffect(() => {
    selectedModelIds.forEach(id => {
      // If we already have data or are currently fetching, skip
      if (outputsMap[id] || inFlightIds.current.has(id)) return;

      inFlightIds.current.add(id);
      setLoadingMap(prev => ({ ...prev, [id]: true }));

      fetchModelOutputs(id)
        .then(data => {
          setOutputsMap(prev => ({ ...prev, [id]: data }));
        })
        .catch(err => console.error(`Failed to load outputs for ${id}`, err))
        .finally(() => {
          setLoadingMap(prev => ({ ...prev, [id]: false }));
          inFlightIds.current.delete(id);
        });
    });
  }, [selectedModelIds, outputsMap]);

  // Calculate Intersection
  const commonData = useMemo(() => {
    if (selectedModelIds.length === 0) return { prompts: [], seeds: [] };

    // Get output lists for selected models
    const outputLists = selectedModelIds
      .map(id => outputsMap[id])
      .filter(list => list !== undefined);

    if (outputLists.length !== selectedModelIds.length) {
      // Still loading some data
      return { prompts: [], seeds: [] };
    }

    // Find intersection
    // 1. Prompts
    const promptsSets = outputLists.map(list => new Set(list.map(o => o.prompt)));
    // Start with first set and filter
    let intersectionPrompts = Array.from(promptsSets[0] || []);
    for (let i = 1; i < promptsSets.length; i++) {
        intersectionPrompts = intersectionPrompts.filter(p => promptsSets[i].has(p));
    }

    // 2. Seeds
    const seedsSets = outputLists.map(list => new Set(list.map(o => o.seed)));
    let intersectionSeeds = Array.from(seedsSets[0] || []);
    for (let i = 1; i < seedsSets.length; i++) {
        intersectionSeeds = intersectionSeeds.filter(s => seedsSets[i].has(s));
    }

    return {
        prompts: intersectionPrompts,
        seeds: intersectionSeeds.sort((a, b) => a - b)
    };
  }, [selectedModelIds, outputsMap]);

  const getImagesForSelection = (prompt: string, seed: string) => {
      return selectedModelIds.map(id => {
          const list = outputsMap[id] || [];
          return list.find(o => o.prompt === prompt && o.seed.toString() === seed);
      });
  };

  return {
    outputsMap,
    loadingMap,
    commonPrompts: commonData.prompts,
    commonSeeds: commonData.seeds,
    getImagesForSelection
  };
}
