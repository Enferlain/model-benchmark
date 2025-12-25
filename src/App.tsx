import React, { useState, useCallback, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import { GalleryProvider } from "./context/GalleryContext";
import { MainLayout } from "./layouts/MainLayout";
import Dashboard from "./pages/Dashboard";
import Gallery from "./pages/Gallery";
import PromptEditor from "./pages/PromptEditor";
import Compare from "./pages/Compare";
import Arena from "./pages/Arena";
import { ModelData } from "./types";
import { fetchModels as apiFetchModels } from "./services/api";

export default function App() {
  const [models, setModels] = useState<ModelData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchModels = useCallback(async () => {
    try {
      const data = await apiFetchModels();
      setModels(data);
    } catch (error) {
      console.error("Failed to fetch models:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return (
    <ThemeProvider>
      <GalleryProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<MainLayout isLoading={isLoading} modelCount={models.length} />}>
              <Route index element={
                <Dashboard
                  models={models}
                  setModels={setModels}
                  isLoading={isLoading}
                  fetchModels={fetchModels}
                />
              } />
              <Route path="gallery" element={<Gallery />} />
              <Route path="prompts" element={<PromptEditor />} />
              <Route path="compare" element={<Compare />} />
              <Route path="arena" element={<Arena />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </GalleryProvider>
    </ThemeProvider>
  );
}
