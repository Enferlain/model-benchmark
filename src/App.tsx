import { useCallback, useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { GalleryProvider } from "./context/GalleryContext";
import { ThemeProvider } from "./context/ThemeContext";
import { MainLayout } from "./layouts/MainLayout";
import Analytics from "./pages/Analytics";
import Arena from "./pages/Arena";
import Compare from "./pages/Compare";
import Dashboard from "./pages/Dashboard";
import Database from "./pages/Database";
import Gallery from "./pages/Gallery";
import PromptEditor from "./pages/PromptEditor";
import { fetchModels as apiFetchModels } from "./services/api";
import type { ModelData } from "./types";

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
						<Route
							path="/"
							element={
								<MainLayout isLoading={isLoading} modelCount={models.length} />
							}
						>
							<Route
								index
								element={
									<Dashboard
										models={models}
										setModels={setModels}
										isLoading={isLoading}
										fetchModels={fetchModels}
									/>
								}
							/>
							<Route path="gallery" element={<Gallery />} />
							<Route path="prompts" element={<PromptEditor />} />
							<Route path="compare" element={<Compare />} />
							<Route
								path="analytics"
								element={
									<Analytics
										models={models}
										setModels={setModels}
										fetchModels={fetchModels}
									/>
								}
							/>
							<Route path="arena" element={<Arena />} />
							<Route path="database" element={<Database />} />
						</Route>
					</Routes>
				</BrowserRouter>
			</GalleryProvider>
		</ThemeProvider>
	);
}
