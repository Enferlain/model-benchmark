import type React from "react";
import {
	createContext,
	type ReactNode,
	useCallback,
	useContext,
	useEffect,
	useState,
} from "react";
import {
	fetchModels as apiFetchModels,
	fetchBenchmarkRuns,
	fetchPrompts,
} from "../services/api";
import type {
	BenchmarkRun,
	ModelData,
	ModelOutput,
	PromptData,
} from "../types";

interface DataContextType {
	// Models
	models: ModelData[];
	isLoadingModels: boolean;
	refreshModels: () => Promise<void>;

	// Prompts
	allPrompts: PromptData[];
	isLoadingPrompts: boolean;
	refreshPrompts: () => Promise<void>;

	// Benchmark Runs
	runs: BenchmarkRun[];
	isLoadingRuns: boolean;
	refreshRuns: () => Promise<void>;

	// Errors
	errors: {
		models: string | null;
		prompts: string | null;
		runs: string | null;
	};

	// Combined Refresher
	refreshAll: () => Promise<void>;

	// Gallery/Compare Cache & State
	outputCache: Record<string, ModelOutput[]>;
	setOutputCache: React.Dispatch<
		React.SetStateAction<Record<string, ModelOutput[]>>
	>;
	selectedModel: string;
	setSelectedModel: (id: string) => void;
	selectedPrompt: string;
	setSelectedPrompt: (prompt: string) => void;
	selectedSeed: string;
	setSelectedSeed: (seed: string) => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export function DataProvider({ children }: { children: ReactNode }) {
	// Models State
	const [models, setModels] = useState<ModelData[]>([]);
	const [isLoadingModels, setIsLoadingModels] = useState(true);
	const [modelError, setModelError] = useState<string | null>(null);

	// Prompts State
	const [allPrompts, setAllPrompts] = useState<PromptData[]>([]);
	const [isLoadingPrompts, setIsLoadingPrompts] = useState(true);
	const [promptError, setPromptError] = useState<string | null>(null);

	// Runs State
	const [runs, setRuns] = useState<BenchmarkRun[]>([]);
	const [isLoadingRuns, setIsLoadingRuns] = useState(true);
	const [runError, setRunError] = useState<string | null>(null);

	// Gallery/Compare State
	const [outputCache, setOutputCache] = useState<Record<string, ModelOutput[]>>(
		{},
	);
	const [selectedModel, setSelectedModel] = useState<string>(
		() => localStorage.getItem("gallery_selectedModel") || "",
	);
	const [selectedPrompt, setSelectedPrompt] = useState<string>(
		() => localStorage.getItem("gallery_selectedPrompt") || "All",
	);
	const [selectedSeed, setSelectedSeed] = useState<string>(
		() => localStorage.getItem("gallery_selectedSeed") || "All",
	);

	// Fetching Logic
	const refreshModels = useCallback(async () => {
		setIsLoadingModels(true);
		setModelError(null);
		try {
			const data = await apiFetchModels();
			setModels(data);
		} catch (error) {
			console.error("Failed to fetch models:", error);
			setModelError(error instanceof Error ? error.message : String(error));
		} finally {
			setIsLoadingModels(false);
		}
	}, []);

	const refreshPrompts = useCallback(async () => {
		setIsLoadingPrompts(true);
		setPromptError(null);
		try {
			const data = await fetchPrompts();
			setAllPrompts(Array.isArray(data) ? data : []);
		} catch (error) {
			console.error("Failed to fetch prompts:", error);
			setPromptError(error instanceof Error ? error.message : String(error));
			setAllPrompts([]);
		} finally {
			setIsLoadingPrompts(false);
		}
	}, []);

	const refreshRuns = useCallback(async () => {
		setIsLoadingRuns(true);
		setRunError(null);
		try {
			const data = await fetchBenchmarkRuns();
			setRuns(data);
		} catch (error) {
			console.error("Failed to fetch benchmark runs:", error);
			setRunError(error instanceof Error ? error.message : String(error));
		} finally {
			setIsLoadingRuns(false);
		}
	}, []);

	const refreshAll = useCallback(async () => {
		// Start all loaders simultaneously
		setIsLoadingModels(true);
		setIsLoadingPrompts(true);
		setIsLoadingRuns(true);

		await Promise.allSettled([
			refreshModels(),
			refreshPrompts(),
			refreshRuns(),
		]);
	}, [refreshModels, refreshPrompts, refreshRuns]);

	// Initial Load
	useEffect(() => {
		refreshAll();
	}, [refreshAll]);

	// Persistence Effects (Gallery States)
	useEffect(() => {
		localStorage.setItem("gallery_selectedModel", selectedModel);
	}, [selectedModel]);

	useEffect(() => {
		localStorage.setItem("gallery_selectedPrompt", selectedPrompt);
	}, [selectedPrompt]);

	useEffect(() => {
		localStorage.setItem("gallery_selectedSeed", selectedSeed);
	}, [selectedSeed]);

	return (
		<DataContext.Provider
			value={{
				models,
				isLoadingModels,
				refreshModels,
				allPrompts,
				isLoadingPrompts,
				refreshPrompts,
				runs,
				isLoadingRuns,
				refreshRuns,
				errors: {
					models: modelError,
					prompts: promptError,
					runs: runError,
				},
				refreshAll,
				outputCache,
				setOutputCache,
				selectedModel,
				setSelectedModel,
				selectedPrompt,
				setSelectedPrompt,
				selectedSeed,
				setSelectedSeed,
			}}
		>
			{children}
		</DataContext.Provider>
	);
}

export function useData() {
	const context = useContext(DataContext);
	if (context === undefined) {
		throw new Error("useData must be used within a DataProvider");
	}
	return context;
}
