import { useCallback, useEffect, useMemo, useState } from "react";
import { AddModelCard } from "../components/dashboard/AddModelCard";
import { DatasetSelector } from "../components/dashboard/DatasetSelector";
import { MetricInfoCard } from "../components/dashboard/MetricInfoCard";
import { MismatchModals } from "../components/dashboard/MismatchModals";
import { TransferListSection } from "../components/dashboard/ModelSelection/TransferListSection";
import { ViewSettingsCard } from "../components/dashboard/ViewSettingsCard";
import {
	DEFAULT_SCAN_OPTIONS,
	ScanSettingsPanel,
} from "../components/ScanSettingsPanel";
import { ScatterPlot } from "../components/ScatterPlot";
import type { Preset } from "../components/TransferList/PresetMenu";
import { METRIC_OPTIONS } from "../constants";
import { useData } from "../context/DataContext";
import { useTheme } from "../context/ThemeContext";
import { useDashboardStatus } from "../hooks/useDashboardStatus";
import {
	analyzeImages,
	archiveModel,
	type CoverageCheckResult,
	checkCoverage,
	checkParams,
	generateImages,
	type ParamCheckResult,
} from "../services/api";
import type { MetricKey, ModelData } from "../types";

export default function Dashboard() {
	const { models, refreshModels, runs: benchmarkRuns } = useData();
	const { isDarkMode } = useTheme();
	const [urlInput, setUrlInput] = useState("");

	const {
		isDownloading,
		downloadError,
		downloadProgress,
		setDownloadError,
		handleDownloadModel,
		handleCancel,
		generationStatus,
		setGenerationStatus,
		setIsScanning,
	} = useDashboardStatus({ fetchModels: refreshModels });

	// Model Selection for Benchmark
	const [selectedModelIds, setSelectedModelIds] = useState<string[]>(() => {
		try {
			const stored = localStorage.getItem("dashboard_selectedModelIds");
			return stored ? JSON.parse(stored) : [];
		} catch {
			return [];
		}
	});
	const [xMetricKey, setXMetricKey] = useState<MetricKey>(() => {
		return (
			(localStorage.getItem("dashboard_xMetricKey") as MetricKey) || "accuracy"
		);
	});
	const [yMetricKey, setYMetricKey] = useState<MetricKey>(() => {
		return (
			(localStorage.getItem("dashboard_yMetricKey") as MetricKey) || "diversity"
		);
	});

	const [paramMismatch, setParamMismatch] = useState<ParamCheckResult | null>(
		null,
	);
	const [showMismatchModal, setShowMismatchModal] = useState(false);
	const [coverageMismatch, setCoverageMismatch] =
		useState<CoverageCheckResult | null>(null);
	const [showCoverageModal, setShowCoverageModal] = useState(false);

	const xMetric =
		METRIC_OPTIONS.find((m) => m.value === xMetricKey) || METRIC_OPTIONS[0];
	const yMetric =
		METRIC_OPTIONS.find((m) => m.value === yMetricKey) || METRIC_OPTIONS[1];

	const [scanOptions, setScanOptions] = useState(() => {
		try {
			const stored = localStorage.getItem("dashboard_scanOptions");
			return stored ? JSON.parse(stored) : DEFAULT_SCAN_OPTIONS;
		} catch {
			return DEFAULT_SCAN_OPTIONS;
		}
	});
	const [activeSource, setActiveSource] = useState<{
		type: "all" | "queue" | "run" | "preset";
		id?: string | number;
	}>(() => {
		try {
			const stored = localStorage.getItem("dashboard_activeSource");
			return stored ? JSON.parse(stored) : { type: "all" };
		} catch {
			return { type: "all" };
		}
	});
	const [chartSearchQuery, setChartSearchQuery] = useState(() => {
		return localStorage.getItem("dashboard_chartSearchQuery") || "";
	});

	// Persistence Effects
	useEffect(() => {
		localStorage.setItem(
			"dashboard_selectedModelIds",
			JSON.stringify(selectedModelIds),
		);
	}, [selectedModelIds]);

	useEffect(() => {
		localStorage.setItem("dashboard_xMetricKey", xMetricKey);
	}, [xMetricKey]);

	useEffect(() => {
		localStorage.setItem("dashboard_yMetricKey", yMetricKey);
	}, [yMetricKey]);

	useEffect(() => {
		localStorage.setItem("dashboard_scanOptions", JSON.stringify(scanOptions));
	}, [scanOptions]);

	useEffect(() => {
		localStorage.setItem(
			"dashboard_activeSource",
			JSON.stringify(activeSource),
		);
	}, [activeSource]);

	useEffect(() => {
		localStorage.setItem("dashboard_chartSearchQuery", chartSearchQuery);
	}, [chartSearchQuery]);

	// Load presets from local storage
	const [presets, setPresets] = useState<Preset[]>([]);
	useEffect(() => {
		try {
			const stored = localStorage.getItem("model_benchmark_presets");
			if (stored) setPresets(JSON.parse(stored));
		} catch (e) {
			console.error("Failed to load presets", e);
		}
	}, []);

	// Derive data for chart
	const chartData = useMemo(() => {
		let baseData: ModelData[] = [];

		if (activeSource.type === "all") {
			baseData = models;
		} else if (activeSource.type === "queue") {
			baseData = models.filter((m) => selectedModelIds.includes(m.id));
		} else if (activeSource.type === "run") {
			const run = benchmarkRuns.find((r) => r.id === activeSource.id);
			if (run) {
				// Convert RunResults to ModelData
				baseData = run.results.map((res) => {
					const baseModel = models.find((m) => m.id === res.model_hash);
					return {
						...(baseModel || {
							id: res.model_hash,
							name: res.model_name,
							source: "Unknown",
							url: "",
							accuracy: 0,
							diversity: 0,
							rating: 0,
						}),
						metrics: res.metrics,
						accuracy: (res.metrics as Record<string, number>).accuracy || 0,
						diversity: (res.metrics as Record<string, number>).diversity || 0,
					} as ModelData;
				});
			}
		} else if (activeSource.type === "preset") {
			const preset = presets.find((p) => p.id === activeSource.id);
			if (preset) {
				baseData = models.filter((m) => preset.modelIds.includes(m.id));
			}
		}

		// Filter by search query
		if (chartSearchQuery.trim()) {
			const query = chartSearchQuery.toLowerCase();
			const filtered = baseData.filter(
				(m) =>
					m.name.toLowerCase().includes(query) ||
					m.id.toLowerCase().includes(query) ||
					m.model_type?.toLowerCase().includes(query),
			);

			// Heuristic: If we matched NO models, but the query matches a Run or Preset name,
			// don't filter the chart yet (the user is likely searching for a source).
			const matchesSource =
				benchmarkRuns.some((r) =>
					`run #${r.id}`.toLowerCase().includes(query),
				) || presets.some((p) => p.name.toLowerCase().includes(query));

			if (filtered.length === 0 && matchesSource) {
				return baseData;
			}

			return filtered;
		}

		return baseData;
	}, [
		models,
		selectedModelIds,
		activeSource,
		benchmarkRuns,
		presets,
		chartSearchQuery,
	]);

	const doGenerate = useCallback(async () => {
		setIsScanning(true);
		setGenerationStatus({
			is_running: true,
			current_model: null,
			progress: { current: 0, total: 0 },
		});
		try {
			await generateImages({
				...scanOptions,
				selected_model_ids:
					selectedModelIds.length > 0 ? selectedModelIds : undefined,
			});
		} catch (error) {
			console.error("Generate error:", error);
		} finally {
			setIsScanning(false);
			setGenerationStatus((prev) => ({ ...prev, is_running: false }));
		}
	}, [scanOptions, selectedModelIds, setIsScanning, setGenerationStatus]);

	const handleGenerate = useCallback(async () => {
		try {
			const checkResult = await checkParams(scanOptions);
			if (!checkResult.matches && checkResult.mismatched_models.length > 0) {
				setParamMismatch(checkResult);
				setShowMismatchModal(true);
				return;
			}
		} catch (error) {
			console.error("Check params error:", error);
		}
		await doGenerate();
	}, [scanOptions, doGenerate]);

	const handleMatchExisting = useCallback(() => {
		const existing = paramMismatch?.existing_params;
		if (existing) {
			setScanOptions((prev) => ({
				...prev,
				steps: existing.steps,
				guidance_scale: existing.cfg,
				sampler: existing.sampler,
				width: existing.width,
				height: existing.height,
			}));
		}
		setShowMismatchModal(false);
		setParamMismatch(null);
	}, [paramMismatch]);

	const handleArchiveAndGenerate = useCallback(async () => {
		if (!paramMismatch) return;

		const archiveErrors: string[] = [];
		for (const model of paramMismatch.mismatched_models) {
			try {
				await archiveModel(model.name);
			} catch (error) {
				console.error(`Failed to archive ${model.name}:`, error);
				archiveErrors.push(model.name);
			}
		}

		if (archiveErrors.length > 0) {
			alert(
				`Warning: Failed to archive images for: ${archiveErrors.join(", ")}. Generation will proceed anyway.`,
			);
		}

		setShowMismatchModal(false);
		setParamMismatch(null);
		await doGenerate();
	}, [paramMismatch, doGenerate]);

	const doAnalyze = useCallback(
		async (commonOnly: boolean) => {
			setIsScanning(true);
			try {
				await analyzeImages({ ...scanOptions, common_only: commonOnly });
				await refreshModels();
			} catch (error) {
				console.error("Analyze error:", error);
			} finally {
				setIsScanning(false);
			}
		},
		[scanOptions, refreshModels, setIsScanning],
	);

	const handleAnalyze = useCallback(async () => {
		try {
			const coverageResult = await checkCoverage();
			if (
				!coverageResult.all_match ||
				coverageResult.model_coverage.some((m) => m.missing_count > 0)
			) {
				setCoverageMismatch(coverageResult);
				setShowCoverageModal(true);
				return;
			}
		} catch (error) {
			console.error("Coverage check error:", error);
		}
		await doAnalyze(false);
	}, [doAnalyze]);

	const handleAnalyzeCommonOnly = useCallback(async () => {
		setShowCoverageModal(false);
		setCoverageMismatch(null);
		await doAnalyze(true);
	}, [doAnalyze]);

	const handleGenerateMissing = useCallback(async () => {
		setShowCoverageModal(false);
		setCoverageMismatch(null);
		setIsScanning(true);
		setGenerationStatus({
			is_running: true,
			current_model: null,
			progress: { current: 0, total: 0 },
		});
		try {
			await generateImages({ ...scanOptions, equalize_counts: true });
		} catch (error) {
			console.error("Generate error:", error);
		} finally {
			setIsScanning(false);
			setGenerationStatus((prev) => ({ ...prev, is_running: false }));
		}
		await doAnalyze(false);
	}, [scanOptions, doAnalyze, setIsScanning, setGenerationStatus]);

	const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

	return (
		<>
			<MismatchModals
				showMismatchModal={showMismatchModal}
				setShowMismatchModal={setShowMismatchModal}
				paramMismatch={paramMismatch}
				setParamMismatch={setParamMismatch}
				handleMatchExisting={handleMatchExisting}
				handleArchiveAndGenerate={handleArchiveAndGenerate}
				showCoverageModal={showCoverageModal}
				setShowCoverageModal={setShowCoverageModal}
				coverageMismatch={coverageMismatch}
				setCoverageMismatch={setCoverageMismatch}
				handleAnalyzeCommonOnly={handleAnalyzeCommonOnly}
				handleGenerateMissing={handleGenerateMissing}
			/>

			<div className="max-w-[1800px] mx-auto px-6 py-8">
				<div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
					{/* Sidebar / Controls */}
					<div className="lg:col-span-3 space-y-6">
						<AddModelCard
							urlInput={urlInput}
							setUrlInput={setUrlInput}
							isDownloading={isDownloading}
							downloadProgress={downloadProgress}
							downloadError={downloadError}
							setDownloadError={setDownloadError}
							handleDownloadModel={handleDownloadModel}
							fetchModels={refreshModels}
						/>

						<ScanSettingsPanel
							options={scanOptions}
							onChange={setScanOptions}
							onGenerate={handleGenerate}
							onAnalyze={handleAnalyze}
							onCancel={handleCancel}
							status={generationStatus}
							onRefreshModels={refreshModels}
						/>

						<ViewSettingsCard
							xMetricKey={xMetricKey}
							setXMetricKey={setXMetricKey}
							yMetricKey={yMetricKey}
							setYMetricKey={setYMetricKey}
						/>

						<MetricInfoCard xMetric={xMetric} yMetric={yMetric} />
					</div>

					{/* Main Content Area */}
					<div className="lg:col-span-9 space-y-8">
						<div className="flex flex-col gap-8 min-h-[600px]">
							<div className="bg-white/90 dark:bg-slate-800/80 rounded-[22px] backdrop-blur-md min-h-[500px] flex flex-col border border-slate-200/50 dark:border-white/5 shadow-xl transition-shadow hover:shadow-2xl overflow-hidden">
								<div className="flex-1 w-full min-h-0 relative">
									<ScatterPlot
										data={chartData}
										xMetric={xMetric}
										yMetric={yMetric}
										onSelect={setSelectedModelId}
										selectedId={selectedModelId}
										isDarkMode={isDarkMode}
										headerExtra={
											<DatasetSelector
												activeSource={activeSource}
												onSelect={setActiveSource}
												searchQuery={chartSearchQuery}
												onSearchChange={setChartSearchQuery}
												runs={benchmarkRuns}
												presets={presets}
												isDarkMode={isDarkMode}
											/>
										}
									/>
								</div>
							</div>

							<TransferListSection
								models={models}
								selectedModelIds={selectedModelIds}
								setSelectedModelIds={setSelectedModelIds}
							/>
						</div>
					</div>
				</div>
			</div>
		</>
	);
}
