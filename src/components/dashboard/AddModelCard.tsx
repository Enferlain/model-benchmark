import {
	Download,
	FileText,
	FolderOpen,
	Info,
	Loader2,
	Plus,
} from "lucide-react";
import { useState } from "react";
import type { DownloadProgress } from "../../hooks/useDashboardStatus";
import { browseSystemPath, registerModelPaths } from "../../services/api";

interface AddModelCardProps {
	urlInput: string;
	setUrlInput: (val: string) => void;
	isDownloading: boolean;
	downloadProgress: DownloadProgress;
	downloadError: string | null;
	setDownloadError: (val: string | null) => void;
	handleDownloadModel: (url: string) => Promise<void>;
	fetchModels: () => Promise<void>;
}

export function AddModelCard({
	urlInput,
	setUrlInput,
	isDownloading,
	downloadProgress,
	downloadError,
	setDownloadError,
	handleDownloadModel,
	fetchModels,
}: AddModelCardProps) {
	const [isRegistering, setIsRegistering] = useState(false);
	const [registerResult, setRegisterResult] = useState<{
		status: string;
		message: string;
	} | null>(null);

	const onDownload = () => handleDownloadModel(urlInput);

	const handleImport = async (type: "folder" | "file") => {
		setIsRegistering(true);
		try {
			const result = await browseSystemPath(type);
			if (result.paths && result.paths.length > 0) {
				const response = await registerModelPaths(result.paths);
				const stats = response.stats || {};
				let msg = `${response.results.length} Files processed.`;
				if (stats.added > 0)
					msg = `${stats.added} New ${type === "folder" ? "Model(s)" : "File(s)"} Imported!`;
				else if (stats.updated > 0)
					msg = `${stats.updated} ${type === "folder" ? "Model(s)" : "File(s)"} Updated.`;
				else if (stats.unchanged > 0)
					msg = `${stats.unchanged} ${type === "folder" ? "Model(s)" : "File(s)"} Verified (No changes).`;

				setRegisterResult({ status: "success", message: msg });
				await fetchModels();
				setTimeout(() => setRegisterResult(null), 3000);
			}
		} catch (e: unknown) {
			console.error("Import error:", e);
			const error = e as Error;
			setDownloadError(error.message || "Failed to import");
		} finally {
			setIsRegistering(false);
		}
	};

	return (
		<div className="p-6 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-white/60 dark:border-white/5 bg-white/90 dark:bg-slate-800/80 backdrop-blur-md transition-shadow hover:shadow-2xl">
			<h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-4 flex items-center gap-2">
				<Plus size={14} /> Add Model
			</h2>
			<div className="space-y-4">
				{/* URL Download Section */}
				<div>
					<label
						htmlFor="model-url"
						className="block text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 ml-1 opacity-80"
					>
						DOWNLOAD FROM URL
					</label>
					<input
						id="model-url"
						type="text"
						value={urlInput}
						onChange={(e) => {
							setUrlInput(e.target.value);
							if (downloadError) setDownloadError(null);
						}}
						placeholder="https://civitai.com/models/..."
						className={`w-full px-4 py-3 border ${
							downloadError
								? "border-red-500/50 bg-red-500/5"
								: "border-slate-200/60 dark:border-white/5 bg-white/50 dark:bg-black/20"
						} rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 dark:focus:ring-blue-400/20 transition-all placeholder:text-slate-400/70 dark:placeholder:text-slate-600 text-slate-800 dark:text-slate-200 backdrop-blur-sm`}
						onKeyDown={(e) => e.key === "Enter" && onDownload()}
					/>
					{downloadError && (
						<p className="text-red-500 text-[10px] mt-1 ml-1 animate-pulse">
							{downloadError}
						</p>
					)}
				</div>
				{isDownloading && downloadProgress.total > 0 && (
					<div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5 mb-1 overflow-hidden">
						<div
							className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
							style={{
								width: `${Math.min(
									100,
									(downloadProgress.current / downloadProgress.total) * 100,
								)}%`,
							}}
						/>
					</div>
				)}
				{isDownloading && (
					<p className="text-[10px] text-slate-400 text-center mb-2">
						{downloadProgress.total > 0
							? `${(downloadProgress.current / 1024 / 1024).toFixed(1)} / ${(
									downloadProgress.total / 1024 / 1024
								).toFixed(1)} MB`
							: "Starting download..."}
					</p>
				)}
				<button
					type="button"
					onClick={onDownload}
					disabled={!urlInput || isDownloading}
					className="w-full bg-blue-600/90 hover:bg-blue-600 dark:bg-blue-500/80 dark:hover:bg-blue-500 text-white disabled:bg-slate-200 dark:disabled:bg-slate-800/50 disabled:text-slate-400 disabled:cursor-not-allowed font-medium py-3 px-4 rounded-xl transition-all duration-300 text-sm flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20 dark:shadow-blue-900/20 backdrop-blur-sm"
				>
					{isDownloading ? (
						<>
							<Loader2 size={16} className="animate-spin" /> Downloading...
						</>
					) : (
						<>
							<Download size={16} /> Download
						</>
					)}
				</button>

				{/* Divider */}
				<div className="relative py-2">
					<div className="absolute inset-0 flex items-center">
						<div className="w-full border-t border-slate-200 dark:border-white/10" />
					</div>
					<div className="relative flex justify-center text-xs uppercase">
						<span className="bg-slate-50 dark:bg-slate-900 px-2 text-slate-400">
							Or Import Local
						</span>
					</div>
				</div>

				{/* Import Section */}
				<div className="space-y-4">
					{registerResult && (
						<div className="p-3 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-300 text-xs rounded-lg flex items-center gap-2">
							<Info size={14} /> {registerResult.message}
						</div>
					)}

					<div className="grid grid-cols-2 gap-3">
						<button
							type="button"
							onClick={() => handleImport("folder")}
							disabled={isRegistering}
							className="bg-indigo-600 hover:bg-indigo-700 text-white disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:cursor-not-allowed font-medium py-3 px-4 rounded-xl transition-all duration-300 flex flex-col items-center justify-center gap-1 shadow-lg shadow-indigo-500/20 text-xs"
						>
							{isRegistering ? (
								<Loader2 size={20} className="animate-spin" />
							) : (
								<FolderOpen size={20} />
							)}
							<span>Import Folder</span>
						</button>

						<button
							type="button"
							onClick={() => handleImport("file")}
							disabled={isRegistering}
							className="bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 font-medium py-3 px-4 rounded-xl transition-all duration-300 flex flex-col items-center justify-center gap-1 text-xs"
						>
							{isRegistering ? (
								<Loader2 size={20} className="animate-spin" />
							) : (
								<FileText size={20} />
							)}
							<span>Import File</span>
						</button>
					</div>
				</div>
			</div>
		</div>
	);
}
