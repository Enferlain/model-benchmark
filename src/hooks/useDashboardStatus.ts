import { useCallback, useEffect, useState } from "react";
import {
	downloadModel as apiDownloadModel,
	cancelOperation,
	getDownloadStatus,
	getStatus,
} from "../services/api";

export interface DownloadProgress {
	current: number;
	total: number;
	status: string;
	filename: string;
}

export interface GenerationStatus {
	is_running: boolean;
	current_model: string | null;
	progress: { current: number; total: number };
}

export const parseUrl = (
	url: string,
): { name: string; source: "Civitai" | "HuggingFace" | "Unknown" } | null => {
	try {
		const urlObj = new URL(url);

		if (url.includes("civitai.com")) {
			const parts = urlObj.pathname.split("/").filter(Boolean);
			const namePart =
				parts.length >= 3 ? parts[2] : parts[1] || "Civitai Model";
			return {
				name: namePart
					.replace(/-/g, " ")
					.replace(/\b\w/g, (l) => l.toUpperCase()),
				source: "Civitai",
			};
		}

		if (url.includes("huggingface.co")) {
			const parts = urlObj.pathname.split("/").filter(Boolean);
			const lastPart = parts[parts.length - 1];

			// Check if the URL points directly to a model file
			if (lastPart && /\.(safetensors|ckpt|pt|bin|pth)$/i.test(lastPart)) {
				return {
					name: lastPart,
					source: "HuggingFace",
				};
			}

			const namePart =
				parts.length >= 2 ? `${parts[0]}/${parts[1]}` : "HF Model";
			return {
				name: namePart,
				source: "HuggingFace",
			};
		}

		if (url) {
			return { name: "Unknown Model", source: "Unknown" };
		}

		return null;
	} catch (_e) {
		return null;
	}
};

interface UseDashboardStatusProps {
	fetchModels: () => Promise<void>;
}

export function useDashboardStatus({ fetchModels }: UseDashboardStatusProps) {
	const [isDownloading, setIsDownloading] = useState(false);
	const [downloadError, setDownloadError] = useState<string | null>(null);
	const [downloadProgress, setDownloadProgress] = useState<DownloadProgress>({
		current: 0,
		total: 0,
		status: "idle",
		filename: "",
	});

	const [isScanning, setIsScanning] = useState(false);
	const [generationStatus, setGenerationStatus] = useState<GenerationStatus>({
		is_running: false,
		current_model: null,
		progress: { current: 0, total: 0 },
	});

	// Poll download status
	useEffect(() => {
		if (!isDownloading) return;

		const interval = setInterval(async () => {
			try {
				const status = await getDownloadStatus();
				setDownloadProgress({
					current: status.progress,
					total: status.total,
					status: status.status,
					filename: status.current_file,
				});

				if (status.status === "completed" || status.status === "error") {
					setIsDownloading(false);
					if (status.status === "completed") {
						await fetchModels();
					} else {
						setDownloadError(status.error || "Download failed");
					}
				}
			} catch (e) {
				console.error("Download status poll error:", e);
				setDownloadError("Error polling download status");
				setIsDownloading(false);
			}
		}, 1000);

		return () => clearInterval(interval);
	}, [isDownloading, fetchModels]);

	// Poll generation status
	useEffect(() => {
		if (!isScanning) return;

		const interval = setInterval(async () => {
			try {
				const status = await getStatus();
				setGenerationStatus(status);
				if (!status.is_running) {
					setIsScanning(false);
				}
			} catch (e) {
				console.error("Status poll error:", e);
			}
		}, 1000);

		return () => clearInterval(interval);
	}, [isScanning]);

	// Initial status check
	useEffect(() => {
		const checkInitialStatus = async () => {
			try {
				const status = await getDownloadStatus();
				if (status.is_downloading || status.status === "downloading") {
					setIsDownloading(true);
					setDownloadProgress({
						current: status.progress,
						total: status.total,
						status: status.status,
						filename: status.current_file,
					});
				}

				const genStatus = await getStatus();
				if (genStatus.is_running) {
					setIsScanning(true);
					setGenerationStatus(genStatus);
				}
			} catch (e) {
				console.error("Failed to restore status:", e);
			}
		};

		checkInitialStatus();
	}, []);

	const handleDownloadModel = useCallback(
		async (url: string) => {
			if (isDownloading) return;
			setDownloadError(null);
			const info = parseUrl(url);
			if (!info) {
				setDownloadError(
					"Please enter a valid Civitai or HuggingFace model URL.",
				);
				return;
			}

			setIsDownloading(true);
			setDownloadProgress({
				current: 0,
				total: 0,
				status: "downloading",
				filename: info.name,
			});

			try {
				await apiDownloadModel(url, info.name, info.source);
			} catch (error: unknown) {
				console.error("Error starting download:", error);
				const message =
					error instanceof Error
						? error.message
						: "Error connecting to backend or starting download.";
				setDownloadError(message);
				setIsDownloading(false);
			}
		},
		[isDownloading],
	);

	const handleCancel = useCallback(async () => {
		try {
			await cancelOperation();
		} catch (error) {
			console.error("Cancel error:", error);
		}
	}, []);

	return {
		isDownloading,
		downloadError,
		downloadProgress,
		setDownloadError,
		isScanning,
		setIsScanning,
		generationStatus,
		setGenerationStatus,
		handleDownloadModel,
		handleCancel,
	};
}
