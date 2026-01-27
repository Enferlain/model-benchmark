export type MetricKey =
	| "accuracy"
	| "diversity"
	| "rating"
	| "vqa_score"
	| "lpips_loss";

export type ModelSource = "Civitai" | "HuggingFace" | "Local" | "Unknown";

export interface ModelData {
	id: string;
	hash?: string; // Content hash for stable coloring
	filename?: string;
	name: string;
	source: ModelSource;
	accuracy: number;
	diversity: number;
	rating: number;
	vqa_score?: number;
	lpips_loss?: number;
	ztsnr?: boolean;
	metrics?: Record<MetricKey, number>;
	url: string;
	is_missing?: boolean;
	model_type?: string;
	prediction_type?: string;
	path?: string;
}

export interface MetricOption {
	value: MetricKey;
	label: string;
	description: string;
	direction?: "higher" | "lower";
	extendedDescription?: string;
}

export interface ModelOutput {
	filename: string;
	url: string;
	prompt: string;
	seed: number;
	prompt_idx: number;
	mtime?: number;
}

export interface PromptData {
	id: string;
	filename: string;
	text: string;
	image?: string;
	type: "paired" | "text_only";
	enabled?: boolean;
	alias?: string;
}
