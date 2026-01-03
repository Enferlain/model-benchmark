// Use environment variable for API base, or default to relative path for proxy
export const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export interface ScanOptions {
  sampler?: string;
  steps?: number;
  guidance_scale?: number;
  seed?: number;
  images_per_prompt?: number;
  num_prompts?: number;
  width?: number;
  height?: number;
  [key: string]: any;
}

export async function fetchModels() {
  const response = await fetch(`${API_BASE}/models`);
  if (!response.ok) throw new Error('Failed to fetch models');
  return response.json();
}

export async function scanModels() {
  const response = await fetch(`${API_BASE}/models/scan`, { method: "POST" });
  if (!response.ok) throw new Error("Failed to scan models");
  return response.json();
}

export async function getStatus() {
  const response = await fetch(`${API_BASE}/status`);
  if (!response.ok) throw new Error('Failed to fetch status');
  return response.json();
}

export async function fetchBenchmarkRuns() {
  const response = await fetch(`${API_BASE}/runs`);
  if (!response.ok) throw new Error('Failed to fetch benchmark runs');
  return response.json();
}

export async function generateImages(options: ScanOptions) {
  const response = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!response.ok) throw new Error('Failed to generate images');
  return response;
}

export async function analyzeImages(options: ScanOptions) {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!response.ok) throw new Error('Failed to analyze images');
  return response;
}

export async function cancelOperation() {
  const response = await fetch(`${API_BASE}/cancel`, { method: 'POST' });
  if (!response.ok) throw new Error('Failed to cancel operation');
  return response;
}

export async function downloadModel(url: string, name: string, source: string) {
  const response = await fetch(`${API_BASE}/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      name,
      source,
    }),
  });
  if (!response.ok) throw new Error('Failed to start download');
  return response.json();
}

export async function getDownloadStatus() {
  const response = await fetch(`${API_BASE}/models/download/status`);
  if (!response.ok) throw new Error('Failed to get download status');
  return response.json();
}

export async function deleteModel(id: string, deleteFile: boolean = false) {
  const response = await fetch(`${API_BASE}/models/${id}?delete_file=${deleteFile}`, { method: "DELETE" });
  if (!response.ok) throw new Error('Failed to delete model');
  return response;
}

export async function fetchModelOutputs(modelId: string) {
  const response = await fetch(`${API_BASE}/models/${modelId}/outputs`);
  if (!response.ok) {
    throw new Error('Failed to fetch model outputs');
  }
  return response.json();
};

export const fetchPrompts = async (): Promise<any[]> => {
  const response = await fetch(`${API_BASE}/prompts`);
  if (!response.ok) {
    throw new Error('Failed to fetch prompts');
  }
  return response.json();
};

export const createPrompt = async (formData: FormData) => {
  const response = await fetch(`${API_BASE}/prompts/create`, {
    method: 'POST',
    body: formData, // Auto-sets Content-Type to multipart/form-data
  });
  if (!response.ok) throw new Error('Failed to create prompt');
  return response.json();
};

export const updatePromptText = async (filename: string, textOrPayload: string | any) => {
  const body = typeof textOrPayload === 'string' ? { text: textOrPayload } : textOrPayload;
  
  const response = await fetch(`${API_BASE}/prompts/${filename}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error('Failed to update prompt');
  return response.json();
};

export const deletePrompt = async (filename: string) => {
  const response = await fetch(`${API_BASE}/prompts/${filename}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete prompt');
  return response.json();
};

export const shufflePrompts = async () => {
  const response = await fetch(`${API_BASE}/prompts/shuffle`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to shuffle prompts');
  return response.json();
};

export const setAllPromptsEnabled = async (enabled: boolean) => {
  const response = await fetch(`${API_BASE}/prompts/bulk/enable`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) throw new Error('Failed to update all prompts');
  return response.json();
};

export const fetchNote = async (noteId: string) => {
  const response = await fetch(`${API_BASE}/notes/${noteId}`);
  if (!response.ok) throw new Error('Failed to fetch note');
  return response.json();
};

export const saveNote = async (noteId: string, content: any) => {
  const response = await fetch(`${API_BASE}/notes/${noteId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(content),
  });
  if (!response.ok) throw new Error('Failed to save note');
  return response.json();
};

export interface ParamCheckResult {
  matches: boolean;
  existing_params: {
    steps: number;
    cfg: number;
    sampler: string;
    width: number;
    height: number;
  } | null;
  mismatched_models: Array<{
    name: string;
    existing_params: {
      steps: number;
      cfg: number;
      sampler: string;
      width: number;
      height: number;
    };
  }>;
  current_params: {
    steps: number;
    cfg: number;
    sampler: string;
    width: number;
    height: number;
  };
}

export async function checkParams(options: ScanOptions): Promise<ParamCheckResult> {
  const response = await fetch(`${API_BASE}/check-params`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!response.ok) throw new Error('Failed to check parameters');
  return response.json();
}

export async function archiveModel(modelName: string) {
  const response = await fetch(`${API_BASE}/archive/${encodeURIComponent(modelName)}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to archive model');
  return response.json();
}

export interface CoverageCheckResult {
  all_match: boolean;
  common_count: number;
  image_count_mismatch: boolean;
  model_coverage: Array<{
    name: string;
    count: number;
    image_count: number;
    missing_count: number;
    extra_count: number;
  }>;
}

export async function checkCoverage(): Promise<CoverageCheckResult> {
  const response = await fetch(`${API_BASE}/analyze/check-coverage`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to check coverage');
  return response.json();
}

