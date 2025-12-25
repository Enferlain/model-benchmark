// Use environment variable for API base, or default to localhost
export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

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

export async function getStatus() {
  const response = await fetch(`${API_BASE}/status`);
  if (!response.ok) throw new Error('Failed to fetch status');
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

export const fetchPrompts = async (): Promise<string[]> => {
  const response = await fetch(`${API_BASE}/prompts`);
  if (!response.ok) {
    throw new Error('Failed to fetch prompts');
  }
  return response.json();
};
