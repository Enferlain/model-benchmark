export const API_BASE = "http://localhost:8000/api";

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

export async function generateImages(options: any) {
  const response = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  });
  if (!response.ok) throw new Error('Failed to generate images');
  return response;
}

export async function analyzeImages(options: any) {
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

export async function analyzeModelUrl(url: string, name: string, source: string) {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      name,
      source,
    }),
  });
  if (!response.ok) throw new Error('Failed to analyze model URL');
  return response.json();
}

export async function deleteModel(id: string) {
  const response = await fetch(`${API_BASE}/models/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error('Failed to delete model');
  return response;
}
