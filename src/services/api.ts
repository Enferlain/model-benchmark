export const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export async function fetchModels() {
  const response = await fetch(`${API_BASE}/models`);
  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }
  return response.json();
}

export async function fetchModel(id: string) {
  const response = await fetch(`${API_BASE}/models/${id}`);
  if (!response.ok) {
    throw new Error('Failed to fetch model');
  }
  return response.json();
}

export async function downloadModel(url: string, filename: string, source: string) {
  const response = await fetch(`${API_BASE}/models/download`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url, filename, source }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Failed to start download');
  }
  return response.json();
}

export async function getDownloadStatus() {
    const response = await fetch(`${API_BASE}/models/download/status`);
    if (!response.ok) {
        throw new Error('Failed to get download status');
    }
    return response.json();
}

export async function deleteModel(id: string, deleteFile: boolean) {
    const response = await fetch(`${API_BASE}/models/${id}?delete_file=${deleteFile}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error('Failed to delete model');
    }
    return response.json();
}

export async function generateImages(options: any) {
  const response = await fetch(`${API_BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options)
  });
  if (!response.ok) {
      throw new Error('Failed to start generation');
  }
  return response.json();
}

export async function analyzeImages(options: any) {
  const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
  });
  if (!response.ok) {
      throw new Error('Failed to start analysis');
  }
  return response.json();
}

export async function cancelOperation() {
    const response = await fetch(`${API_BASE}/cancel`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to cancel');
    return response.json();
}

export async function getStatus() {
    const response = await fetch(`${API_BASE}/status`);
    if (!response.ok) throw new Error('Failed to get status');
    return response.json();
}

export async function fetchModelOutputs(modelId: string) {
    const response = await fetch(`${API_BASE}/models/${modelId}/outputs`);
    if (!response.ok) {
        throw new Error('Failed to fetch model outputs');
    }
    return response.json(); // returns ModelOutput[]
}

export async function fetchNote(noteId: string) {
    const response = await fetch(`${API_BASE}/notes/${noteId}`);
    if (response.status === 404) {
        return null; // No note yet
    }
    if (!response.ok) {
        throw new Error('Failed to fetch note');
    }
    return response.json();
}

export async function saveNote(noteId: string, data: { content: string, timestamp: string }) {
    const response = await fetch(`${API_BASE}/notes/${noteId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        throw new Error('Failed to save note');
    }
    return response.json();
}

export async function fetchPrompts() {
  const response = await fetch(`${API_BASE}/prompts`);
  if (!response.ok) {
      throw new Error('Failed to fetch prompts');
  }
  return response.json();
}

export async function savePrompts(prompts: any[]) {
    const response = await fetch(`${API_BASE}/prompts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(prompts)
    });
    if (!response.ok) {
        throw new Error('Failed to save prompts');
    }
    return response.json();
}

export async function createPrompt(formData: FormData) {
    const response = await fetch(`${API_BASE}/prompts/create`, {
        method: 'POST',
        body: formData // No Content-Type header needed for FormData
    });
    if (!response.ok) {
        throw new Error('Failed to create prompt');
    }
    return response.json();
}

export async function updatePromptText(filename: string, data: any) {
    const response = await fetch(`${API_BASE}/prompts/${filename}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        throw new Error('Failed to update prompt');
    }
    return response.json();
}

export async function deletePrompt(filename: string) {
    const response = await fetch(`${API_BASE}/prompts/${filename}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error('Failed to delete prompt');
    }
    return response.json();
}

export async function shufflePrompts() {
    const response = await fetch(`${API_BASE}/prompts/shuffle`, {
        method: 'POST',
    });
    if (!response.ok) {
        throw new Error('Failed to shuffle prompts');
    }
    return response.json();
}

export async function setAllPromptsEnabled(enabled: boolean) {
    const response = await fetch(`${API_BASE}/prompts/bulk/enable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
    });
    if (!response.ok) {
        throw new Error('Failed to set all prompts enabled');
    }
    return response.json();
}
