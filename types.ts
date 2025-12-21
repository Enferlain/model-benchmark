export type MetricKey = 'accuracy' | 'diversity' | 'rating' | 'vqa_score' | 'lpips_loss';

export type ModelSource = 'Civitai' | 'HuggingFace' | 'Unknown';

export interface ModelData {
  id: string;
  name: string;
  source: ModelSource;
  accuracy: number;
  diversity: number;
  rating: number;
  vqa_score?: number;
  lpips_loss?: number;
  metrics?: Record<MetricKey, number>;
  url: string;
}

export interface MetricOption {
  value: MetricKey;
  label: string;
  description: string;
  direction?: 'higher' | 'lower';
  extendedDescription?: string;
}
