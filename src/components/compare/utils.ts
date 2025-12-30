export const getImageUrl = (url: string) =>
  `${import.meta.env.VITE_API_BASE?.replace('/api', '') || 'http://localhost:8000'}${url}`;
