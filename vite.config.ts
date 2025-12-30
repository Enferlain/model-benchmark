import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const apiTarget = env.VITE_API_BASE || 'http://localhost:8000';

    // If VITE_API_BASE is set, we use its origin for proxying.
    // If not, we default to http://localhost:8000
    // Note: If VITE_API_BASE is a relative path (e.g. /api), this might be weird for a proxy target.
    // Usually VITE_API_BASE in dev is the full URL of backend.

    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        proxy: {
            '/api': {
                target: apiTarget,
                changeOrigin: true,
                secure: false,
            },
            '/assets': {
                target: apiTarget.replace(/\/api$/, '') || 'http://localhost:8000',
                changeOrigin: true,
                secure: false,
            }
        }
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
