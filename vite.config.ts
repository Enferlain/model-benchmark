import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
	const env = loadEnv(mode, ".", "");

	// Determine the API target URL.
	// If VITE_API_BASE is set (e.g., http://localhost:8001/api), we use its origin.
	// Otherwise, default to http://localhost:8000.
	let apiTarget = env.VITE_API_BASE || "http://localhost:8000";

	// Normalize: remove trailing slash and /api suffix
	apiTarget = apiTarget.replace(/\/$/, "").replace(/\/api$/, "");

	return {
		server: {
			port: 3000,
			host: "0.0.0.0",
			proxy: {
				"/api": {
					target: apiTarget,
					changeOrigin: true,
					secure: false,
				},
				"/assets": {
					target: apiTarget,
					changeOrigin: true,
					secure: false,
				},
			},
		},
		plugins: [react()],
		define: {
			"process.env.API_KEY": JSON.stringify(env.GEMINI_API_KEY),
			"process.env.GEMINI_API_KEY": JSON.stringify(env.GEMINI_API_KEY),
		},
		resolve: {
			alias: {
				"@": path.resolve(__dirname, "."),
			},
		},
	};
});
