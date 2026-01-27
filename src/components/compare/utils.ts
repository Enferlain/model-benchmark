export const getImageUrl = (url: string, mtime?: number) => {
	if (url.startsWith("http")) return url;
	// If we have a configured base, use it (stripping /api if present)
	// Otherwise assume relative paths work via proxy
	const base = import.meta.env.VITE_API_BASE
		? import.meta.env.VITE_API_BASE.replace(/\/api$/, "")
		: "";

	const fullUrl = `${base}${url}`;
	return mtime ? `${fullUrl}?v=${mtime}` : fullUrl;
};
