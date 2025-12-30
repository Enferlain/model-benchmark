import { API_BASE } from '../../services/api';

export const getImageUrl = (url: string) => {
    if (!url) return '';
    if (url.startsWith('http')) return url;

    // If API_BASE is an absolute URL, we might need to prepend its origin.
    // If API_BASE is relative (e.g. /api), we just return url (assuming it is relative to root like /assets).
    if (API_BASE.startsWith('http')) {
        try {
            const origin = new URL(API_BASE).origin;
            return `${origin}${url.startsWith('/') ? '' : '/'}${url}`;
        } catch (e) {
            return url;
        }
    }

    return url;
};
