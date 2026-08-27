/**
 * Centralized API Base URL Configuration
 * 
 * In local development, defaults to 'http://127.0.0.1:8000'.
 * In production (e.g. Vercel), configure `REACT_APP_API_URL` 
 * in your environment variables to point to your Render backend URL:
 * https://your-backend-name.onrender.com
 */

export const API_BASE_URL = (
  process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');

export default API_BASE_URL;
