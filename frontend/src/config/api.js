/**
 * Centralized API Base URL Configuration
 * 
 * Defaults to the production Render backend URL:
 * https://smartforming.onrender.com
 * 
 * Can be overridden via REACT_APP_API_URL environment variable.
 */

export const API_BASE_URL = (
  process.env.REACT_APP_API_URL || 'https://smartforming.onrender.com'
).replace(/\/+$/, '');

export default API_BASE_URL;
