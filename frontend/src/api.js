const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      // Keep the HTTP message when the response is not JSON.
    }
    throw new Error(message);
  }

  return response.json();
}

export function mediaUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export const api = {
  health: () => request("/api/health"),
  listProjects: () => request("/api/projects"),
  getProject: (projectId) => request(`/api/projects/${projectId}`),
  createProject: (payload) =>
    request("/api/projects", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runProject: (projectId) =>
    request(`/api/projects/${projectId}/run`, {
      method: "POST",
      body: JSON.stringify({ job_type: "pipeline" }),
    }),
  assembleProject: (projectId) =>
    request(`/api/projects/${projectId}/assemble`, {
      method: "POST",
      body: JSON.stringify({ reencode: true }),
    }),
  getJob: (jobId) => request(`/api/jobs/${jobId}`),
  getProjectPlan: (projectId) => request(`/api/projects/${projectId}/project-plan`),
  getAssetIndex: (projectId) => request(`/api/projects/${projectId}/asset-index`),
  getWanManifest: (projectId) => request(`/api/projects/${projectId}/wan-manifest`),
};
