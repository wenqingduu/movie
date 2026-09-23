import { useEffect, useMemo, useState } from "react";
import { api, mediaUrl } from "./api.js";

const SAMPLE_STORY =
  "一个年轻女孩在雨夜的小巷中奔跑，身后有一名侦探追踪她。女孩冲进一家旧书店，发现桌上有一张写着自己名字的照片。";

function formatTime(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function StatusPill({ value }) {
  return <span className={`pill ${value || "unknown"}`}>{value || "unknown"}</span>;
}

function Section({ title, children, actions }) {
  return (
    <section className="section">
      <div className="sectionHeader">
        <h2>{title}</h2>
        {actions ? <div className="actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

function ProjectForm({ onCreated }) {
  const [title, setTitle] = useState("");
  const [story, setStory] = useState(SAMPLE_STORY);
  const [autorun, setAutorun] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await api.createProject({
        title: title.trim() || null,
        story,
        autorun,
      });
      onCreated(result.project.id);
      setTitle("");
      setAutorun(false);
    } catch (exc) {
      setError(exc.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="projectForm" onSubmit={handleSubmit}>
      <label>
        项目名
        <input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="可选，不填会自动截取剧情开头"
        />
      </label>
      <label>
        剧情
        <textarea value={story} onChange={(event) => setStory(event.target.value)} rows={8} />
      </label>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={autorun}
          onChange={(event) => setAutorun(event.target.checked)}
        />
        创建后立即启动生成
      </label>
      {error ? <div className="error">{error}</div> : null}
      <button type="submit" disabled={submitting || !story.trim()}>
        {submitting ? "创建中..." : "创建项目"}
      </button>
    </form>
  );
}

function ProjectList({ projects, selectedId, onSelect, onRefresh }) {
  return (
    <Section
      title="项目"
      actions={
        <button className="ghost" onClick={onRefresh}>
          刷新
        </button>
      }
    >
      <div className="projectList">
        {projects.length === 0 ? <div className="empty">还没有项目</div> : null}
        {projects.map((project) => (
          <button
            key={project.id}
            className={`projectItem ${selectedId === project.id ? "selected" : ""}`}
            onClick={() => onSelect(project.id)}
          >
            <span className="projectTitle">{project.title}</span>
            <StatusPill value={project.status} />
            <span className="projectMeta">{formatTime(project.created_at)}</span>
          </button>
        ))}
      </div>
    </Section>
  );
}

function JobCard({ job }) {
  if (!job) {
    return <div className="empty">暂无任务</div>;
  }
  return (
    <div className="jobCard">
      <div className="jobTop">
        <strong>{job.type}</strong>
        <StatusPill value={job.status} />
      </div>
      <div className="progressTrack">
        <div className="progressBar" style={{ width: `${job.progress || 0}%` }} />
      </div>
      <div className="jobMessage">{job.message || "-"}</div>
      {job.error ? <pre className="errorBlock">{job.error}</pre> : null}
    </div>
  );
}

function ArtifactGrid({ artifacts }) {
  const previewArtifacts = useMemo(
    () => artifacts.filter((item) => item.kind === "image" || item.kind === "video"),
    [artifacts],
  );

  if (artifacts.length === 0) {
    return <div className="empty">项目目录里还没有产物</div>;
  }

  return (
    <>
      <div className="artifactGrid">
        {previewArtifacts.slice(0, 12).map((artifact) => (
          <a
            key={artifact.name}
            className="artifactCard"
            href={mediaUrl(artifact.url)}
            target="_blank"
            rel="noreferrer"
          >
            {artifact.kind === "image" ? (
              <img src={mediaUrl(artifact.url)} alt={artifact.name} />
            ) : (
              <video src={mediaUrl(artifact.url)} controls />
            )}
            <span>{artifact.name}</span>
          </a>
        ))}
      </div>
      <details className="fileDetails">
        <summary>全部文件 ({artifacts.length})</summary>
        <ul>
          {artifacts.map((artifact) => (
            <li key={artifact.name}>
              <a href={mediaUrl(artifact.url)} target="_blank" rel="noreferrer">
                {artifact.name}
              </a>
              <span>{Math.round(artifact.bytes / 1024)} KB</span>
            </li>
          ))}
        </ul>
      </details>
    </>
  );
}

function JsonPanel({ title, loader, selectedId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function load() {
    setError("");
    setLoading(true);
    try {
      setData(await loader(selectedId));
    } catch (exc) {
      setData(null);
      setError(exc.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setData(null);
    setError("");
  }, [selectedId]);

  return (
    <div className="jsonPanel">
      <div className="jsonHeader">
        <strong>{title}</strong>
        <button className="ghost" onClick={load} disabled={!selectedId || loading}>
          {loading ? "读取中..." : "读取"}
        </button>
      </div>
      {error ? <div className="error">{error}</div> : null}
      {data ? <pre>{JSON.stringify(data, null, 2)}</pre> : null}
    </div>
  );
}

function ProjectDetail({ projectId, onRefreshProjects }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [busyAction, setBusyAction] = useState("");

  async function refresh() {
    if (!projectId) return;
    try {
      setError("");
      setDetail(await api.getProject(projectId));
    } catch (exc) {
      setError(exc.message);
    }
  }

  async function runAction(action) {
    setBusyAction(action);
    setError("");
    try {
      if (action === "run") {
        await api.runProject(projectId);
      } else if (action === "assemble") {
        await api.assembleProject(projectId);
      }
      await refresh();
      await onRefreshProjects();
    } catch (exc) {
      setError(exc.message);
    } finally {
      setBusyAction("");
    }
  }

  useEffect(() => {
    refresh();
  }, [projectId]);

  useEffect(() => {
    if (!projectId) return undefined;
    const timer = setInterval(refresh, 2500);
    return () => clearInterval(timer);
  }, [projectId]);

  if (!projectId) {
    return (
      <Section title="项目详情">
        <div className="empty">从左侧选择一个项目</div>
      </Section>
    );
  }

  if (!detail) {
    return (
      <Section title="项目详情">
        <div className="empty">{error || "加载中..."}</div>
      </Section>
    );
  }

  const { project, artifacts } = detail;
  const finalVideo = artifacts.find((item) => item.name === "final/final_video.mp4");

  return (
    <div className="detailStack">
      <Section
        title="项目详情"
        actions={
          <>
            <button className="ghost" onClick={refresh}>
              刷新
            </button>
            <button onClick={() => runAction("run")} disabled={Boolean(busyAction)}>
              {busyAction === "run" ? "启动中..." : "启动生成"}
            </button>
            <button onClick={() => runAction("assemble")} disabled={Boolean(busyAction)}>
              {busyAction === "assemble" ? "拼接中..." : "拼接视频"}
            </button>
          </>
        }
      >
        {error ? <div className="error">{error}</div> : null}
        <div className="metaGrid">
          <span>项目 ID</span>
          <strong>{project.id}</strong>
          <span>状态</span>
          <StatusPill value={project.status} />
          <span>目录</span>
          <code>{project.project_dir}</code>
          <span>模型</span>
          <code>{project.generation_model}</code>
        </div>
        <JobCard job={project.latest_job} />
      </Section>

      {finalVideo ? (
        <Section title="最终视频">
          <video className="finalVideo" src={mediaUrl(finalVideo.url)} controls />
        </Section>
      ) : null}

      <Section title="产物预览">
        <ArtifactGrid artifacts={artifacts} />
      </Section>

      <Section title="JSON 调试面板">
        <div className="jsonGrid">
          <JsonPanel title="project_plan.json" loader={api.getProjectPlan} selectedId={projectId} />
          <JsonPanel title="asset_index.json" loader={api.getAssetIndex} selectedId={projectId} />
          <JsonPanel title="wan_manifest_product.json" loader={api.getWanManifest} selectedId={projectId} />
        </div>
      </Section>
    </div>
  );
}

export default function App() {
  const [projects, setProjects] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [apiError, setApiError] = useState("");

  async function refreshProjects() {
    try {
      setApiError("");
      const result = await api.listProjects();
      setProjects(result.projects);
      if (!selectedId && result.projects.length > 0) {
        setSelectedId(result.projects[0].id);
      }
    } catch (exc) {
      setApiError(exc.message);
    }
  }

  useEffect(() => {
    refreshProjects();
  }, []);

  return (
    <main className="appShell">
      <header className="topBar">
        <div>
          <h1>Movie Local Studio</h1>
          <p>本地剧情到多镜头视频的开发工作台</p>
        </div>
        <code>FastAPI: 8000 / React: 3000</code>
      </header>

      {apiError ? <div className="error banner">{apiError}</div> : null}

      <div className="layout">
        <aside className="sidebar">
          <Section title="新建项目">
            <ProjectForm
              onCreated={async (projectId) => {
                setSelectedId(projectId);
                await refreshProjects();
              }}
            />
          </Section>
          <ProjectList
            projects={projects}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onRefresh={refreshProjects}
          />
        </aside>
        <ProjectDetail projectId={selectedId} onRefreshProjects={refreshProjects} />
      </div>
    </main>
  );
}
