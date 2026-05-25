import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "../ui/button";
import { Card } from "../ui/card";
import { Textarea } from "../ui/textarea";
import { Badge } from "../ui/badge";
import { Spinner } from "../ui/spinner";
import {
  cancelResearchJob,
  getResearchHistory,
  getResearchSession,
  getResearchJobResult,
  getResearchJobStatus,
  startResearchJob,
} from "../../services/researchService";
import type { JobResultResponse, JobStatus, Paper, ResearchHistoryItem, ResearchResult } from "../../types/api";

const agentNames = [
  "QueryAgent",
  "QueryExpansionAgent",
  "RetrievalAgent",
  "RelevanceVerifierAgent",
  "AcquisitionAgent",
  "RetrieverAgent",
  "CompressionAgent",
  "EvidenceExtractorAgent",
  "ReasoningAgent",
  "ReportGenerator",
];

type AgentState = { name: string; status: "idle" | "running" | "completed" | "failed" };

function formatPaperSource(paper: Paper) {
  return paper.source || paper.url || "OpenAlex";
}

function clampProgress(value: number) {
  return Math.max(0, Math.min(100, value));
}

function AnswerMarkdown({ content }: { content: string }) {
  return (
    <div className="max-w-none text-slate-300">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ node, ...props }) => (
            <h1
              className="mt-10 mb-5 text-3xl font-bold tracking-tight text-white"
              {...props}
            />
          ),

          h2: ({ node, ...props }) => (
            <h2
              className="mt-10 mb-4 border-b border-white/5 pb-3 text-2xl font-semibold tracking-tight text-white"
              {...props}
            />
          ),

          h3: ({ node, ...props }) => (
            <h3
              className="mt-6 mb-2 text-lg font-semibold text-violet-300"
              {...props}
            />
          ),

          p: ({ node, ...props }) => (
            <p
              className="mb-5 text-[15px] font-normal leading-8 text-slate-300"
              {...props}
            />
          ),

          ul: ({ node, ...props }) => (
            <ul
              className="mb-6 ml-5 list-disc space-y-3 text-[15px] leading-8 text-slate-300"
              {...props}
            />
          ),

          ol: ({ node, ...props }) => (
            <ol
              className="mb-6 ml-5 list-decimal space-y-3 text-[15px] leading-8 text-slate-300"
              {...props}
            />
          ),

          li: ({ node, ...props }) => (
            <li
              className="font-normal leading-8 text-slate-300"
              {...props}
            />
          ),

          strong: ({ node, ...props }) => (
            <strong
              className="font-semibold text-slate-100"
              {...props}
            />
          ),

          a: ({ node, ...props }) => (
            <a
              className="text-violet-300 underline underline-offset-4 transition hover:text-violet-200"
              {...props}
            />
          ),

          blockquote: ({ node, ...props }) => (
            <blockquote
              className="my-6 border-l-2 border-violet-400/40 pl-4 italic text-slate-400"
              {...props}
            />
          ),

          code: ({ inline, className, children, ...props }: any) =>
            inline ? (
              <code
                className="rounded bg-slate-800 px-1.5 py-1 text-sm text-violet-200"
                {...props}
              >
                {children}
              </code>
            ) : (
              <pre className="my-6 overflow-x-auto rounded-2xl border border-white/5 bg-slate-950/80 p-5 text-sm text-slate-200">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            ),

          hr: ({ node, ...props }) => (
            <hr className="my-8 border-white/5" {...props} />
          ),

          table: ({ node, ...props }) => (
            <div className="my-6 overflow-x-auto rounded-2xl border border-white/5 bg-slate-950/60">
              <table className="min-w-full divide-y divide-white/5 text-sm" {...props} />
            </div>
          ),

          th: ({ node, ...props }) => (
            <th
              className="bg-slate-900/80 px-4 py-3 text-left text-sm font-semibold text-slate-100"
              {...props}
            />
          ),

          td: ({ node, ...props }) => (
            <td
              className="border-t border-white/5 px-4 py-3 text-sm text-slate-300"
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default function WorkspacePage() {
  const [query, setQuery] = useState("");
  const [currentResult, setCurrentResult] = useState<ResearchResult | null>(null);
  const [history, setHistory] = useState<ResearchHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentStates, setAgentStates] = useState<AgentState[]>(
    agentNames.map((name) => ({ name, status: "idle" }))
  );
  const [logs, setLogs] = useState<JobStatus["logs"]>([]);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [isJobBasedMode, setIsJobBasedMode] = useState(true);
  const [pollingEnabled, setPollingEnabled] = useState(true);
  const [selectedPaperIndex, setSelectedPaperIndex] = useState(0);
  const [paperModalOpen, setPaperModalOpen] = useState(false);
  const pollingRef = useRef<number | null>(null);

  const THEME_STORAGE_KEY = "researchPilotTheme";
  const JOB_MODE_STORAGE_KEY = "researchPilotJobBased";
  const POLLING_STORAGE_KEY = "researchPilotPollingEnabled";

  useEffect(() => {
    setHistoryLoading(true);
    getResearchHistory()
      .then((items) => setHistory(items))
      .catch(() => {
        /* history loading can be silent */
      })
      .finally(() => setHistoryLoading(false));

    const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    const storedJobMode = localStorage.getItem(JOB_MODE_STORAGE_KEY);
    const storedPolling = localStorage.getItem(POLLING_STORAGE_KEY);

    const initialTheme = storedTheme === "light" ? "light" : "dark";
    setTheme(initialTheme);
    setIsJobBasedMode(storedJobMode !== "false");
    setPollingEnabled(storedPolling !== "false");
    document.documentElement.classList.toggle("dark", initialTheme === "dark");
    document.documentElement.classList.toggle("light", initialTheme === "light");
    document.body.style.colorScheme = initialTheme;
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.classList.toggle("light", theme === "light");
    document.body.style.colorScheme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    const pollStatus = async () => {
      try {
        const status = await getResearchJobStatus(jobId);
        setJobStatus(status);
        setLogs(status.logs || []);

        if (status.partial_results) {
          setCurrentResult((prev) => ({
            ...prev,
            ...status.partial_results,
          } as ResearchResult));
        }

        const current = status.current_agent || "";
        const index = agentNames.findIndex((name) => name === current);

        setAgentStates(
          agentNames.map((name, nameIndex) => {
            if (status.status === "completed") {
              return { name, status: "completed" };
            }
            if (status.status === "failed" && name === current) {
              return { name, status: "failed" };
            }
            if (nameIndex < index) {
              return { name, status: "completed" };
            }
            if (nameIndex === index) {
              return { name, status: status.status === "running" ? "running" : "idle" };
            }
            return { name, status: "idle" };
          })
        );

        if (status.status === "completed") {
          const resultResponse = await getResearchJobResult(jobId);
          if (resultResponse.final_result) {
            setCurrentResult(resultResponse.final_result);
          }
          setLoading(false);
          if (pollingRef.current) {
            window.clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        }

        if (status.status === "failed" || status.status === "cancelled") {
          setLoading(false);
          if (pollingRef.current) {
            window.clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          if (status.error_message) {
            setError(status.error_message);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to fetch job status.");
        setLoading(false);
        if (pollingRef.current) {
          window.clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
      }
    };

    const shouldPoll = pollingEnabled && isJobBasedMode;

    if (shouldPoll && !pollingRef.current) {
      pollStatus();
      pollingRef.current = window.setInterval(pollStatus, 3500);
    }

    if (!shouldPoll && pollingRef.current) {
      window.clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [jobId, pollingEnabled, isJobBasedMode]);

  const paperCards = useMemo(() => {
    if (!currentResult) return [];
    return currentResult.papers?.map((paper, index) => ({
      ...paper,
      relevance_score: paper.relevance_score ?? (((currentResult.papers?.length ?? 1) - index) / (currentResult.papers?.length ?? 1)) * 10,
      source: formatPaperSource(paper),
    })) ?? [];
  }, [currentResult]);

  useEffect(() => {
    setSelectedPaperIndex(0);
  }, [paperCards.length]);

  function getProgressPercent(status?: JobStatus | null) {
    if (!status) return 0;
    if (typeof status.progress_percentage === "number") return status.progress_percentage;
    if (status.status === "completed") return 100;
    if (status.status === "failed" || status.status === "cancelled") return 100;
    const index = agentNames.findIndex((name) => name === status.current_agent);
    if (index >= 0) {
      return Math.round(((index + 1) / agentNames.length) * 92) + 4;
    }
    return status.status === "running" ? 12 : 0;
  }

  function previewText(summary?: string, query?: string, title?: string) {
    const fallback = query ? `Research session on ${query}` : title ? title : "Research session";
    if (!summary) return fallback;
    const lines = summary.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const first = lines.slice(0, 2).join(" ");
    if (first.length > 140) return first.slice(0, 137) + "…";
    return first;
  }

  function executivePreview(result?: ResearchResult | null) {
    if (!result) return null;
    const summary = result.executive_summary;
    const fallbackSource = result.full_report ?? result.generated_answer;

    const textSource = summary ?? null;
    if (!textSource) return null;

    // Split into paragraphs and take up to 3 concise paragraphs
    const paragraphs = textSource.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
    const firstThree = paragraphs.slice(0, 3).join("\n\n");
    // Truncate if excessively long
    if (firstThree.length > 1200) return firstThree.slice(0, 1197) + "…";
    return firstThree;
  }

  async function loadSession(item: ResearchHistoryItem) {
    setError(null);
    setLoading(true);
    setActiveSessionId(item.id);
    try {
      const result = await getResearchSession(item.id);
      if (result) {
        setCurrentResult(result);
      }

      // If we have an associated job id, try to load job result/status to reconstruct timeline/logs
      if (item.job_id) {
        try {
          const jobResp = await getResearchJobResult(item.job_id);
          setJobStatus({
            job_id: jobResp.job_id,
            status: jobResp.status,
            current_agent: jobResp.current_agent,
            progress_percentage: jobResp.progress_percentage ?? 100,
            logs: jobResp.logs ?? [],
          } as JobStatus);
          setLogs(jobResp.logs ?? []);

          // build agentStates from known agent list and current_agent
          const current = jobResp.current_agent || "";
          const index = agentNames.findIndex((name) => name === current);
          setAgentStates(
            agentNames.map((name, nameIndex) => {
              if (jobResp.status === "completed") return { name, status: "completed" };
              if (jobResp.status === "failed" && name === current) return { name, status: "failed" };
              if (nameIndex < index) return { name, status: "completed" };
              if (nameIndex === index) return { name, status: jobResp.status === "running" ? "running" : "idle" };
              return { name, status: "idle" };
            })
          );
        } catch (err) {
          // ignore job result errors
        }
      } else {
        // If no job id, clear jobStatus/logs
        setJobStatus(null);
        setLogs([]);
        setAgentStates(agentNames.map((name) => ({ name, status: "idle" })));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load session.");
    } finally {
      setLoading(false);
    }
  }

  const progress = clampProgress(getProgressPercent(jobStatus));

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Please enter a research query to continue.");
      return;
    }
    setError(null);
    setLoading(true);
    setCurrentResult(null);
    setJobId(null);
    setJobStatus(null);
    setLogs([]);
    setAgentStates(agentNames.map((name) => ({ name, status: "idle" })));

    try {
      const response = await startResearchJob(query.trim());
      setJobId(response.job_id);
      setJobStatus({
        job_id: response.job_id,
        status: response.status as JobStatus["status"],
        current_agent: "QueryAgent",
        progress_percentage: 0,
        logs: [],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start research job.");
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;
    try {
      await cancelResearchJob(jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to cancel job.");
    }
  };

  const emptyState = (
    <div className="rounded-3xl border border-dashed border-white/10 bg-surface-900/80 p-10 text-center text-slate-400">
      <p className="text-lg font-medium text-slate-200">Ready for your next research query.</p>
      <p className="mt-3 text-sm leading-6 text-slate-400">Submit a query and monitor progress while the model generates the final research report.</p>
    </div>
  );

  return (
    <div className="mx-auto max-w-[1600px] pb-16 pt-6">
      <div className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-accent-400">Research Workspace</p>
          <h1 className="mt-3 text-4xl font-semibold text-white sm:text-5xl">Live multi-agent research assistant</h1>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="sm"
            variant={theme === "dark" ? "default" : "secondary"}
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "Switch to light" : "Switch to dark"}
          </Button>
          <Button
            size="sm"
            variant={isJobBasedMode ? "default" : "secondary"}
            onClick={() => {
              setIsJobBasedMode((value) => {
                const next = !value;
                localStorage.setItem(JOB_MODE_STORAGE_KEY, String(next));
                return next;
              });
            }}
          >
            {isJobBasedMode ? "Job-based ON" : "Job-based OFF"}
          </Button>
          <Button
            size="sm"
            variant={pollingEnabled ? "default" : "secondary"}
            onClick={() => {
              setPollingEnabled((value) => {
                const next = !value;
                localStorage.setItem(POLLING_STORAGE_KEY, String(next));
                return next;
              });
            }}
          >
            {pollingEnabled ? "Polling ON" : "Polling OFF"}
          </Button>
        </div>
      </div>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <Badge variant={theme === "dark" ? "success" : "muted"}>{theme === "dark" ? "Dark theme" : "Light theme"}</Badge>
        <Badge variant={isJobBasedMode ? "success" : "secondary"}>{isJobBasedMode ? "Job-based" : "Manual mode"}</Badge>
        <Badge variant={pollingEnabled ? "success" : "warning"}>{pollingEnabled ? "Polling enabled" : "Polling paused"}</Badge>
      </div>

      <div className="grid gap-6 xl:grid-cols-[360px_1fr_420px]">
        <section className="space-y-6">
          <Card>
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Query builder</p>
                <h2 className="mt-2 text-xl font-semibold text-white">Search controls</h2>
              </div>
              <Badge variant={isJobBasedMode ? "success" : "secondary"}>{isJobBasedMode ? "Async" : "Manual"}</Badge>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300" htmlFor="query">
                  Research Query
                </label>
                <Textarea
                  id="query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="E.g. Latest advances in foundation model evaluation for healthcare applications"
                />
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <Button className="w-full sm:w-auto" onClick={handleSubmit} disabled={loading}>
                  {loading ? "Starting research job…" : "Launch research job"}
                </Button>
                <Button variant="secondary" className="w-full sm:w-auto" onClick={handleCancel} disabled={!jobId || !loading}>
                  Cancel job
                </Button>
              </div>
            </div>
            <div className="mt-6 space-y-3 rounded-3xl border border-white/10 bg-surface-800/80 p-4">
              <p className="text-sm font-semibold text-slate-200">Search tips</p>
              <ul className="space-y-2 text-sm leading-6 text-slate-400">
                <li>• Use precise research topics and target domains.</li>
                <li>• Run a query once, then monitor progress while inference executes.</li>
                <li>• Cancel long-running jobs if needed.</li>
              </ul>
            </div>
          </Card>

          <Card>
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-accent-400">History</p>
                <h2 className="text-xl font-semibold text-white">Recent sessions</h2>
              </div>
            </div>
            <div className="space-y-3">
              {historyLoading ? (
                // Loading skeletons for history
                Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="animate-pulse rounded-3xl border border-white/5 bg-surface-900/60 p-4">
                    <div className="h-4 w-3/4 rounded bg-slate-800/70" />
                    <div className="mt-3 h-3 w-5/6 rounded bg-slate-800/70" />
                    <div className="mt-4 flex items-center justify-between text-xs text-slate-500">
                      <div className="h-3 w-12 rounded bg-slate-800/70" />
                      <div className="h-3 w-20 rounded bg-slate-800/70" />
                    </div>
                  </div>
                ))
              ) : history.length === 0 ? (
                <p className="text-sm text-slate-400">No previous sessions found yet.</p>
              ) : (
                history.slice(0, 5).map((item) => {
                  const isActive = activeSessionId === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => loadSession(item)}
                      className={`w-full text-left rounded-3xl border border-white/10 p-4 transition-colors ${isActive ? "ring-2 ring-accent-400 bg-surface-800/90" : "bg-surface-900/75 hover:bg-surface-900/90"}`}
                    >
                      <p className="font-medium text-slate-100">{(item.title || item.query || "Research session").slice(0, 60)}</p>
                      <p className="mt-2 text-sm text-slate-400">{previewText(item.summary, item.query, item.title)}</p>
                      <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
                        <span>{item.paper_count ?? 0} papers</span>
                        <span>{new Date(item.created_at).toLocaleDateString()}</span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </Card>
        </section>

        <section className="space-y-6">
          <Card className="space-y-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Research console</p>
                <h2 className="text-xl font-semibold text-white">Job progress</h2>
              </div>
              {loading && <Spinner />}
            </div>

            {jobStatus ? (
              <div>
                <div className="mb-4 rounded-full bg-slate-800/80 p-1">
                  <div
                    className="h-3 rounded-full bg-gradient-to-r from-accent-400 to-sky-400 transition-all"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-slate-400">
                  <span>Job ID: {jobStatus.job_id}</span>
                  <span>•</span>
                  <span>Status: {jobStatus.status}</span>
                  <span>•</span>
                  <span>Agent: {jobStatus.current_agent || "Waiting"}</span>
                </div>
                <div className="rounded-3xl border border-white/10 bg-surface-900/80 p-4">
                  <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Live logs</p>
                  <div className="mt-3 max-h-48 space-y-2 overflow-y-auto text-sm text-slate-300">
                    {logs.length === 0 ? (
                      <p className="text-slate-500">Waiting for job activity...</p>
                    ) : (
                      logs.slice(-8).map((entry) => (
                        <div key={entry.timestamp} className="rounded-2xl bg-surface-800/90 p-3">
                          <p className="text-xs text-slate-500">{new Date(entry.timestamp).toLocaleTimeString()}</p>
                          <p className="text-sm text-slate-200">{entry.message}</p>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            ) : null}

            {error ? (
              <div className="rounded-3xl border border-rose-500/30 bg-rose-500/5 p-5 text-sm text-rose-200">
                {error}
              </div>
            ) : null}

            <div className="space-y-4">
              {currentResult ? (
                <div className="space-y-4">
                  <div className="rounded-3xl border border-white/10 bg-surface-800/80 p-6">
                    <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Answer</p>
                    <div className="mt-4 text-base leading-8 text-slate-200">
                      {currentResult.full_report || currentResult.generated_answer ? (
                        <div className="max-h-[60vh] overflow-y-auto pr-4">
                          <AnswerMarkdown content={currentResult.full_report ?? currentResult.generated_answer ?? ""} />
                        </div>
                      ) : (
                        <p>Waiting for the model to produce the report.</p>
                      )}
                    </div>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <Card className="p-5">
                      <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Diagnostics</p>
                      <div className="mt-4 space-y-3 text-sm text-slate-300">
                        <div className="flex justify-between">
                          <span>Retrieved chunks</span>
                          <span>{currentResult.diagnostics?.retrieved_chunk_count ?? 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Evidence objects</span>
                          <span>{currentResult.evidence_objects?.length ?? 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Compression ratio</span>
                          <span>{currentResult.diagnostics?.compression_ratio?.toFixed(2) ?? "—"}</span>
                        </div>
                      </div>
                    </Card>
                    <Card className="p-5">
                      <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Query insights</p>
                      <div className="mt-4 space-y-3 text-sm text-slate-300">
                        <div>
                          <p className="text-slate-200">Intent</p>
                          <p>{currentResult.query_intent?.query_type || "General"}</p>
                        </div>
                        <div>
                          <p className="text-slate-200">Expanded queries</p>
                          <p>{currentResult.expanded_queries?.length ? currentResult.expanded_queries.join(", ") : "—"}</p>
                        </div>
                      </div>
                    </Card>
                  </div>
                </div>
              ) : (
                emptyState
              )}
            </div>
          </Card>

          <Card className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Agent workflow</p>
                <h2 className="text-xl font-semibold text-white">Execution timeline</h2>
              </div>
            </div>
            <div className="space-y-3">
              {agentStates.map((agent) => (
                <div key={agent.name} className="grid gap-3 rounded-3xl border border-white/10 bg-surface-800/80 p-4 sm:grid-cols-[1fr_120px]">
                  <div>
                    <p className="font-medium text-slate-100">{agent.name}</p>
                    <p className="mt-1 text-sm text-slate-400">{agent.status === "running" ? "Active" : agent.status === "completed" ? "Completed" : agent.status === "failed" ? "Failed" : "Pending"}</p>
                  </div>
                  <div className="flex items-center justify-end">
                    <Badge variant={agent.status === "completed" ? "success" : agent.status === "failed" ? "warning" : "muted"}>
                      {agent.status.toUpperCase()}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </section>

        <section className="space-y-6">
          <Card className="space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Retrieved papers</p>
                <h2 className="text-xl font-semibold text-white">Paper sources</h2>
              </div>
              <Badge variant="secondary">Top {paperCards.length}</Badge>
            </div>

            <div className="space-y-4">
              {paperCards.length === 0 ? (
                <p className="text-sm text-slate-400">No papers available yet. Run a query to load results.</p>
              ) : (
                <>
                  <div className="rounded-3xl border border-white/10 bg-surface-900/80 p-6">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Featured paper</p>
                        <p className="mt-1 text-xs text-slate-400">Showing {selectedPaperIndex + 1} of {paperCards.length}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setSelectedPaperIndex((prev) => (prev - 1 + paperCards.length) % paperCards.length)}
                          className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/5 text-white transition hover:bg-white/10"
                        >
                          ←
                        </button>
                        <button
                          type="button"
                          onClick={() => setSelectedPaperIndex((prev) => (prev + 1) % paperCards.length)}
                          className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/5 text-white transition hover:bg-white/10"
                        >
                          →
                        </button>
                      </div>
                    </div>

                    {paperCards[selectedPaperIndex] && (
                      <div className="mt-6 rounded-[2rem] border border-white/10 bg-surface-800/90 p-6 shadow-soft">
                        <p className="text-lg font-semibold text-white">{paperCards[selectedPaperIndex].title}</p>
                        <p className="mt-3 text-sm leading-7 text-slate-300">{paperCards[selectedPaperIndex].authors?.slice(0, 3).join(", ") || "Unknown authors"}</p>
                        <div className="mt-5 flex justify-end">
                          <button
                            type="button"
                            onClick={() => setPaperModalOpen(true)}
                            className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white transition hover:bg-white/10"
                          >
                            View all →
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </Card>

          {paperModalOpen ? (
            <div className="fixed inset-0 z-50 bg-slate-950/95 p-0">
              <div className="flex h-screen w-full flex-col overflow-hidden bg-slate-950/95 shadow-soft">
                <div className="flex flex-col gap-4 border-b border-white/10 px-6 py-5 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Paper details</p>
                    <h3 className="text-2xl font-semibold text-white">Complete paper collection</h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => setPaperModalOpen(false)}
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white transition hover:bg-white/10"
                  >
                    Close
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-6">
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                    {paperCards.map((paper, index) => (
                      <button
                        key={paper.id || `${paper.title}-${index}`}
                        type="button"
                        onClick={() => {
                          setSelectedPaperIndex(index);
                          setPaperModalOpen(false);
                        }}
                        className="rounded-3xl border border-white/10 bg-slate-900/90 p-6 text-left transition hover:bg-slate-800/90"
                      >
                        <p className="text-lg font-semibold text-white">{paper.title}</p>
                        <p className="mt-3 text-sm leading-6 text-slate-400">{paper.authors?.slice(0, 3).join(", ") || "Unknown authors"}</p>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          <Card className="space-y-4">
            <div>
              <p className="text-sm uppercase tracking-[0.24em] text-accent-400">Generated report</p>
              <h2 className="text-xl font-semibold text-white">Summary & references</h2>
            </div>
            {currentResult ? (
              <div className="space-y-4">
                    <div className="rounded-3xl border border-white/10 bg-surface-800/90 p-5 text-slate-200">
                      <div className="max-w-none">
                        <p className="text-sm font-semibold uppercase tracking-[0.12em] text-accent-400">Executive summary</p>
                        <div className="mt-3 max-h-56 overflow-y-auto text-sm text-slate-300">
                          {executivePreview(currentResult) ? (
                            <AnswerMarkdown content={executivePreview(currentResult) as string} />
                          ) : (
                            <p className="text-slate-400">{currentResult.query ? `Research session on ${currentResult.query}` : "Executive summary not available."}</p>
                          )}
                        </div>
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-semibold uppercase tracking-[0.24em] text-accent-400">References</p>
                      <div className="mt-3 grid max-h-64 grid-cols-1 gap-3 overflow-y-auto text-sm">
                        {currentResult.references?.length === 0 ? (
                          <p className="text-slate-500">No references extracted yet.</p>
                        ) : (
                          currentResult.references.map((reference, index) => (
                            <a
                              key={`${reference.title}-${index}`}
                              href={reference.url ?? "#"}
                              target="_blank"
                              rel="noreferrer"
                              className="block rounded-2xl border border-white/5 bg-surface-900/70 p-3 transition hover:border-accent-400"
                            >
                              <p className="font-medium text-white truncate">{reference.title}</p>
                              <p className="mt-1 text-xs text-slate-400">{reference.source || "Unknown source"}</p>
                            </a>
                          ))
                        )}
                      </div>
                    </div>
              </div>
            ) : (
              <p className="text-sm text-slate-400">Your structured report will appear once research results are available.</p>
            )}
          </Card>
        </section>
      </div>
    </div>
  );
}
