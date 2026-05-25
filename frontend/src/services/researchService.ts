import api from "./api";
import type { JobResultResponse, JobStatus, ResearchHistoryItem, ResearchResult } from "../types/api";

export async function startResearchJob(query: string): Promise<{ job_id: string; status: string }> {
  const response = await api.post<{ job_id: string; status: string }>("/api/research/run", { query });
  return response.data;
}

export async function getResearchHistory(): Promise<ResearchHistoryItem[]> {
  const response = await api.get<ResearchHistoryItem[]>("/api/research/history");
  return response.data;
}

export async function getResearchSession(sessionId: string): Promise<ResearchResult | null> {
  const response = await api.get<ResearchResult>(`/api/research/session/${sessionId}`);
  return response.data;
}

export async function getResearchJobStatus(jobId: string): Promise<JobStatus> {
  const response = await api.get<JobStatus>(`/api/research/status/${jobId}`);
  return response.data;
}

export async function getResearchJobResult(jobId: string): Promise<JobResultResponse> {
  const response = await api.get<JobResultResponse>(`/api/research/result/${jobId}`);
  return response.data;
}

export async function cancelResearchJob(jobId: string): Promise<{ job_id: string; status: string }> {
  const response = await api.delete<{ job_id: string; status: string }>(`/api/research/cancel/${jobId}`);
  return response.data;
}
