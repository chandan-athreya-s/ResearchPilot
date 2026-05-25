export interface Paper {
  id: string;
  title: string;
  authors?: string[];
  abstract?: string;
  relevance_score?: number;
  source?: string;
  pdf_url?: string;
  url?: string;
  year?: string;
}

export interface ReferenceItem {
  title: string;
  url?: string;
  source?: string;
}

export interface ResearchDiagnostics {
  retrieval_time?: number;
  reasoning_time?: number;
  paper_count?: number;
  retrieved_chunk_count?: number;
  candidate_chunk_count?: number;
  compression_ratio?: number;
}

export interface ResearchResult {
  query: string;
  query_intent: Record<string, any>;
  papers: Paper[];
  // Full structured report text (long)
  full_report?: string;
  // Short executive summary intended for sidebar display (1-3 concise paragraphs)
  executive_summary?: string;
  // Backwards compatibility: older servers may still return this field
  generated_answer?: string;
  references: ReferenceItem[];
  diagnostics: ResearchDiagnostics;
  errors: string[];
  expanded_queries: string[];
  filtered_papers: Paper[];
  evidence_objects: Record<string, any>[];
  papers_with_extracted_text: string[];
  created_at?: string;
  status?: string;
}

export interface ResearchHistoryItem {
  id: string;
  created_at: string;
  // Original search query that generated this session (optional)
  query?: string;
  // Associated job id if the session originated from a job-based run
  job_id?: string;
  paper_count: number;
  // Short display title for the session
  title: string;
  // Short summary / preview for the session. May be truncated by server; optional.
  summary?: string;
}

export interface JobStatus {
  job_id: string;
  status: "started" | "running" | "completed" | "failed" | "cancelled";
  current_agent?: string;
  progress_percentage: number;
  logs: Array<{ timestamp: string; message: string }>;
  partial_results?: Partial<ResearchResult>;
  error_message?: string;
}

export interface JobResultResponse {
  job_id: string;
  status: "started" | "running" | "completed" | "failed" | "cancelled";
  final_result?: ResearchResult;
  partial_results?: Partial<ResearchResult>;
  progress_percentage?: number;
  current_agent?: string;
  logs?: Array<{ timestamp: string; message: string }>;
  error_message?: string;
}
