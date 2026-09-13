export type Tone = 'success' | 'error' | 'warning' | 'active' | 'pending';

export interface RunMetrics {
  final_loss: number | null;
  final_accuracy: number | null;
  gradient_count: number | null;
  database_reaches: number | null;
  total_steps: number | null;
  total_epochs: number | null;
  wall_time_seconds: number | null;
}

export interface Run {
  task_id: string;
  run_name: string | null;
  dataset: string | null;
  model: string | null;
  optimizer: string | null;
  family: string | null;
  suite: string | null;
  state: string;
  state_label: string;
  state_detail: string;
  state_tone: Tone;
  task_status: string;
  artifact_status: string | null;
  artifact_bytes: number | null;
  artifact_files: number | null;
  slurm_job_id: string | null;
  executor: string | null;
  queue_name: string | null;
  seed: number | null;
  stop_condition: Record<string, number> | null;
  submitted_by: string;
  submitter_name: string | null;
  submission_id: string | null;
  created_at: string | null;
  queued_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string | null;
  error_message: string | null;
  runner_version: string | null;
  gpu_model: string | null;
  metrics: RunMetrics | null;
  stop_reason: string | null;
  stop_reason_label: string | null;
  converged: boolean | null;
  can_manage?: boolean;
}

export interface Quantiles { median: number | null; q1: number | null; q3: number | null; min?: number | null }

export interface LeaderboardRow {
  rank: number;
  optimizer: string;
  family: string | null;
  dataset: string | null;
  model: string | null;
  suite: string | null;
  n_runs: number;
  n_seeds: number;
  final_loss: Quantiles;
  final_accuracy: Quantiles;
  gradient_count: { median: number | null };
  database_reaches: { median: number | null };
  stop_reason_mode: string | null;
  task_ids: string[];
  score: number | null;
}

export interface ScoreFormula {
  id: string; label: string; direction: 'asc' | 'desc'; column: string; note: string;
}

export interface FileEntry {
  path: string; name: string; is_dir: boolean;
  size: number; modified: number; preview: string | null;
}

export interface AggregatedSeries {
  label: string;
  family: string | null;
  n_runs: number;
  x: number[];
  median: (number | null)[];
  q1: (number | null)[];
  q3: (number | null)[];
  n_at_x: number[];
  full_until_index: number;
}

export interface User {
  id: string; email: string; role: string; display_name: string | null;
  is_admin: boolean; is_verified: boolean; has_join_info: boolean;
}

export interface Vocabulary {
  metrics: Record<string, { label: string; short: string; hint: string }>;
  stop_reasons: Record<string, { label: string; note: string; converged: boolean }>;
  run_states: Record<string, { label: string; detail: string; tone: Tone }>;
}
