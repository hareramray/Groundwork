export type Point = [number, number];
export type Box = [number, number, number, number];
export type ReviewStatus = 'draft' | 'reviewed' | 'excluded';
export interface Element { id: string; class_id: number; label: string; bbox: Box; click_point: Point }
export interface Example { id: string; instruction: string; target_present: boolean; element_id: string | null; status: ReviewStatus; ambiguous: boolean }
export interface Screenshot { id: string; filename: string; width: number; height: number; url: string; group: string; status: string; synthetic: boolean; elements: Element[]; examples: Example[] }
export type Metrics = Record<string, unknown>;
export interface Version { id: string; name: string; created_at: string; classes: string[]; seed: number; group_by: string; stats: Metrics; fingerprint?: string; records?: unknown[] }
export interface Config { image_size: number; batch_size: number; epochs: number; learning_rate: number; seed: number; grad_accum: number; width: number; text_dim: number; max_tokens: number; device: string; mixed_precision: boolean; checkpoint_every: number }
export interface Run { id: string; name: string; version_id: string; mode: string; parent_run_id?: string; source_checkpoint?: string; config: Config; status: string; created_at: string; error?: string; progress?: { epoch?: number; global_step?: number; loss?: number; losses?: Metrics; validation?: Metrics; learning_rate?: number; elapsed_seconds?: number; step_seconds?: number; gpu_memory_mb?: number; checkpoint?: string; [key: string]: unknown }; latest_checkpoint?: string; best_checkpoint?: string }
export interface Checkpoint { name?: string; filename?: string; path?: string; size_bytes?: number; [key: string]: unknown }
export interface ChatModelSource { id: string; name: string; kind: 'grounding' | 'chat'; status: string; latest_checkpoint: string; capabilities: string[] }
export interface ModelExport { id: string; run_id: string; name?: string; created_at?: string; download_url?: string; [key: string]: unknown }
export interface Prediction { image_id?: string; image_url?: string; target_present: boolean; presence_score: number; class_id: number | null; class_name: string | null; bbox: Box | null; click_point: Point | null; bbox_pixels: Box | null; click_point_pixels: Point | null; width: number; height: number; latency_ms: number; memory_mb: number; instruction: string; [key: string]: unknown }
export interface Evaluation { id: string; run_id: string; checkpoint: string; version_id: string; split: string; metrics: Metrics; baselines: Metrics; examples: { id: string; instruction: string; bbox: Box | null; click_point?: Point | null; target_present: boolean; class_id: number | null; prediction: Prediction; success: boolean; iou: number; image_url: string; width: number; height: number }[]; created_at: string }
export interface Validation { errors: unknown[]; warnings: unknown[]; stats: Metrics }
