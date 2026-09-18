import { Children, cloneElement, isValidElement, useId } from 'react';
import type { ReactElement, ReactNode } from 'react';
import { ArrowRight, Inbox, LoaderCircle } from 'lucide-react';
import type { Metrics } from './types';
export const pretty = (text: string) => text.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
export const date = (value?: string) => value ? new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
export function fmt(value: unknown): string {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : Math.abs(value) < .001 && value !== 0 ? value.toExponential(2) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
export function Badge({ children, tone = '' }: { children: ReactNode; tone?: string }) { return <span className={`badge ${tone}`}>{children}</span>; }
export function Status({ value }: { value: string }) { return <Badge tone={['reviewed', 'completed'].includes(value) ? 'green' : ['running', 'draft', 'queued'].includes(value) ? 'amber' : ['error', 'interrupted'].includes(value) ? 'red' : ''}>{pretty(value)}</Badge>; }
export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) { return <div className="empty"><span className="empty-icon"><Inbox size={25} strokeWidth={1.5}/></span><h3>{title}</h3><p>{children}</p>{action}</div>; }
export function Busy({ label = 'Working…' }: { label?: string }) { return <span className="busy"><LoaderCircle size={16} className="spin"/>{label}</span>; }
export function MetricsGrid({ metrics, compact = false }: { metrics?: Metrics; compact?: boolean }) {
  const entries = Object.entries(metrics ?? {});
  if (!entries.length) return <p className="muted small">Metrics appear after a measured update.</p>;
  return <div className={`metrics-grid ${compact ? 'compact' : ''}`}>{entries.map(([key, value]) => key === 'definitions' && typeof value === 'object' && value !== null ? <details className="metric-notes" key={key}><summary>Metric definitions</summary><MetricsGrid metrics={value as Metrics} compact/></details> : Array.isArray(value) ? <div className="metric metric-text" key={key}><span>{pretty(key)}</span><code>{JSON.stringify(value)}</code></div> : typeof value === 'object' && value !== null ? <div className="metric-nested" key={key}><p className="eyebrow">{pretty(key)}</p><MetricsGrid metrics={value as Metrics} compact/></div> : <div className={`metric ${typeof value === 'string' && value.length > 55 ? 'metric-text' : ''}`} key={key}><span>{pretty(key)}</span><strong>{fmt(value)}</strong></div>)}</div>;
}
export function SectionHead({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) { return <div className="section-head"><div><h2>{title}</h2>{children && <p>{children}</p>}</div>{action}</div>; }
export function PageHead({ eyebrow, title, children, action }: { eyebrow: string; title: string; children?: ReactNode; action?: ReactNode }) { return <div className="page-head"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1>{children && <p className="subtitle">{children}</p>}</div>{action}</div>; }
export function TextLink({ children, onClick }: { children: ReactNode; onClick: () => void }) { return <button className="text-link" onClick={onClick}>{children}<ArrowRight size={15}/></button>; }
export function Field({ label, children, hint, className = '' }: { label: string; children: ReactNode; hint?: ReactNode; className?: string }) {
  const labelId = useId(); const hintId = useId();
  const controls = Children.map(children, child => isValidElement(child) && typeof child.type === 'string' && ['input', 'select', 'textarea'].includes(child.type) ? cloneElement(child as ReactElement<Record<string, unknown>>, { 'aria-labelledby': labelId, ...(hint ? { 'aria-describedby': hintId } : {}) }) : child);
  return <label className={`field ${className}`}><span id={labelId}>{label}</span>{controls}{hint && <small id={hintId}>{hint}</small>}</label>;
}
