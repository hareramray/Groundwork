import { useEffect, useRef, useState } from 'react';
import type { PointerEvent, WheelEvent } from 'react';
import { Check, ChevronLeft, ChevronRight, Crosshair, Hand, MousePointer2, Plus, Save, Scan, SquareDashed, Trash2, X, ZoomIn, ZoomOut } from 'lucide-react';
import { api } from './api';
import { annotationErrors, center, clamp, moveElement, orderBox, resizeElement } from './coordinates';
import type { Element, Example, Point, Screenshot } from './types';
import { Badge, Busy, Field, Status } from './ui';

type Tool = 'select' | 'draw' | 'pan' | 'point';
type Drag = { kind: 'draw' | 'move' | 'resize' | 'pan' | 'point'; start: Point; original?: Element; corner?: number; pan?: Point };
const id = () => crypto.randomUUID();
const colors = ['#167d6f', '#de9840', '#637ad4', '#c15d83', '#9771bf', '#5086aa'];

export default function Annotation({ image, classes, index, total, onClose, onNavigate, onSaved, notify }: { image: Screenshot; classes: string[]; index: number; total: number; onClose: () => void; onNavigate: (index: number) => void; onSaved: (image: Screenshot) => void; notify: (text: string, error?: boolean) => void }) {
  const [draft, setDraft] = useState<Screenshot>(structuredClone(image));
  const [selected, setSelected] = useState<string | null>(image.elements[0]?.id ?? null);
  const [tool, setTool] = useState<Tool>('select');
  const [panel, setPanel] = useState('elements');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>([0, 0]);
  const [box, setBox] = useState<Element['bbox'] | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const svg = useRef<SVGSVGElement>(null);
  const group = useRef<SVGGElement>(null);
  const drag = useRef<Drag | null>(null);
  const holdingSpace = useRef(false);
  const active = draft.elements.find(e => e.id === selected);
  useEffect(() => { const previous = document.body.style.overflow; document.body.style.overflow = 'hidden'; return () => { document.body.style.overflow = previous; }; }, []);
  useEffect(() => { setDraft(structuredClone(image)); setSelected(image.elements[0]?.id ?? null); setDirty(false); setErrors([]); setZoom(1); setPan([0, 0]); }, [image.id]);
  const update = (fn: (value: Screenshot) => Screenshot) => { setDraft(fn); setDirty(true); setErrors([]); };
  function editElement(element: Element) { update(d => ({ ...d, elements: d.elements.map(e => e.id === element.id ? element : e), examples: d.examples.map(e => e.element_id === element.id && e.status === 'reviewed' ? { ...e, status: 'draft' } : e) })); }
  function deleteElement() { if (selected) { update(d => ({ ...d, elements: d.elements.filter(e => e.id !== selected), examples: d.examples.map(e => e.element_id === selected ? { ...e, element_id: null, status: 'draft' } : e) })); setSelected(null); } }
  function editExample(example: Example, changes: Partial<Example>) { update(d => ({ ...d, examples: d.examples.map(e => e.id === example.id ? { ...example, ...changes, status: changes.status ?? 'draft' } : e) })); }
  function addExample(absent = false) { update(d => ({ ...d, examples: [...d.examples, { id: id(), instruction: '', target_present: !absent, element_id: absent ? null : selected, status: 'draft', ambiguous: false }] })); setPanel('instructions'); }
  async function save(andNext = false) {
    const found = annotationErrors(draft, classes); setErrors(found); if (found.length) return;
    setSaving(true);
    try { const saved = await api<Screenshot>(`/images/${draft.id}`, { method: 'PUT', body: JSON.stringify({ elements: draft.elements, examples: draft.examples, group: draft.group }) }); setDraft(saved); setDirty(false); onSaved(saved); notify('Annotations saved locally.'); if (andNext && index < total - 1) onNavigate(index + 1); }
    catch (error) { notify((error as Error).message, true); } finally { setSaving(false); }
  }
  function navigate(next: number) { if (!dirty || window.confirm('Leave this screenshot and discard unsaved changes?')) onNavigate(next); }
  function close() { if (!dirty || window.confirm('Close the workspace and discard unsaved changes?')) onClose(); }
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.closest('input,textarea,select,[contenteditable=true]')) return;
      if (event.code === 'Space') { event.preventDefault(); holdingSpace.current = true; }
      if ((event.ctrlKey || event.metaKey) && event.key === 's') { event.preventDefault(); void save(); }
      if (event.key === 'v') setTool('select'); if (event.key === 'd') setTool('draw'); if (event.key === 'h') setTool('pan');
      if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); deleteElement(); }
      if (event.key === 'ArrowRight' && index < total - 1) navigate(index + 1);
      if (event.key === 'ArrowLeft' && index > 0) navigate(index - 1);
      if (event.key === 'Escape') { setTool('select'); setSelected(null); }
    };
    const up = (event: KeyboardEvent) => { if (event.code === 'Space') holdingSpace.current = false; };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  });
  function point(event: { clientX: number; clientY: number }, outer = false): Point {
    const matrix = (outer ? svg.current : group.current)?.getScreenCTM();
    if (!matrix) return [0, 0];
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return outer ? [p.x, p.y] : [p.x / draft.width, p.y / draft.height];
  }
  function start(event: PointerEvent<SVGElement>, element?: Element, corner?: number, clickPoint = false) {
    if (event.button !== 0 && event.button !== 1) return;
    event.preventDefault(); event.stopPropagation(); svg.current?.setPointerCapture(event.pointerId);
    if (tool === 'pan' || holdingSpace.current || event.button === 1) { drag.current = { kind: 'pan', start: point(event, true), pan }; return; }
    const p = point(event); const normalized: Point = [clamp(p[0]), clamp(p[1])];
    if (tool === 'draw') { drag.current = { kind: 'draw', start: normalized }; setBox([...normalized, ...normalized] as Element['bbox']); return; }
    if (tool === 'point' && active) { const click: Point = [clamp(p[0], active.bbox[0], active.bbox[2]), clamp(p[1], active.bbox[1], active.bbox[3])]; editElement({ ...active, click_point: click }); drag.current = { kind: 'point', start: p, original: active }; return; }
    if (element) { setSelected(element.id); drag.current = { kind: clickPoint ? 'point' : corner !== undefined ? 'resize' : 'move', start: p, original: element, corner }; }
    else setSelected(null);
  }
  function move(event: PointerEvent<SVGSVGElement>) {
    const d = drag.current; if (!d) return;
    if (d.kind === 'pan') { const p = point(event, true); setPan([(d.pan?.[0] ?? 0) + p[0] - d.start[0], (d.pan?.[1] ?? 0) + p[1] - d.start[1]]); return; }
    const p = point(event);
    if (d.kind === 'draw') { setBox(orderBox(d.start, [clamp(p[0]), clamp(p[1])])); return; }
    if (!d.original) return;
    if (d.kind === 'move') editElement(moveElement(d.original, p[0] - d.start[0], p[1] - d.start[1]));
    if (d.kind === 'resize') editElement(resizeElement(d.original, d.corner!, p));
    if (d.kind === 'point') editElement({ ...d.original, click_point: [clamp(p[0], d.original.bbox[0], d.original.bbox[2]), clamp(p[1], d.original.bbox[1], d.original.bbox[3])] });
  }
  function end(event: PointerEvent<SVGSVGElement>) {
    if (drag.current?.kind === 'draw' && box && (box[2] - box[0]) * draft.width > 2 && (box[3] - box[1]) * draft.height > 2) {
      const element = { id: id(), class_id: active?.class_id ?? 0, label: '', bbox: box, click_point: center(box) };
      update(d => ({ ...d, elements: [...d.elements, element] })); setSelected(element.id); setTool('select'); setPanel('elements');
    }
    drag.current = null; setBox(null); if (svg.current?.hasPointerCapture(event.pointerId)) svg.current.releasePointerCapture(event.pointerId);
  }
  function zoomTo(value: number, anchor: Point = [draft.width / 2, draft.height / 2]) { const next = clamp(value, .3, 8); setPan([anchor[0] - ((anchor[0] - pan[0]) / zoom) * next, anchor[1] - ((anchor[1] - pan[1]) / zoom) * next]); setZoom(next); }
  function wheel(event: WheelEvent<SVGSVGElement>) { if (event.ctrlKey || event.metaKey) event.preventDefault(); zoomTo(zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1), point(event, true)); }
  const scale = Math.max(draft.width / 1100, 1) / zoom;
  return <div className="annotation-screen">
    <div className="annotation-header"><button className="icon-button" onClick={close} title="Close workspace"><X size={19}/></button><div className="grow"><h2>{draft.filename}</h2><span className="small muted">{draft.width} × {draft.height} px <span className="dot-separator">·</span> {index + 1} of {total}</span></div>{draft.synthetic && <Badge tone="purple">Synthetic</Badge>}<Status value={dirty ? 'draft' : draft.status}/><button className="button primary" onClick={() => void save()} disabled={saving}>{saving ? <Busy label="Saving"/> : <><Save size={16}/>{dirty ? 'Save changes' : 'Save annotations'}</>}</button><div className="button-group"><button className="icon-button" title="Previous image (←)" onClick={() => navigate(index - 1)} disabled={index === 0}><ChevronLeft size={18}/></button><button className="icon-button" title="Next image (→)" onClick={() => navigate(index + 1)} disabled={index === total - 1}><ChevronRight size={18}/></button></div></div>
    <div className="annotation-layout"><div className="canvas-column"><div className="canvas-toolbar"><div className="button-group">{([{ value: 'select', icon: MousePointer2, title: 'Select / move (V)' }, { value: 'draw', icon: SquareDashed, title: 'Draw target (D)' }, { value: 'pan', icon: Hand, title: 'Pan (H or hold Space)' }, { value: 'point', icon: Crosshair, title: 'Adjust click point' }] as const).map(t => <button key={t.value} title={t.title} className={`icon-button ${tool === t.value ? 'active' : ''}`} onClick={() => setTool(t.value)} disabled={t.value === 'point' && !active}><t.icon size={18}/></button>)}</div><span className="tool-label">{tool === 'draw' ? 'Drag to draw a target box' : tool === 'point' ? 'Click inside the box to adjust the click point' : tool === 'pan' ? 'Drag to pan your screenshot' : 'Select a box • drag its corners to resize'}</span><div className="button-group"><button className="icon-button" title="Zoom out" onClick={() => zoomTo(zoom / 1.25)}><ZoomOut size={17}/></button><span className="zoom-label">{Math.round(zoom * 100)}%</span><button className="icon-button" title="Zoom in" onClick={() => zoomTo(zoom * 1.25)}><ZoomIn size={17}/></button><button className="icon-button" title="Fit screenshot" onClick={() => { setZoom(1); setPan([0, 0]); }}><Scan size={17}/></button></div></div>
      <div className={`annotation-canvas tool-${tool}`}><svg ref={svg} viewBox={`0 0 ${draft.width} ${draft.height}`} onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={() => { drag.current = null; setBox(null); }} onWheel={wheel} role="img" aria-label="Interactive screenshot annotation canvas"><g ref={group} transform={`translate(${pan[0]} ${pan[1]}) scale(${zoom})`}><image href={draft.url} width={draft.width} height={draft.height}/>{draft.elements.map((e, i) => { const b = e.bbox; const chosen = selected === e.id; const color = colors[e.class_id % colors.length]; const x = b[0] * draft.width, y = b[1] * draft.height, w = (b[2] - b[0]) * draft.width, h = (b[3] - b[1]) * draft.height; return <g key={e.id}><rect x={x} y={y} width={w} height={h} fill={color} fillOpacity={chosen ? .17 : .06} stroke={color} strokeWidth={(chosen ? 2.7 : 1.5) * scale} onPointerDown={event => start(event, e)} style={{ cursor: 'move' }}/><g pointerEvents="none"><rect x={x} y={Math.max(0, y - 21 * scale)} width={Math.min(140, Math.max(66, (classes[e.class_id]?.length ?? 0) * 8 + 29)) * scale} height={21 * scale} rx={3 * scale} fill={color}/><text x={x + 6 * scale} y={Math.max(15 * scale, y - 6 * scale)} fill="white" fontSize={12 * scale} fontFamily="sans-serif">{i + 1} · {classes[e.class_id]}</text></g>{chosen && [[x, y], [x + w, y], [x + w, y + h], [x, y + h]].map(([cx, cy], corner) => <rect key={corner} x={cx - 4.5 * scale} y={cy - 4.5 * scale} width={9 * scale} height={9 * scale} fill="white" stroke={color} strokeWidth={1.5 * scale} onPointerDown={event => start(event, e, corner)} style={{ cursor: corner % 2 ? 'nesw-resize' : 'nwse-resize' }}/>)}<circle cx={e.click_point[0] * draft.width} cy={e.click_point[1] * draft.height} r={(chosen ? 6 : 3.5) * scale} fill={color} stroke="white" strokeWidth={1.7 * scale} onPointerDown={event => start(event, e, undefined, true)} style={{ cursor: 'crosshair' }}/></g>; })}{box && <rect x={box[0] * draft.width} y={box[1] * draft.height} width={(box[2] - box[0]) * draft.width} height={(box[3] - box[1]) * draft.height} fill="#167d6f" fillOpacity={.12} stroke="#167d6f" strokeWidth={2 * scale} strokeDasharray={`${6 * scale} ${4 * scale}`} pointerEvents="none"/>}</g></svg></div>
      <div className="canvas-footer"><span><span className="status-dot"/>Original aspect ratio · normalized coordinates</span><span><kbd>D</kbd> draw <kbd>V</kbd> select <kbd>Space</kbd> pan <kbd>←</kbd><kbd>→</kbd> images</span></div>
    </div><aside className="annotation-properties"><div className="tab-strip"><button className={panel === 'elements' ? 'selected' : ''} onClick={() => setPanel('elements')}>1. Elements <span>{draft.elements.length}</span></button><button className={panel === 'instructions' ? 'selected' : ''} onClick={() => setPanel('instructions')}>2. Instructions <span>{draft.examples.length}</span></button></div><div className="properties-scroll">
      <Field label="Website / template / session group" hint="Related screenshots stay together when splitting by group."><input value={draft.group ?? ''} onChange={event => update(d => ({ ...d, group: event.target.value }))} placeholder="e.g. shop-checkout"/></Field>
      {errors.length > 0 && <div className="notice error"><strong>Check your annotations</strong>{errors.map((error, i) => <p key={i}>{error}</p>)}</div>}
      {panel === 'elements' ? <><div className="small-section-head"><h3>Visible elements</h3><button className="text-link" onClick={() => setTool('draw')}><Plus size={15}/>Draw</button></div><p className="small muted">Box each target, then associate it with an instruction.</p><div className="element-list">{draft.elements.map((e, i) => <button key={e.id} className={`element-item ${selected === e.id ? 'selected' : ''}`} onClick={() => { setSelected(e.id); setTool('select'); }}><span className="element-number" style={{ background: colors[e.class_id % colors.length] }}>{i + 1}</span><span><strong>{e.label || classes[e.class_id] || 'Unknown class'}</strong><small>{classes[e.class_id]}</small></span><MousePointer2 size={14}/></button>)}{!draft.elements.length && <div className="mini-empty">Choose the draw tool, then drag over a visible element.</div>}</div>
      {active && <div className="element-details"><div className="small-section-head"><h3>Target properties</h3><button className="icon-button danger" title="Delete selected element" onClick={deleteElement}><Trash2 size={16}/></button></div><Field label="Element class"><select value={active.class_id} onChange={event => editElement({ ...active, class_id: Number(event.target.value) })}>{classes.map((c, i) => <option key={c} value={i}>{c}</option>)}</select></Field><Field label="Visible label (optional)"><input value={active.label} onChange={event => editElement({ ...active, label: event.target.value })} placeholder="e.g. Search products"/></Field><p className="eyebrow">Normalized bounding box</p><div className="coordinate-inputs">{['X min', 'Y min', 'X max', 'Y max'].map((label, i) => <Field key={label} label={label}><input type="number" min="0" max="1" step="0.001" value={Number(active.bbox[i].toFixed(5))} onChange={event => { const bbox = [...active.bbox] as Element['bbox']; bbox[i] = Number(event.target.value); editElement({ ...active, bbox }); }}/></Field>)}</div><p className="small mono muted">Pixels: {active.bbox.map((v, i) => Math.round(v * (i % 2 ? draft.height : draft.width))).join(', ')}</p><div className="small-section-head"><h3>Candidate click point</h3><button className="text-link" onClick={() => editElement({ ...active, click_point: center(active.bbox) })}>Center</button></div><div className="form-grid two">{['X', 'Y'].map((label, i) => <Field key={label} label={label}><input type="number" min="0" max="1" step="0.001" value={Number(active.click_point[i].toFixed(5))} onChange={event => { const click_point = [...active.click_point] as Point; click_point[i] = Number(event.target.value); editElement({ ...active, click_point }); }}/></Field>)}</div><button className="button full" onClick={() => addExample()}><Plus size={16}/>Add instruction for this target</button></div>}</> : <><div className="small-section-head"><h3>Instruction–target pairs</h3><button className="icon-button" title="Add instruction" onClick={() => addExample()}><Plus size={17}/></button></div><div className="notice subtle">Only explicitly reviewed, unambiguous examples can enter a dataset version. Changes return affected examples to draft.</div>{draft.examples.map((example, i) => <div className="instruction-card" key={example.id}><div className="small-section-head"><span className="eyebrow">Example {i + 1}</span><Status value={example.status}/><button className="icon-button danger" title="Delete example" onClick={() => update(d => ({ ...d, examples: d.examples.filter(e => e.id !== example.id) }))}><Trash2 size={14}/></button></div><Field label="Instruction"><textarea rows={2} value={example.instruction} onChange={event => editExample(example, { instruction: event.target.value })} placeholder="Find the search field"/></Field><Field label="Matching target"><select value={example.target_present ? example.element_id ?? '' : 'absent'} onChange={event => { const absent = event.target.value === 'absent'; editExample(example, { target_present: !absent, element_id: absent ? null : event.target.value || null }); if (!absent) setSelected(event.target.value); }}><option value="">Select an element…</option>{draft.elements.map((e, j) => <option key={e.id} value={e.id}>{j + 1}. {e.label || classes[e.class_id]}</option>)}<option value="absent">Target is absent (no box)</option></select></Field><label className="check-label"><input type="checkbox" checked={example.ambiguous} onChange={event => editExample(example, { ambiguous: event.target.checked })}/>Ambiguous — needs review</label><Field label="Review status"><select value={example.status} onChange={event => editExample(example, { status: event.target.value as Example['status'] })}><option value="draft">Draft</option><option value="reviewed" disabled={example.ambiguous}>Reviewed</option><option value="excluded">Excluded</option></select></Field></div>)}<div className="stacked-actions"><button className="button" onClick={() => addExample()}><Plus size={15}/>Add instruction</button><button className="button ghost" onClick={() => addExample(true)}>Add absent-target example</button>{draft.examples.some(e => e.status === 'draft' && !e.ambiguous) && <button className="button" onClick={() => { const found = annotationErrors(draft, classes); if (found.length) { setErrors(found); return; } update(d => ({ ...d, examples: d.examples.map(e => e.status === 'draft' && !e.ambiguous ? { ...e, status: 'reviewed' } : e) })); }}><Check size={16}/>Mark unambiguous drafts reviewed</button>}</div></>}
      </div><div className="properties-footer"><span className="small muted">{dirty ? 'Unsaved changes' : 'All changes saved locally'}</span><button className="button primary full" onClick={() => void save(true)} disabled={saving}>{saving ? <Busy/> : <>Save{index < total - 1 ? ' & next image' : ' annotations'}<ChevronRight size={16}/></>}</button></div></aside></div>
  </div>;
}
