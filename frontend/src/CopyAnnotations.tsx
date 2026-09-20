import { useEffect, useRef, useState } from 'react';
import { Copy, X } from 'lucide-react';
import { api } from './api';
import { copyAnnotations } from './annotation-copy';
import type { Element, Example, Screenshot } from './types';
import { Badge, Busy, Empty, Field } from './ui';

export function CopyAnnotations({ destinationId, classes, onCopy, onClose }: {
  destinationId: string; classes: string[];
  onCopy: (copied: { elements: Element[]; examples: Example[] }) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const [sources, setSources] = useState<Screenshot[]>([]);
  const [sourceId, setSourceId] = useState('');
  const [source, setSource] = useState<Screenshot | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [includeAbsent, setIncludeAbsent] = useState(true);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingSource, setLoadingSource] = useState(false);
  const [listError, setListError] = useState('');
  const [sourceError, setSourceError] = useState('');
  const [copyError, setCopyError] = useState('');
  const [listRevision, setListRevision] = useState(0);
  const [sourceRevision, setSourceRevision] = useState(0);
  const absentCount = source?.examples.filter(example => !example.target_present).length ?? 0;
  const linkedCount = source?.examples.filter(example => example.target_present && example.element_id && selectedIds.includes(example.element_id)).length ?? 0;
  const instructionCount = linkedCount + (includeAbsent ? absentCount : 0);

  useEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    closeButton.current?.focus();
    return () => { if (element?.open) element.close(); };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setLoadingList(true); setListError(''); setSource(null); setSourceId(''); setSelectedIds([]);
    void api<Screenshot[]>('/images', { signal: controller.signal }).then(images => {
      if (controller.signal.aborted) return;
      const available = images.filter(image => image.id !== destinationId && (image.elements.length || image.examples.length));
      setSources(available); setSourceId(available[0]?.id ?? '');
    }).catch(error => { if (!controller.signal.aborted) setListError((error as Error).message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingList(false); });
    return () => controller.abort();
  }, [destinationId, listRevision]);
  useEffect(() => {
    const controller = new AbortController();
    setSource(null); setSelectedIds([]); setSourceError(''); setCopyError(''); setIncludeAbsent(true);
    if (!sourceId) { setLoadingSource(false); return () => controller.abort(); }
    setLoadingSource(true);
    void api<Screenshot>(`/images/${encodeURIComponent(sourceId)}`, { signal: controller.signal }).then(image => {
      if (controller.signal.aborted) return;
      if (image.id !== sourceId || image.id === destinationId) throw new Error('Choose a different source screenshot.');
      setSource(image); setSelectedIds(image.elements.map(element => element.id));
    }).catch(error => { if (!controller.signal.aborted) setSourceError((error as Error).message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingSource(false); });
    return () => controller.abort();
  }, [sourceId, destinationId, sourceRevision]);

  function add() {
    if (!source || loadingSource || source.id !== sourceId) return;
    try {
      const copied = copyAnnotations(source, { elementIds: selectedIds, includeAbsent });
      if (!copied.elements.length && !copied.examples.length) throw new Error('Select at least one element or absent-target instruction to copy.');
      onCopy(copied);
    } catch (error) { setCopyError((error as Error).message); }
  }

  return <dialog ref={dialog} className="modal copy-annotations-dialog" aria-labelledby="copy-annotations-title" aria-describedby="copy-annotations-description" onCancel={event => { event.preventDefault(); onClose(); }}>
    <div className="section-head"><div><p className="eyebrow">REUSE YOUR ANNOTATIONS</p><h2 id="copy-annotations-title">Copy annotations from another image</h2></div><button ref={closeButton} className="icon-button" aria-label="Close copy dialog" onClick={onClose}><X size={18}/></button></div>
    <p id="copy-annotations-description" className="small muted">Add saved elements and their linked instructions to this screenshot. Your existing annotations stay in place.</p>
    {loadingList ? <div className="copy-loading"><Busy label="Loading source images"/></div> : listError ? <div className="notice error" role="alert"><p>{listError}</p><button className="button compact" onClick={() => setListRevision(value => value + 1)}>Try again</button></div> : sources.length ? <>
      <Field label="Source image"><select value={sourceId} onChange={event => { setSource(null); setSelectedIds([]); setLoadingSource(true); setSourceId(event.target.value); setSourceRevision(value => value + 1); }}>{sources.map(image => <option key={image.id} value={image.id}>{image.filename} · {image.width} × {image.height} · {image.elements.length} elements · {image.examples.length} instructions</option>)}</select></Field>
      {loadingSource ? <div className="copy-loading"><Busy label="Loading saved annotations"/></div> : sourceError ? <div className="notice error" role="alert"><p>{sourceError}</p><button className="button compact" onClick={() => setSourceRevision(value => value + 1)}>Reload source</button></div> : source && <>
        <div className="copy-source-preview"><img src={source.url} alt={`Source screenshot: ${source.filename}`}/><div><strong>{source.filename}</strong><p>{source.width} × {source.height} pixels</p><p>{source.elements.length} elements · {source.examples.length} instructions</p><small>Latest saved annotations</small></div></div>
        <div className="small-section-head"><h3>Elements to copy</h3><Badge>{selectedIds.length} selected</Badge></div>
        {source.elements.length ? <div className="copy-element-list">{source.elements.map((element, index) => {
          const label = element.label || classes[element.class_id] || 'Unknown class';
          const instructions = source.examples.filter(example => example.target_present && example.element_id === element.id).length;
          return <label className="copy-element-option" key={element.id}><input type="checkbox" aria-label={`Copy element ${index + 1}: ${label}`} checked={selectedIds.includes(element.id)} onChange={event => { setCopyError(''); setSelectedIds(previous => event.target.checked ? [...previous, element.id] : previous.filter(value => value !== element.id)); }}/><span><strong>{index + 1}. {label}</strong><small>{classes[element.class_id] || 'Unknown class'} · {instructions} linked {instructions === 1 ? 'instruction' : 'instructions'}</small></span></label>;
        })}</div> : <p className="small muted copy-no-elements">This image has no element boxes to copy.</p>}
        <label className="check-label copy-absent-option"><input type="checkbox" aria-label="Include absent-target instructions" checked={includeAbsent} onChange={event => { setIncludeAbsent(event.target.checked); setCopyError(''); }}/><span>Include absent-target instructions<small>{absentCount} {absentCount === 1 ? 'instruction has' : 'instructions have'} no target box.</small></span></label>
        <div className="notice subtle copy-guidance">Boxes keep their relative positions on the new screenshot. After copying, drag a box or its corners to fit the new layout. All copied instructions start as drafts for your review.</div>
      </>}
    </> : <Empty title="No saved annotations to copy">Save elements or instructions on another screenshot first, then open this dialog again.</Empty>}
    {copyError && <div className="notice error" role="alert">{copyError}</div>}
    <div className="copy-dialog-footer"><p className="small muted">{source && !loadingSource ? `${selectedIds.length} elements and ${instructionCount} instructions will be added. ` : ''}Nothing is saved until you save your changes.</p><div className="row"><button className="button" onClick={onClose}>Cancel</button><button className="button primary" disabled={!source || loadingList || loadingSource || (!selectedIds.length && !instructionCount)} onClick={add}><Copy size={16}/>Add to this image</button></div></div>
  </dialog>;
}
