import { useEffect, useRef, useState } from 'react';
import { ArrowDownToLine, MessageCircle, Pencil, Play, Plus, RotateCcw, Save, Send, Square, Trash2, X } from 'lucide-react';
import { api, post } from './api';
import type { ChatModelSource } from './types';
import { Badge, Busy, date, Empty, Field, fmt, PageHead, SectionHead, Status } from './ui';
import './ChatTraining.css';

type Notify = (text: string, error?: boolean) => void;
type ChatExample = { id: string; prompt: string; response: string; created_at: string; updated_at: string };
type ChatConfig = { epochs: number; batch_size: number; learning_rate: number; device: string; seed: number };
type ChatRun = {
  id: string; name: string; config: ChatConfig; status: string; created_at: string; example_count: number;
  progress?: { epoch?: number; global_step?: number; loss?: number; elapsed_seconds?: number };
  error?: string; latest_checkpoint?: string; source_run_id?: string; source_chat_run_id?: string;
  capabilities?: string[]; architecture?: string;
};
type Prediction = { reply: string; run_id: string; checkpoint: string; unknown_characters?: number };
type TestReply = Prediction & { prompt: string; name: string; id: number };
const isUnified = (run: ChatRun) => Boolean(run.capabilities?.includes('grounding') && run.capabilities.includes('chat'));

function elapsed(seconds?: number) {
  if (seconds === undefined) return '—';
  return seconds >= 60 ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s` : `${seconds.toFixed(1)} s`;
}

export function ChatTraining({ notify }: { notify: Notify }) {
  const [examples, setExamples] = useState<ChatExample[]>([]);
  const [runs, setRuns] = useState<ChatRun[]>([]);
  const [sources, setSources] = useState<ChatModelSource[]>([]);
  const [sourceKey, setSourceKey] = useState('');
  const [config, setConfig] = useState<ChatConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [pollError, setPollError] = useState('');
  const [revision, setRevision] = useState(0);
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [editing, setEditing] = useState('');
  const [deleting, setDeleting] = useState('');
  const [exampleBusy, setExampleBusy] = useState('');
  const [name, setName] = useState('');
  const [selected, setSelected] = useState('');
  const [runBusy, setRunBusy] = useState('');
  const [testRunId, setTestRunId] = useState('');
  const [message, setMessage] = useState('');
  const [replies, setReplies] = useState<TestReply[]>([]);
  const [testing, setTesting] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const mounted = useRef(true);
  const run = runs.find(item => item.id === selected) ?? runs[0];
  const source = sources.find(item => `${item.kind}:${item.id}` === sourceKey);
  const availableModels = runs.filter(item => item.latest_checkpoint && isUnified(item));
  const testRun = availableModels.find(item => item.id === testRunId) ?? availableModels[0];
  function sourceName(item: ChatRun) {
    const id = item.source_chat_run_id || item.source_run_id;
    return sources.find(value => value.id === id)?.name ?? runs.find(value => value.id === id)?.name ?? id ?? 'Unknown source';
  }

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setLoadError('');
    void Promise.all([
      api<ChatExample[]>('/chat/examples', { signal: controller.signal }),
      api<ChatRun[]>('/chat/runs', { signal: controller.signal }),
      api<ChatConfig>('/chat/training/defaults', { signal: controller.signal }),
      api<ChatModelSource[]>('/chat/sources', { signal: controller.signal }),
    ]).then(([savedExamples, savedRuns, defaults, savedSources]) => {
      if (controller.signal.aborted) return;
      setExamples(savedExamples); setRuns(savedRuns); setConfig(defaults); setSources(savedSources);
    }).catch(error => {
      if (!controller.signal.aborted) { setLoadError((error as Error).message); notify((error as Error).message, true); }
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [revision, notify]);

  useEffect(() => {
    if (loading || loadError) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const [savedRuns, savedSources] = await Promise.all([
          api<ChatRun[]>('/chat/runs', { signal: controller.signal }),
          api<ChatModelSource[]>('/chat/sources', { signal: controller.signal }),
        ]);
        if (!controller.signal.aborted) { setRuns(savedRuns); setSources(savedSources); setPollError(''); }
      } catch (error) {
        if (!controller.signal.aborted) setPollError((error as Error).message);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(() => void poll(), 2000);
      }
    }
    timer = setTimeout(() => void poll(), 2000);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [loading, loadError, revision]);

  function clearEditor() { setEditing(''); setPrompt(''); setResponse(''); }
  function editExample(example: ChatExample) {
    setEditing(example.id); setPrompt(example.prompt); setResponse(example.response); setDeleting('');
    editorRef.current?.focus(); editorRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
  async function saveExample() {
    if (exampleBusy || !prompt.trim() || !response.trim()) return;
    setExampleBusy('save');
    try {
      const body = { prompt: prompt.trim(), response: response.trim() };
      const saved = editing
        ? await api<ChatExample>(`/chat/examples/${editing}`, { method: 'PUT', body: JSON.stringify(body) })
        : await post<ChatExample>('/chat/examples', body);
      if (!mounted.current) return;
      setExamples(previous => editing ? previous.map(item => item.id === saved.id ? saved : item) : [saved, ...previous]);
      clearEditor(); notify(editing ? 'Chat example updated. Use it in your next training run.' : 'Chat example saved.');
    } catch (error) { if (mounted.current) notify((error as Error).message, true); }
    finally { if (mounted.current) setExampleBusy(''); }
  }
  async function deleteExample(id: string) {
    if (exampleBusy) return;
    setExampleBusy(id);
    try {
      await api(`/chat/examples/${id}`, { method: 'DELETE' });
      if (!mounted.current) return;
      setExamples(previous => previous.filter(item => item.id !== id)); setDeleting('');
      if (editing === id) clearEditor();
      notify('Chat example deleted. Existing runs keep their saved training data.');
    } catch (error) { if (mounted.current) notify((error as Error).message, true); }
    finally { if (mounted.current) setExampleBusy(''); }
  }
  async function start() {
    if (!config || !source || !examples.length || runBusy || exampleBusy) return;
    setRunBusy('start');
    try {
      const result = await post<ChatRun>('/chat/runs', {
        name: name.trim() || `Chat experiment ${runs.length + 1}`, config, source_checkpoint: 'latest',
        ...(source.kind === 'chat' ? { source_chat_run_id: source.id } : { source_run_id: source.id }),
      });
      if (!mounted.current) return;
      setRuns(previous => [result, ...previous.filter(item => item.id !== result.id)]); setSelected(result.id); setName('');
      notify(`Chat training started from ${source.name}. The saved model will support grounding and chat.`);
    } catch (error) { if (mounted.current) notify((error as Error).message, true); }
    finally { if (mounted.current) setRunBusy(''); }
  }
  async function control(action: 'stop' | 'resume') {
    if (!run || runBusy || (action === 'resume' && !isUnified(run))) return;
    setRunBusy(action);
    try {
      await post(`/chat/runs/${run.id}/${action}`);
      const updated = await api<ChatRun[]>('/chat/runs');
      if (!mounted.current) return;
      setRuns(updated); notify(action === 'stop' ? 'Stop requested. The worker will save its training state.' : 'Resuming this chat experiment with its saved examples and settings.');
    } catch (error) { if (mounted.current) notify((error as Error).message, true); }
    finally { if (mounted.current) setRunBusy(''); }
  }
  async function downloadModel() {
    if (!run?.latest_checkpoint || downloading) return;
    setDownloading(true);
    try {
      const result = await fetch(`/api/chat/runs/${run.id}/download`);
      if (!result.ok) {
        const detail = await result.json().catch(() => ({})) as { detail?: string };
        throw new Error(detail.detail || `Download failed (${result.status}).`);
      }
      const url = URL.createObjectURL(await result.blob());
      const link = document.createElement('a'); link.href = url; link.download = `chat-${run.id}.pt`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { if (mounted.current) notify((error as Error).message, true); }
    finally { if (mounted.current) setDownloading(false); }
  }
  async function predict() {
    if (!testRun || !message.trim() || testing) return;
    const sent = message.trim();
    setTesting(true);
    try {
      const result = await post<Prediction>('/chat/predict', { run_id: testRun.id, message: sent });
      if (!mounted.current) return;
      setReplies(previous => [...previous, { ...result, prompt: sent, name: testRun.name, id: Date.now() }]); setMessage('');
    } catch (error) { if (mounted.current) notify((error as Error).message, true); }
    finally { if (mounted.current) setTesting(false); }
  }
  function teachReply(value: string) {
    setEditing(''); setPrompt(value); setResponse('');
    editorRef.current?.focus(); editorRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
  function changeConfig(key: keyof ChatConfig, value: number | string) { setConfig(previous => previous ? { ...previous, [key]: value } : previous); }

  return <>
    <PageHead eyebrow="YOUR WORDS, YOUR MODEL" title="Chat training" action={<Badge tone="green"><MessageCircle size={13}/>Grounding + chat</Badge>}>Teach your grounding model to chat using messages and replies you write yourself.</PageHead>
    {loading ? <section className="card"><div className="card-padding"><Busy label="Loading chat training"/></div></section> : loadError ? <div className="notice error" role="alert"><strong>Could not load chat training</strong><p>{loadError}</p><button className="button compact" onClick={() => setRevision(value => value + 1)}>Try again</button></div> : <>
      <div className="chat-intro"><span className="mode-icon"><MessageCircle size={21}/></span><div><strong>One model for grounding and chat</strong><p>Add a message such as “hey” and your preferred reply, then choose your saved grounding model below. Training adds chat to a new version of that model, with both abilities saved in the same file. You can test its replies here and use it on the Prediction page.</p></div></div>
      <div className="chat-teaching-layout">
        <section className="card chat-example-editor">
          <SectionHead title={editing ? 'Edit your example' : '1. Write a chat example'} action={editing ? <Badge>Editing</Badge> : undefined}>You provide both sides. Nothing is added automatically.</SectionHead>
          <form className="card-padding" onSubmit={event => { event.preventDefault(); void saveExample(); }}>
            <Field label="User message"><textarea ref={editorRef} value={prompt} onChange={event => setPrompt(event.target.value)} placeholder="hey" rows={3} required maxLength={500} disabled={Boolean(exampleBusy)}/></Field>
            <Field label="Your preferred reply"><textarea value={response} onChange={event => setResponse(event.target.value)} placeholder="Hey! How can I help?" rows={4} required maxLength={500} disabled={Boolean(exampleBusy)}/></Field>
            <div className="chat-actions"><button className="button primary" type="submit" disabled={Boolean(exampleBusy) || !prompt.trim() || !response.trim()}>{exampleBusy === 'save' ? <Busy label="Saving"/> : <><Save size={16}/>{editing ? 'Update example' : 'Save example'}</>}</button>{(editing || prompt || response) && <button className="button" type="button" disabled={Boolean(exampleBusy)} onClick={clearEditor}><X size={15}/>{editing ? 'Cancel edit' : 'Clear'}</button>}</div>
          </form>
        </section>
        <section className="card chat-examples">
          <SectionHead title="Your saved examples" action={<Badge>{examples.length} {examples.length === 1 ? 'example' : 'examples'}</Badge>}>Every new run learns from all examples saved here.</SectionHead>
          {examples.length ? <div className="chat-example-list">{examples.map((example, index) => <article className={`chat-example ${editing === example.id ? 'selected' : ''}`} key={example.id}>
            <div className="row space-between"><span className="eyebrow">EXAMPLE {index + 1}</span><div className="row"><button type="button" className="icon-button" aria-label={`Edit example ${index + 1}`} title="Edit example" disabled={Boolean(exampleBusy)} onClick={() => editExample(example)}><Pencil size={15}/></button><button type="button" className="icon-button danger" aria-label={`Delete example ${index + 1}`} title="Delete example" disabled={Boolean(exampleBusy)} onClick={() => setDeleting(example.id)}><Trash2 size={15}/></button></div></div>
            <dl className="chat-example-pair"><div><dt>User</dt><dd>{example.prompt}</dd></div><div><dt>Reply</dt><dd>{example.response}</dd></div></dl>
            {deleting === example.id && <div className="chat-delete-confirm"><p>Delete this example? Existing runs keep their saved copy.</p><div className="chat-actions"><button className="button compact danger-solid" type="button" disabled={Boolean(exampleBusy)} onClick={() => void deleteExample(example.id)}>{exampleBusy === example.id ? <Busy label="Deleting"/> : 'Delete example'}</button><button className="button compact" type="button" disabled={Boolean(exampleBusy)} onClick={() => setDeleting('')}>Keep example</button></div></div>}
          </article>)}</div> : <Empty title="Start with your first exchange">Write a user message and the reply you want your model to learn, then save it.</Empty>}
        </section>
      </div>
      <section className="card chat-config">
        <SectionHead title="2. Train your grounding model for chat" action={<Badge>{examples.length} saved {examples.length === 1 ? 'example' : 'examples'}</Badge>}>A new version of your selected model learns your replies while retaining its grounding weights.</SectionHead>
        {config && <form className="card-padding" onSubmit={event => { event.preventDefault(); void start(); }}>
          <Field label="Grounding model to train for chat" hint="Uses this model's latest saved checkpoint. Select a grounding + chat version to continue teaching it."><select required value={source ? sourceKey : ''} onChange={event => setSourceKey(event.target.value)} disabled={Boolean(runBusy) || !sources.length}><option value="">Select a saved model…</option>{sources.map(item => <option key={`${item.kind}:${item.id}`} value={`${item.kind}:${item.id}`}>{item.name} · {item.kind === 'chat' ? 'Grounding + chat' : 'Grounding'}</option>)}</select></Field>
          {!sources.length && <div className="notice subtle"><p>Train a grounding model in the Grounding tab first. Once it saves a checkpoint, you can select it here and teach it your chat examples.</p></div>}
          <div className="form-grid three"><Field label="Chat experiment name"><input value={name} maxLength={120} onChange={event => setName(event.target.value)} placeholder={`Chat experiment ${runs.length + 1}`} disabled={Boolean(runBusy)}/></Field><Field label="Epochs" hint="How many passes through your examples."><input type="number" min={1} max={10000} step={1} required value={config.epochs} onChange={event => changeConfig('epochs', Number(event.target.value))} disabled={Boolean(runBusy)}/></Field><Field label="Learning rate"><input type="number" min={0.000001} max={1} step="any" required value={config.learning_rate} onChange={event => changeConfig('learning_rate', Number(event.target.value))} disabled={Boolean(runBusy)}/></Field></div>
          <details className="detail-block"><summary>Training settings</summary><div className="form-grid three"><Field label="Batch size"><input type="number" min={1} max={256} step={1} required value={config.batch_size} onChange={event => changeConfig('batch_size', Number(event.target.value))} disabled={Boolean(runBusy)}/></Field><Field label="Compute device"><select value={config.device} onChange={event => changeConfig('device', event.target.value)} disabled={Boolean(runBusy)}><option value="auto">Automatic (CUDA if available)</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></Field><Field label="Random seed"><input type="number" min={0} max={2147483647} step={1} required value={config.seed} onChange={event => changeConfig('seed', Number(event.target.value))} disabled={Boolean(runBusy)}/></Field></div></details>
          <div className="chat-start-row"><p className="small muted">{!examples.length ? 'Save at least one example above to enable training.' : !source ? 'Select the saved grounding model you want to teach.' : 'This run saves a copy of your selected model and all current examples. Later edits apply to your next run.'}</p><button className="button primary" type="submit" disabled={!source || !examples.length || Boolean(runBusy) || Boolean(exampleBusy)}>{runBusy === 'start' ? <Busy label="Starting training"/> : <><Play size={16}/>Train chat model</>}</button></div>
        </form>}
      </section>
      {pollError && <div className="notice error" role="status"><strong>Live training updates are temporarily unavailable.</strong><p>{pollError} Retrying automatically.</p></div>}
      <div className="training-layout chat-training-monitor">
        <section className="card runs-card"><SectionHead title="Chat experiments" action={<Badge>{runs.length}</Badge>}/>{runs.length ? <div className="run-list">{runs.map(item => <button className={item.id === run?.id ? 'selected' : ''} key={item.id} onClick={() => setSelected(item.id)} aria-pressed={item.id === run?.id}><div className="row space-between"><strong>{item.name}</strong><span className={`run-dot ${item.status}`}/></div><small>{item.example_count} examples · {date(item.created_at)}</small><small>{isUnified(item) ? `Source: ${sourceName(item)}` : 'Legacy standalone chat model'}</small><div className="row space-between"><Status value={item.status}/><span className="small mono">step {item.progress?.global_step ?? 0}</span></div></button>)}</div> : <Empty title="No chat runs yet">Save an example, select your grounding model, and train it for chat.</Empty>}</section>
        <section className="card live-run">{run ? <><div className="card-padding">
          <div className="run-progress-head"><div><p className="eyebrow">{isUnified(run) ? 'GROUNDING + CHAT EXPERIMENT' : 'LEGACY CHAT EXPERIMENT'}</p><h2>{run.name}</h2><p className="small muted">{date(run.created_at)} · {isUnified(run) ? `Source: ${sourceName(run)}` : 'Legacy standalone chat model'}</p></div><Status value={run.status}/></div>
          {!isUnified(run) && <div className="notice subtle"><p>This older experiment contains only a standalone chat model. Select your grounding model above to create a version with both abilities. Legacy runs cannot be resumed here.</p></div>}
          <div className="progress-track" role="progressbar" aria-label="Chat training epochs" aria-valuemin={0} aria-valuemax={run.config.epochs} aria-valuenow={Math.min(run.config.epochs, run.progress?.epoch ?? 0)}><span style={{ width: `${Math.min(100, ((run.progress?.epoch ?? 0) / run.config.epochs) * 100)}%` }}/></div><div className="row space-between small muted"><span>Epoch {run.progress?.epoch ?? 0} / {run.config.epochs}</span><span>{run.example_count} saved {run.example_count === 1 ? 'example' : 'examples'}</span></div>
          {run.error && <div className="notice error" role="alert"><strong>Training needs attention</strong><p>{run.error}</p></div>}
          <div className="training-stat-grid">{[{ label: 'Training loss', value: fmt(run.progress?.loss) }, { label: 'Optimizer steps', value: fmt(run.progress?.global_step ?? 0) }, { label: 'Learning rate', value: fmt(run.config.learning_rate) }, { label: 'Elapsed', value: elapsed(run.progress?.elapsed_seconds) }].map(metric => <div className="metric" key={metric.label}><span>{metric.label}</span><strong>{metric.value}</strong></div>)}</div>
          <p className="small muted chat-loss-note">Loss is measured during training. Try the saved model below to check what it has learned.</p>
          <div className="checkpoint-status"><Save size={16}/><span><strong>Latest saved checkpoint</strong><small className="mono">{run.latest_checkpoint ?? 'Waiting for the first saved training state'}</small></span></div>
        </div><div className="run-controls">{['running', 'queued'].includes(run.status) ? <button className="button" disabled={Boolean(runBusy)} onClick={() => void control('stop')}>{runBusy === 'stop' ? <Busy label="Stopping"/> : <><Square size={14}/>Stop & save</>}</button> : isUnified(run) && ['stopped', 'interrupted', 'error'].includes(run.status) ? <button className="button primary" disabled={Boolean(runBusy)} onClick={() => void control('resume')}>{runBusy === 'resume' ? <Busy label="Resuming"/> : <><RotateCcw size={16}/>Resume run</>}</button> : <span className="small muted grow">{!isUnified(run) ? 'Legacy experiment retained for reference.' : run.status === 'completed' ? 'Training finished. Test your model or select this version above to teach more examples.' : 'The worker is preparing this experiment.'}</span>}{run.latest_checkpoint && <button className="button" disabled={downloading} onClick={() => void downloadModel()}>{downloading ? <Busy label="Downloading"/> : <><ArrowDownToLine size={16}/>{isUnified(run) ? 'Download grounding + chat model' : 'Download legacy chat model'}</>}</button>}</div></> : <Empty title="Your chat training monitor">Training progress, measured loss, and saved model checkpoints will appear here.</Empty>}</section>
      </div>
      <section className="card chat-playground">
        <SectionHead title="3. Try your trained model" action={<MessageCircle size={20}/>}>Each message is tested on its own. Earlier messages are not passed to the model.</SectionHead>
        <div className="card-padding">
          <Field label="Chat model"><select value={testRun?.id ?? ''} onChange={event => setTestRunId(event.target.value)} disabled={!availableModels.length || testing}>{!availableModels.length && <option value="">Train a model to save your first checkpoint</option>}{availableModels.map(item => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>)}</select></Field>
          <div className="chat-transcript" role="log" aria-label="Model test replies" aria-live="polite">{replies.length ? replies.map(reply => <article className="chat-test-pair" key={reply.id}><div className="chat-message user"><span>You</span><p>{reply.prompt}</p></div><div className="chat-message model"><span>{reply.name}</span><p>{reply.reply || '(The model returned an empty reply.)'}</p>{Boolean(reply.unknown_characters) && <small>{reply.unknown_characters} {reply.unknown_characters === 1 ? 'character was' : 'characters were'} unseen during training. Add examples with these characters in a new run.</small>}<button className="text-link" type="button" disabled={Boolean(exampleBusy)} onClick={() => teachReply(reply.prompt)}><Plus size={13}/>Teach a better reply</button></div></article>) : <div className="chat-test-empty"><MessageCircle size={27}/><h3>{availableModels.length ? 'Say something to your model' : 'Your model’s replies will appear here'}</h3><p>{availableModels.length ? 'Try a message from your examples, then try a variation.' : 'Save your examples and train a model first. Replies are generated from its learned weights.'}</p></div>}{testing && <div className="chat-thinking"><Busy label="Generating a reply from your model"/></div>}</div>
          <form className="chat-composer" onSubmit={event => { event.preventDefault(); void predict(); }}><Field label="Test message"><input value={message} onChange={event => setMessage(event.target.value)} placeholder="Type a message, for example hey" maxLength={500} required disabled={!testRun || testing}/></Field><button className="button primary" type="submit" disabled={!testRun || !message.trim() || testing}>{testing ? <Busy label="Replying"/> : <><Send size={15}/>Send</>}</button></form>
          <p className="small muted">Testing uses the latest saved checkpoint. To teach more replies, save your examples and select this grounding + chat version as the source for a new run.</p>
        </div>
      </section>
    </>}
  </>;
}
