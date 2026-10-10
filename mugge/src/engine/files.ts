/**
 * How a coder model writes files: each owned file in full, as its path on one line and its
 * content in a fenced code block, then an optional `Note:` line. Plain files rather than JSON,
 * because code escaped inside JSON scores worse (Aider's code-in-JSON test). llmap trains the
 * specialists on exactly this shape (stages/harness.py, which must stay in sync).
 */
import type { WorkerOutput } from '../tickets/schema.ts';

const FENCE_LANG: Record<string, string> = {
  '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp',
  '.s': 'asm', '.html': 'html', '.css': 'css', '.json': 'json', '.md': 'markdown', '.sh': 'bash', '.toml': 'toml',
  '.yaml': 'yaml', '.yml': 'yaml',
};

/** The answer for these files; a fence longer than any backtick run inside keeps the content intact. */
export function renderFiles(files: Record<string, string>, note = ''): string {
  const parts = Object.entries(files).map(([path, content]) => {
    const dot = path.lastIndexOf('.');
    const lang = dot >= 0 ? (FENCE_LANG[path.slice(dot).toLowerCase()] ?? '') : '';
    const runs = (content.match(/^`{3,}/gm) ?? []).map((r) => r.length);
    const fence = '`'.repeat(Math.max(3, ...runs.map((n) => n + 1)));
    return `${path}\n${fence}${lang}\n${content.replace(/\n+$/, '')}\n${fence}`;
  });
  if (note) parts.push(`Note: ${note}`);
  return parts.join('\n\n');
}

const TRIM = /^[*`: ]+|[*`: ]+$/g;

/** A file path on its own line, minus markdown around it (`**a.py**`, `### \`a.py\`:`, `File: a.py`). */
function pathOf(line: string): string {
  const p = line.trim().replace(/^#+\s*/, '').replace(TRIM, '').replace(/^(?:file|path|filename)\s*:\s*/i, '').replace(TRIM, '');
  return p && !/\s/.test(p) ? p : '';
}

/** A path in a fence's info string: ```python src/a.py or ```ts title="a.ts". */
function infoPath(info: string | undefined): string {
  const p = (info ?? '').trim().replace(/^(?:title|file|filename|path)=/i, '').replace(/^["']+|["']+$/g, '');
  return p && !/\s/.test(p) && /[./]/.test(p) ? p : '';
}

/** Whether text names path as a whole word ("a.py" is not named by "data.py"). */
function names(text: string, path: string): boolean {
  return new RegExp(`(?<![\\w./-])${path.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\w/-])`).test(text);
}

const base = (path: string) => path.slice(path.lastIndexOf('/') + 1);

/** The one owned path a line names (the longest when several match), else the one owned path whose file name it names, else ''. */
function mentioned(text: string, owns: string[]): string {
  const hits = owns.filter((p) => names(text, p)).sort((a, b) => b.length - a.length);
  if (hits.length) return hits[0]!;
  const byName = owns.filter((p) => names(text, base(p)));
  return byName.length === 1 ? byName[0]! : '';
}

/**
 * Files and note from an answer. A fenced block counts when the line before it is a path, or
 * when its info string names one. With `owns` (the ticket's files), a file written under the bare
 * name of an owned path moves to that path, a block also counts for an owned path that the line
 * before it or its own first line mentions ("Here is the fixed `src/a.py`:"), and a ticket that
 * owns one file takes its last unlabelled block. Other blocks are not files. Thinking
 * (`<think>…</think>`) is ignored.
 */
export function parseFiles(raw: string, owns: string[] = []): WorkerOutput {
  const lines = raw.replace(/<think>[\s\S]*?<\/think>/g, '').split('\n');
  const files: Record<string, string> = {};
  const loose: { before: string; first: string; content: string }[] = [];
  let note = '', prev = '';
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const open = line.trim().match(/^(`{3,})[\w+#.-]*(?:\s+(.*?))?\s*$/);
    if (open) {
      const fence = open[1]!, body: string[] = [];
      for (i++; i < lines.length && lines[i]!.trim() !== fence; i++) body.push(lines[i]!);
      const content = body.length ? body.join('\n') + '\n' : '';
      const path = pathOf(prev) || infoPath(open[2]);
      if (path) files[path] = content;
      else loose.push({ before: prev, first: body[0] ?? '', content });
      prev = '';
      continue;
    }
    if (/^note:/i.test(line.trim())) note = line.trim().slice(5).trim();
    if (line.trim()) prev = line;
  }
  for (const path of Object.keys(files).filter((p) => owns.length && !owns.includes(p))) {
    const same = owns.filter((o) => base(o) === base(path)); // "a.py" for an owned "src/a.py"
    if (same.length === 1 && !(same[0]! in files)) {
      files[same[0]!] = files[path]!;
      delete files[path];
    }
  }
  const rest: string[] = [];
  for (const b of loose) {
    const path = mentioned(b.before, owns) || mentioned(b.first, owns);
    if (path && !(path in files)) files[path] = b.content;
    else rest.push(b.content);
  }
  if (owns.length === 1 && !(owns[0]! in files) && rest.length) files[owns[0]!] = rest[rest.length - 1]!;
  return { files, note };
}
