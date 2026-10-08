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

/** A file path on its own line, minus markdown around it (`**a.py**`, `### \`a.py\`:`). */
function pathOf(line: string): string {
  const p = line.trim().replace(/^#+\s*/, '').replace(/^[*`: ]+|[*`: ]+$/g, '');
  return p && !/\s/.test(p) ? p : '';
}

/**
 * Files and note from an answer. A fenced block counts when the line before it is a path; a
 * block without one is not a file. Thinking (`<think>…</think>`) is ignored.
 */
export function parseFiles(raw: string): WorkerOutput {
  const lines = raw.replace(/<think>[\s\S]*?<\/think>/g, '').split('\n');
  const files: Record<string, string> = {};
  let note = '', prev = '';
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const open = line.trim().match(/^(`{3,})[\w+#.-]*\s*$/);
    if (open) {
      const fence = open[1]!, body: string[] = [];
      for (i++; i < lines.length && lines[i]!.trim() !== fence; i++) body.push(lines[i]!);
      const path = pathOf(prev);
      if (path) files[path] = body.length ? body.join('\n') + '\n' : '';
      prev = '';
      continue;
    }
    if (/^note:/i.test(line.trim())) note = line.trim().slice(5).trim();
    if (line.trim()) prev = line;
  }
  return { files, note };
}
