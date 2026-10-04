/**
 * `mugge ship`: hand the finished code to the user with no stamp. The integrated tree becomes
 * one ordinary commit authored by the user, with their message and no trailers, on a branch of
 * their repo or copied into a local folder. Tickets, plans and logs never leave the state store.
 */
import { existsSync, mkdirSync } from 'node:fs';
import { git, gitOk, INTEGRATION_BRANCH, type Author } from '../engine/git.ts';

export interface ShipOptions {
  /** The engine's repo holding the integration branch. */
  repo: string;
  ref?: string;
  author: Author;
  message: string;
}

/** Makes one squashed commit of `ref`'s tree onto `parent` (or as a root commit). */
export function squashCommit(o: ShipOptions & { parent?: string }): string {
  const tree = gitOk(o.repo, ['rev-parse', `${o.ref ?? INTEGRATION_BRANCH}^{tree}`]);
  return gitOk(o.repo, ['commit-tree', tree, ...(o.parent ? ['-p', o.parent] : []), '-m', o.message], o.author);
}

/** Pushes the squashed commit to `branch` of a git remote. */
export function shipToRemote(o: ShipOptions & { remote: string; branch: string }): string {
  const fetched = git(o.repo, ['fetch', '-q', o.remote, o.branch]);
  const parent = fetched.code === 0 ? gitOk(o.repo, ['rev-parse', 'FETCH_HEAD']) : undefined;
  const commit = squashCommit({ ...o, parent });
  gitOk(o.repo, ['push', '-q', o.remote, `${commit}:refs/heads/${o.branch}`]);
  return commit;
}

/** Copies the tree into a local folder; when the folder is a git repo, commits it there as the user. */
export function shipToFolder(o: ShipOptions & { dir: string }): string | null {
  mkdirSync(o.dir, { recursive: true });
  const archive = Bun.spawnSync(['git', 'archive', '--format=tar', o.ref ?? INTEGRATION_BRANCH], { cwd: o.repo, stdout: 'pipe' });
  if (archive.exitCode !== 0) throw new Error('git archive failed');
  const untar = Bun.spawnSync(['tar', '-x', '-C', o.dir], { stdin: archive.stdout, stdout: 'pipe', stderr: 'pipe' });
  if (untar.exitCode !== 0) throw new Error(`tar failed: ${untar.stderr.toString()}`);
  if (!existsSync(`${o.dir}/.git`)) return null;
  gitOk(o.dir, ['add', '-A']);
  if (git(o.dir, ['diff', '--cached', '--quiet']).code === 0) return null;
  gitOk(o.dir, ['commit', '-q', '-m', o.message], o.author);
  return gitOk(o.dir, ['rev-parse', 'HEAD']);
}
