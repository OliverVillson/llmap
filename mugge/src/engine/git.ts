/**
 * Git plumbing for the engine: one worktree per ticket off its base, branches named
 * `mugge/<ticket>`, and merges for dependency bases and the integrate step.
 *
 * Commits made here carry the configured author (the user) and nothing else: no trailers.
 */
import { existsSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

export interface GitResult {
  code: number;
  out: string;
}

export interface Author {
  name: string;
  email: string;
}

export const DEFAULT_AUTHOR: Author = { name: 'mugge', email: 'mugge@localhost' };

export function git(cwd: string, args: string[], author?: Author): GitResult {
  const env: Record<string, string> = { ...(process.env as Record<string, string>), GIT_TERMINAL_PROMPT: '0' };
  if (author) {
    env.GIT_AUTHOR_NAME = env.GIT_COMMITTER_NAME = author.name;
    env.GIT_AUTHOR_EMAIL = env.GIT_COMMITTER_EMAIL = author.email;
  }
  const r = Bun.spawnSync(['git', ...args], { cwd, env, stdout: 'pipe', stderr: 'pipe' });
  return { code: r.exitCode ?? 1, out: (r.stdout.toString() + r.stderr.toString()).trim() };
}

export function gitOk(cwd: string, args: string[], author?: Author): string {
  const r = git(cwd, args, author);
  if (r.code !== 0) throw new Error(`git ${args.join(' ')} failed in ${cwd}:\n${r.out}`);
  return r.out;
}

export const branchFor = (ticketId: string) => `mugge/${ticketId}`;
export const INTEGRATION_BRANCH = 'mugge/integration';

export function headCommit(repo: string, ref = 'HEAD'): string {
  return gitOk(repo, ['rev-parse', ref]);
}

/** Initializes `dir` as a repo holding its current files as one base commit. Returns the commit. */
export function initRepo(dir: string, author: Author = DEFAULT_AUTHOR, message = 'scaffold'): string {
  if (!existsSync(join(dir, '.git'))) gitOk(dir, ['init', '-q', '-b', 'main']);
  gitOk(dir, ['add', '-A']);
  gitOk(dir, ['commit', '-q', '--allow-empty', '-m', message], author);
  return headCommit(dir);
}

export class Worktrees {
  constructor(
    readonly repo: string,
    /** Folder that holds every worktree; outside the repo so tools never see it twice. */
    readonly root: string,
    readonly author: Author = DEFAULT_AUTHOR,
  ) {
    mkdirSync(root, { recursive: true });
  }

  pathFor(name: string): string {
    return join(this.root, name.replace(/[^a-z0-9-]/gi, '_'));
  }

  /**
   * A fresh worktree on `branch` starting at `base` (a commit), or at a merge of several
   * branches when `merge` is given (the ticket's dependencies). A merge conflict throws: the
   * plan promised the dependencies own different files.
   */
  create(name: string, branch: string, base: string, merge: string[] = []): string {
    const path = this.pathFor(name);
    this.remove(name, branch);
    gitOk(this.repo, ['worktree', 'add', '-q', '-B', branch, path, base]);
    if (merge.length) {
      const r = git(path, ['merge', '-q', '--no-edit', '-m', `merge ${merge.join(', ')}`, ...merge], this.author);
      if (r.code !== 0) {
        git(path, ['merge', '--abort']);
        throw new Error(`merging ${merge.join(', ')} for ${name} failed:\n${r.out}`);
      }
    }
    return path;
  }

  /** Commits exactly `paths` (the ticket's owned files) and returns the commit, or null when nothing changed. */
  commit(path: string, paths: string[], message: string): string | null {
    gitOk(path, ['add', '-A', '--', ...paths]);
    if (git(path, ['diff', '--cached', '--quiet']).code === 0) return null;
    gitOk(path, ['commit', '-q', '-m', message], this.author);
    return headCommit(path);
  }

  remove(name: string, branch?: string): void {
    const path = this.pathFor(name);
    if (existsSync(path)) {
      git(this.repo, ['worktree', 'remove', '--force', path]);
      rmSync(path, { recursive: true, force: true });
    }
    git(this.repo, ['worktree', 'prune']);
    if (branch) git(this.repo, ['branch', '-D', branch]);
  }
}
