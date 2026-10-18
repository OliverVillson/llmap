import { describe, expect, test } from 'bun:test';
import { ancestors, loadPlan, PROJECT_DIR, referenceAnswers } from '../demo/lib.ts';
import { fakeProject } from '../demo/fake-prompt.ts';
import { checkPlan, importsOf, plannerMessage } from '../demo/orchestrator.ts';

describe('demo plan', () => {
  const plan = loadPlan();

  test('ten agents have more work than they can take at the start, and one clash is settled', () => {
    expect(plan.tickets.length).toBe(17);
    expect(plan.tickets.filter((t) => !t.depends_on.length).length).toBeGreaterThan(10);
    expect(plan.settled).toEqual([{ ticket: 'block-code', files: ['mdlite/theme.css'], with: ['block-table'] }]);
  });

  test('every ticket has a reference answer for exactly the files it owns', () => {
    const answers = referenceAnswers(plan);
    for (const t of plan.tickets) expect(Object.keys(answers[t.id]!.files).sort()).toEqual([...t.owns].sort());
  });

  test('the prepared plan passes the check an orchestrator plan must pass', () => {
    expect(checkPlan(plan, PROJECT_DIR)).toEqual([]);
  });

  test('the render ticket waits, through its dependencies, for every module it runs', () => {
    expect([...ancestors(plan, 'render')].sort()).toContain('inline-emphasis');
  });
});

describe('orchestrator plan check', () => {
  const plan = loadPlan();
  const without = (id: string, dep: string) => ({ ...plan, tickets: plan.tickets.map((t) => (t.id === id ? { ...t, depends_on: t.depends_on.filter((d) => d !== dep) } : t)) });

  test('a ticket that does not wait for a module its test needs is caught, and so is every ticket that only reaches it through that one', () => {
    expect(checkPlan(without('inline', 'inline-links'), PROJECT_DIR)).toEqual([
      'inline: its test needs mdlite.inline_links (inline-links) but does not wait for it',
      'render: its test needs mdlite.inline_links (inline-links) but does not wait for it',
      'cli: its test needs mdlite.inline_links (inline-links) but does not wait for it',
    ]);
  });

  test('a stub nobody fills in, a test nobody runs and a file outside the scaffold are caught', () => {
    const p = { ...plan, tickets: plan.tickets.filter((t) => t.id !== 'toc').map((t) => (t.id === 'slug' ? { ...t, owns: [...t.owns, 'mdlite/nope.py'] } : t)) };
    expect(checkPlan(p, PROJECT_DIR).sort()).toEqual(['no ticket fills in mdlite/toc.py', 'no ticket runs tests.test_toc', 'slug owns mdlite/nope.py, which is not in the scaffold']);
  });

  test('two tickets filling in the same stub are caught', () => {
    const p = { ...plan, tickets: plan.tickets.map((t) => (t.id === 'escape' ? { ...t, owns: [...t.owns, 'mdlite/slug.py'] } : t)) };
    expect(checkPlan(p, PROJECT_DIR)).toContain('mdlite/slug.py is owned by both escape and slug');
  });

  test('imports are read from both import forms', () => {
    expect(importsOf('from mdlite import block_code, block_list\nfrom mdlite.inline import render_inline\nimport mdlite.toc\nfrom mdlite.types import Heading')).toEqual(['mdlite.block_code', 'mdlite.block_list', 'mdlite.inline', 'mdlite.toc']);
  });

  test('the planning message lists every stub and test', () => {
    const m = plannerMessage(PROJECT_DIR);
    expect(m).toContain('- mdlite/block_heading.py (imports mdlite.slug)');
    expect(m).toContain('also owns mdlite/theme.css');
    expect(m).toContain('- tests/test_render.py: imports mdlite.render');
  });
});

describe('the mock orchestrator for --fake --prompt', () => {
  test('plans about one ticket per agent in groups of 16, most ready at once', () => {
    const fp = fakeProject(64);
    expect(fp.design.components.length).toBe(4);
    expect(fp.plan.tickets.length).toBe(64);
    expect(fp.plan.tickets.filter((t) => !t.depends_on.length).length).toBe(60);
    for (const t of fp.plan.tickets) expect(Object.keys(fp.solutions[t.id]!.files)).toEqual(t.owns);
  });

  test('a small crew gets one group, planned in one call', () => {
    const fp = fakeProject(10);
    expect(fp.design.components.length).toBe(1);
    expect(fp.plan.tickets.length).toBe(10);
  });
});
