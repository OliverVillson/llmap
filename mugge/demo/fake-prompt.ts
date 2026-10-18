/**
 * The mock orchestrator for `run.ts --fake --prompt "..."`: whatever the prompt, it answers the
 * engine's planning calls (src/plan/architect.ts) with a project sized to the crew, and the mock
 * workers answer each ticket with working code. The project is `kit`, unit converters in groups:
 * one design component per 16 agents (kit/<group>/), each with a converter module per agent but
 * one, and a table module that waits on its group's converters. Above 16 agents the engine splits
 * the scaffold and the tickets per component; the mock answers each part with its own group. It
 * proves the path a real prompt takes at any crew size (N tickets planned, most ready at once,
 * N agents at work, every branch merged, the whole suite green), not how well a model plans.
 */
import { crewShares, DESIGN_JSON_SCHEMA, type Design } from '../src/plan/architect.ts';
import type { ChatMessage } from '../src/engine/inference.ts';
import { validatePlan, WORKER_OUTPUT_JSON_SCHEMA, type Plan, type Ticket, type WorkerOutput } from '../src/tickets/schema.ts';

export interface FakeProject {
  design: Design;
  plan: Plan;
  solutions: Record<string, WorkerOutput>;
  /** The mock orchestrator's answer to one planning call, by its schema and messages. */
  answer(schema: object | undefined, messages: ChatMessage[]): unknown;
}

const GROUPS = ['length', 'mass', 'time', 'volume', 'area', 'speed', 'energy', 'power', 'pressure', 'data', 'angle', 'force'];
const pad = (i: number) => String(i).padStart(2, '0');

/** One unit conversion per module: a factor and an offset, so each module has a real answer. */
function unit(i: number) {
  const factor = Math.round((1.5 + ((i * 37) % 97) / 10) * 1000) / 1000;
  const offset = (i * 13) % 7;
  return { factor, offset, at: (x: number) => Math.round((x * factor + offset) * 1e6) / 1e6 };
}

interface Group {
  name: string;
  files: Record<string, string>;
  tickets: Ticket[];
}

export function fakeProject(n: number): FakeProject {
  const count = n > 16 ? Math.min(GROUPS.length, Math.ceil(n / 16)) : 1;
  const names = GROUPS.slice(0, count);
  const design: Design = {
    summary: `kit: unit converters in ${count} group${count > 1 ? 's' : ''}, one converter module per code agent and a table per group. (The mock orchestrator's stand-in for whatever was asked.)`,
    components: names.map((g) => ({
      name: g, language: 'Python', image: 'python', path: `kit/${g}/`, responsibility: `the ${g} converters and their table`,
      interfaces: [{ path: `kit/${g}/__init__.py`, description: `the ${g} group: one module per converter, each with convert(x)` }],
    })),
    data_flow: 'table.convert(name, x) looks a converter of its group up and calls it.',
    decisions: [{ decision: `about ${n} tickets, one per agent`, why: 'so every agent in the crew has a ticket from the start' }],
    questions: [],
  };
  const shares = crewShares(design, n);
  const shared: Record<string, string> = {
    'kit/__init__.py': '"""kit: unit converters in groups."""\n',
    'tests/__init__.py': '',
    'README.md': `# kit\n\nUnit converters in ${count} groups (\`kit/<group>/conv_*.py\`), each with a table (\`kit/<group>/table.py\`).\nRun the tests with \`python3 -m unittest discover -s tests -t .\`.\n`,
    '.gitignore': '__pycache__/\n*.pyc\n',
  };
  for (const g of names) shared[`kit/${g}/__init__.py`] = `"""The ${g} converters: kit.${g}.conv_NN.convert(x), and kit.${g}.table."""\n`;
  const solutions: Record<string, WorkerOutput> = {};
  const ticket = (t: Pick<Ticket, 'id' | 'title' | 'owns' | 'reads' | 'context' | 'acceptance' | 'depends_on'>): Ticket => ({
    role: 'implement', model: 'coder', priority: 2, image: 'python', budget: { attempts: 4, max_tokens: 96_000 }, escalate_to: null, ...t,
  });
  let k = 0;
  const groups: Group[] = names.map((g, gi) => {
    const files: Record<string, string> = {};
    const tickets: Ticket[] = [];
    const nums = Array.from({ length: Math.max(1, shares[gi]! - 1) }, (_, j) => pad(j + 1));
    for (const s of nums) {
      const u = unit(++k);
      const path = `kit/${g}/conv_${s}.py`, test = `tests/test_${g}_conv_${s}.py`;
      const doc = `    """Convert x to ${g} unit ${s}: x * ${u.factor} + ${u.offset}, rounded to 6 decimals."""\n`;
      files[path] = `"""${g} converter ${s}."""\n\n\ndef convert(x: float) -> float:\n${doc}    raise NotImplementedError\n`;
      files[test] = ['import unittest', '', `from kit.${g}.conv_${s} import convert`, '', '', 'class ConvertTest(unittest.TestCase):', '    def test_values(self):',
        ...[0, 1, 2.5, -4].map((x) => `        self.assertAlmostEqual(convert(${x}), ${u.at(x)})`), ''].join('\n');
      const id = `${g}-conv-${s}`;
      solutions[id] = { files: { [path]: `"""${g} converter ${s}."""\n\n\ndef convert(x: float) -> float:\n${doc}    return round(x * ${u.factor} + ${u.offset}, 6)\n` }, note: `${g} converter ${s}` };
      tickets.push(ticket({ id, title: `${g} converter ${s}`, owns: [path], reads: [test], depends_on: [],
        context: `Fill in convert() in ${path} as its docstring says; ${test} is the definition of done.`, acceptance: [`python3 -m unittest -v ${test.replace(/\//g, '.').replace(/\.py$/, '')}`] }));
    }
    const table = `kit/${g}/table.py`, ttest = `tests/test_${g}_table.py`;
    files[table] = `"""Every ${g} converter by number."""\n\n\ndef convert(name: str, x: float) -> float:\n    """The ${g} converter numbered name ("01", "02", ...) applied to x (CONVERTERS maps each number to its module's convert); KeyError for an unknown name."""\n    raise NotImplementedError\n`;
    files[ttest] = ['import unittest', '', `from kit.${g}.table import CONVERTERS, convert`, '', '', 'class TableTest(unittest.TestCase):',
      `    def test_every_converter(self):\n        self.assertEqual(len(CONVERTERS), ${nums.length})`,
      `    def test_convert(self):\n        self.assertAlmostEqual(convert("01", 1), CONVERTERS["01"](1))`,
      '    def test_unknown(self):\n        with self.assertRaises(KeyError):\n            convert("nope", 1)', ''].join('\n');
    solutions[`${g}-table`] = { files: { [table]: [`"""Every ${g} converter by number."""`, '', ...nums.map((s) => `from kit.${g} import conv_${s}`), '',
      `CONVERTERS = {${nums.map((s) => `"${s}": conv_${s}.convert`).join(', ')}}`, '', '', 'def convert(name: str, x: float) -> float:',
      `    """The ${g} converter numbered name applied to x; KeyError for an unknown name."""`, '    return CONVERTERS[name](x)', ''].join('\n') }, note: `${g} table` };
    tickets.push(ticket({ id: `${g}-table`, title: `${g} table`, owns: [table], reads: [ttest], depends_on: tickets.map((t) => t.id),
      context: `Fill in ${table}: CONVERTERS maps each ${g} converter's number to its convert; ${ttest} is the definition of done.`, acceptance: [`python3 -m unittest -v tests.test_${g}_table`] }));
    return { name: g, files, tickets };
  });
  const plan = validatePlan({ interfaces: names.map((g) => `kit/${g}/__init__.py`), tickets: groups.flatMap((g) => g.tickets) });
  const groupOf = (messages: ChatMessage[]) => groups.find((g) => (messages.at(-1)?.content ?? '').includes(`## Your component: ${g.name} (`));
  return {
    design, plan, solutions,
    answer(schema, messages) {
      const system = messages[0]?.content ?? '';
      if (schema === DESIGN_JSON_SCHEMA) return design;
      if (schema === WORKER_OUTPUT_JSON_SCHEMA) {
        if (system.startsWith('You write the shared part')) return { files: shared, note: 'package, test setup and one interface per group' };
        const g = groupOf(messages);
        if (g) return { files: g.files, note: `${Object.keys(g.files).length / 2} ${g.name} stubs and their tests` };
        return { files: { ...shared, ...Object.assign({}, ...groups.map((x) => x.files)) }, note: 'every stub and its test' };
      }
      const g = system.startsWith('You split one component') ? groupOf(messages) : undefined;
      return g ? { interfaces: [`kit/${g.name}/__init__.py`], tickets: g.tickets } : plan;
    },
  };
}
