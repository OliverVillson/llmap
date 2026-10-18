/**
 * The mock orchestrator for `run.ts --fake --prompt "..."`: whatever the prompt, it answers the
 * design, scaffold and ticket calls with a project sized to the crew (one module per code agent,
 * plus a registry that waits on all of them), and the mock workers answer each ticket with
 * working code. It proves the path a real prompt takes at any crew size (N tickets planned,
 * N agents at once, N branches merged, the whole suite green), not how well a model plans.
 */
import type { Design } from '../src/plan/architect.ts';
import { validatePlan, type Plan, type Ticket, type WorkerOutput } from '../src/tickets/schema.ts';

export interface FakeProject {
  design: Design;
  scaffold: WorkerOutput;
  plan: Plan;
  solutions: Record<string, WorkerOutput>;
}

const pad = (i: number, n: number) => String(i).padStart(String(n).length, '0');

/** One unit conversion per module: a factor and an offset, so each module has a real answer. */
function unit(i: number) {
  const factor = Math.round((1.5 + ((i * 37) % 97) / 10) * 1000) / 1000;
  const offset = (i * 13) % 7;
  const at = (x: number) => Math.round((x * factor + offset) * 1e6) / 1e6;
  return { factor, offset, at };
}

export function fakeProject(n: number): FakeProject {
  const ids = Array.from({ length: n }, (_, k) => pad(k + 1, n));
  const mod = (s: string) => `conv_${s}`;
  const files: Record<string, string> = {
    'kit/__init__.py': '"""kit: unit converters, one module each, and a registry of them."""\n',
    'tests/__init__.py': '',
    'README.md': `# kit\n\n${n} unit converters (\`kit/conv_*.py\`) and a registry (\`kit/registry.py\`).\nRun the tests with \`python3 -m unittest discover -s tests -t .\`.\n`,
    '.gitignore': '__pycache__/\n*.pyc\n',
  };
  const solutions: Record<string, WorkerOutput> = {};
  const tickets: Ticket[] = [];
  const ticket = (t: Pick<Ticket, 'id' | 'title' | 'owns' | 'reads' | 'context' | 'acceptance' | 'depends_on'>): Ticket => ({
    role: 'implement', model: 'coder', priority: 2, image: 'python', budget: { attempts: 4, max_tokens: 96_000 }, escalate_to: null, ...t,
  });
  for (const s of ids) {
    const u = unit(Number(s));
    const path = `kit/${mod(s)}.py`;
    const doc = `    """Convert x to unit ${s}: x * ${u.factor} + ${u.offset}, rounded to 6 decimals."""\n`;
    files[path] = `"""Converter ${s}."""\n\n\ndef convert(x: float) -> float:\n${doc}    raise NotImplementedError\n`;
    files[`tests/test_${mod(s)}.py`] = [
      'import unittest',
      '',
      `from kit.${mod(s)} import convert`,
      '',
      '',
      'class ConvertTest(unittest.TestCase):',
      '    def test_values(self):',
      ...[0, 1, 2.5, -4].map((x) => `        self.assertAlmostEqual(convert(${x}), ${u.at(x)})`),
      '',
    ].join('\n');
    solutions[`conv-${s}`] = { files: { [path]: `"""Converter ${s}."""\n\n\ndef convert(x: float) -> float:\n${doc}    return round(x * ${u.factor} + ${u.offset}, 6)\n` }, note: `converter ${s}` };
    tickets.push(ticket({
      id: `conv-${s}`, title: `Converter ${s}`, owns: [path], reads: [`tests/test_${mod(s)}.py`], depends_on: [],
      context: `Fill in convert() in ${path} as its docstring says; tests/test_${mod(s)}.py is the definition of done.`,
      acceptance: [`python3 -m unittest -v tests.test_${mod(s)}`],
    }));
  }
  const registry = [
    '"""Every converter by name."""',
    '',
    ...ids.map((s) => `from kit import ${mod(s)}`),
    '',
    `CONVERTERS = {${ids.map((s) => `"${s}": ${mod(s)}.convert`).join(', ')}}`,
    '',
    '',
    'def convert(name: str, x: float) -> float:',
    '    """The converter called name applied to x; KeyError for an unknown name."""',
    '    return CONVERTERS[name](x)',
    '',
  ].join('\n');
  files['kit/registry.py'] = '"""Every converter by name."""\n\n\ndef convert(name: str, x: float) -> float:\n    """The converter called name applied to x (CONVERTERS maps each kit.conv_* module\'s number to its convert); KeyError for an unknown name."""\n    raise NotImplementedError\n';
  files['tests/test_registry.py'] = [
    'import unittest',
    '',
    'from kit.registry import CONVERTERS, convert',
    '',
    '',
    'class RegistryTest(unittest.TestCase):',
    `    def test_every_converter(self):\n        self.assertEqual(len(CONVERTERS), ${n})`,
    `    def test_convert(self):\n        self.assertAlmostEqual(convert("${ids[0]}", 1), ${unit(1).at(1)})`,
    '    def test_unknown(self):\n        with self.assertRaises(KeyError):\n            convert("nope", 1)',
    '',
  ].join('\n');
  solutions.registry = { files: { 'kit/registry.py': registry }, note: 'registry of every converter' };
  tickets.push(ticket({
    id: 'registry', title: 'Registry of every converter', owns: ['kit/registry.py'], reads: ['tests/test_registry.py'], depends_on: tickets.map((t) => t.id),
    context: 'Fill in kit/registry.py: CONVERTERS maps each converter number to its module\'s convert; tests/test_registry.py is the definition of done.',
    acceptance: ['python3 -m unittest -v tests.test_registry'],
  }));
  const design: Design = {
    summary: `kit: ${n} unit converters, one module each so ${n} agents can write them at once, and a registry that imports them all. (The mock orchestrator's stand-in for whatever was asked.)`,
    components: [{
      name: 'kit', language: 'Python', image: 'python', path: 'kit/', responsibility: 'the converters and their registry',
      interfaces: [{ path: 'kit/registry.py', description: 'convert(name, x) and CONVERTERS' }],
    }],
    data_flow: 'registry.convert(name, x) looks the converter up and calls it.',
    decisions: [{ decision: `one module per agent (${n})`, why: 'so every agent in the crew has a ticket from the start' }],
    questions: [],
  };
  return { design, scaffold: { files, note: `${n} converter stubs, a registry stub and their tests` }, plan: validatePlan({ interfaces: [], tickets }), solutions };
}
