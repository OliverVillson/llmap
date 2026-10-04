// Entry point: bun cli/bin.ts <command>. Not part of tsconfig (it uses Bun/Node globals).
import { main } from './src/main.ts';

declare const process: { argv: string[]; env: Record<string, string | undefined>; exit(code: number): never };

const code = await main(process.argv.slice(2), {
  fetch: (url, init) => fetch(url, init),
  env: process.env,
  out: (line) => console.log(line),
  err: (line) => console.error(line),
});
process.exit(code);
