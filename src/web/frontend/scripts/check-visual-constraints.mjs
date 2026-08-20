#!/usr/bin/env node
// §14.2 bans gradients, glassmorphism, neon glow and heavy shadows. This greps
// for them so the ban is enforced by CI rather than by reviewer memory.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const FORBIDDEN = [
  [/linear-gradient|radial-gradient|conic-gradient/i, 'gradient'],
  [/backdrop-filter/i, 'glassmorphism'],
  [/box-shadow:[^;]*\b([2-9]\d|\d{3,})px/i, 'heavy shadow (use a 1px border)'],
  [/text-shadow:\s*(?!none)/i, 'text shadow / glow'],
  [/@import\s+url\(["']?https?:/i, 'remote stylesheet (breaks CSP)'],
  [/fonts\.googleapis\.com|fonts\.gstatic\.com/i, 'font loaded from a CDN'],
];

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (/\.(css|tsx|ts)$/.test(name)) out.push(full);
  }
  return out;
}

let failures = 0;
for (const file of walk('src')) {
  const lines = readFileSync(file, 'utf8').split('\n');
  lines.forEach((line, index) => {
    if (line.includes('visual-constraints:allow')) return;
    for (const [pattern, label] of FORBIDDEN) {
      if (pattern.test(line)) {
        console.error(`${file}:${index + 1}  ${label}\n    ${line.trim()}`);
        failures += 1;
      }
    }
  });
}

if (failures > 0) {
  console.error(`\n${failures} violation(s) of the visual constraints in §14.2.`);
  process.exit(1);
}
console.log('Visual constraints: clean.');
