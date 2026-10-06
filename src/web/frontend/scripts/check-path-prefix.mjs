import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(scriptDir, '..');
const sourceRoot = path.join(frontendRoot, 'src');
const projectRoot = path.resolve(frontendRoot, '../../..');

function sourceFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(candidate);
    return /\.tsx?$/.test(entry.name) ? [candidate] : [];
  });
}

const unsafeBrowserUrls = [
  ['fetch', /\bfetch\s*\(\s*(?:`|["'])\/api\//g],
  ['EventSource', /\bnew\s+EventSource\s*\(\s*(?:`|["'])\/api\//g],
  ['JSX href/src', /\b(?:href|src)\s*=\s*(?:["']\/api\/|\{\s*`\/api\/)/g],
  ['URL variable', /\b\w*(?:href|url)\w*\s*=\s*`\/api\//gi],
];

const violations = [];
for (const filename of sourceFiles(sourceRoot)) {
  const source = fs.readFileSync(filename, 'utf8');
  for (const [label, pattern] of unsafeBrowserUrls) {
    for (const match of source.matchAll(pattern)) {
      const line = source.slice(0, match.index).split('\n').length;
      violations.push(`${path.relative(frontendRoot, filename)}:${line} (${label})`);
    }
  }
}

if (violations.length) {
  throw new Error(
    `Browser URL bypasses withBase(); it will escape a path-prefix deployment:\n${violations.join('\n')}`,
  );
}

const compose = fs.readFileSync(path.join(projectRoot, 'docker-compose.yml'), 'utf8');
const webStart = compose.indexOf('\n  web:\n');
const nextService = compose.indexOf('\n  outbox_publisher:\n', webStart);
const webService = compose.slice(webStart, nextService);
if (!webService.includes('BASE_PATH: ${BASE_PATH:-}')) {
  throw new Error('docker-compose.yml does not pass BASE_PATH into the web image build');
}

const [distArgument, prefixArgument] = process.argv.slice(2);
if (distArgument || prefixArgument) {
  const dist = path.resolve(frontendRoot, distArgument ?? 'dist');
  const prefix = prefixArgument ?? '/benchmark/';
  const html = fs.readFileSync(path.join(dist, 'index.html'), 'utf8');
  const localAssets = [...html.matchAll(/(?:src|href)="(\/[^"#]+)"/g)].map((match) => match[1]);

  if (localAssets.length === 0) throw new Error('built index.html contains no root-relative assets');
  const escaped = localAssets.filter((url) => !url.startsWith(prefix));
  if (escaped.length) {
    throw new Error(`built assets escape ${prefix}: ${escaped.join(', ')}`);
  }
}
