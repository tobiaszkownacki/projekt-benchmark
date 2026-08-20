import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    plugins: { react },
    languageOptions: {
      globals: {
        window: 'readonly', document: 'readonly', localStorage: 'readonly',
        fetch: 'readonly', console: 'readonly', setTimeout: 'readonly',
        clearTimeout: 'readonly', EventSource: 'readonly', URL: 'readonly',
        URLSearchParams: 'readonly', navigator: 'readonly', Blob: 'readonly',
        AbortController: 'readonly', HTMLElement: 'readonly', SVGSVGElement: 'readonly',
        matchMedia: 'readonly', requestAnimationFrame: 'readonly', KeyboardEvent: 'readonly',
        MouseEvent: 'readonly', Event: 'readonly', alert: 'readonly',
      },
    },
    rules: {
      // File contents come from competition entrants. React escapes anything
      // rendered as a child, so the only way to get script onto this origin is
      // to opt out of that -- which is what this forbids. A rule nobody enforces
      // stops being a rule by the third pull request.
      'react/no-danger': 'error',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
);
