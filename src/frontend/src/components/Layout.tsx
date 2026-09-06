import { useEffect, useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import type { User } from '../types';
import { api } from '../api';

const LINKS = [
  { to: '/', label: 'Przegląd', end: true },
  { to: '/leaderboard', label: 'Ranking' },
  { to: '/runs', label: 'Uruchomienia' },
  { to: '/compare', label: 'Porównanie' },
  { to: '/submit', label: 'Zgłoszenie' },
  { to: '/docs', label: 'Protokół' },
];

type Theme = 'system' | 'light' | 'dark';

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('theme');
    return stored === 'light' || stored === 'dark' ? stored : 'system';
  });

  useEffect(() => {
    if (theme === 'system') {
      document.documentElement.removeAttribute('data-theme');
      localStorage.removeItem('theme');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem('theme', theme);
    }
  }, [theme]);

  const next: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' };
  const label: Record<Theme, string> = { system: 'Motyw: systemowy', light: 'Motyw: jasny', dark: 'Motyw: ciemny' };

  return (
    <button className="subtle tiny" onClick={() => setTheme(next[theme])} title={label[theme]}>
      {label[theme]}
    </button>
  );
}

export function Layout({
  user, onUserChange, children,
}: {
  user: User | null;
  onUserChange: (user: User | null) => void;
  children: React.ReactNode;
}) {
  return (
    <div className="shell">
      <a className="skip-link" href="#main">Przejdź do treści</a>
      <header className="masthead">
        <div className="masthead-inner">
          <Link to="/" className="wordmark">
            Benchmark Czarnej Skrzynki <span>· Politechnika Warszawska</span>
          </Link>
          <nav className="nav" aria-label="Nawigacja główna">
            {LINKS.map((link) => (
              <NavLink key={link.to} to={link.to} end={link.end}
                       className={({ isActive }) => (isActive ? 'active' : '')}>
                {link.label}
              </NavLink>
            ))}
            {user?.is_admin && (
              <>
                <NavLink to="/admin" end className={({ isActive }) => (isActive ? 'active' : '')}>
                  Panel
                </NavLink>
                <NavLink to="/admin/queue" className={({ isActive }) => (isActive ? 'active' : '')}>
                  Kolejka
                </NavLink>
              </>
            )}
          </nav>
          <div className="row" style={{ gap: 'var(--space-2)' }}>
            <ThemeToggle />
            {user ? (
              <>
                <span className="tiny muted mono" title={user.email}>
                  {user.display_name ?? user.email}
                </span>
                <button
                  className="subtle tiny"
                  onClick={async () => { await api.post('/api/auth/logout'); onUserChange(null); }}
                >
                  Wyloguj
                </button>
              </>
            ) : (
              <Link className="button tiny" to="/login" style={{ padding: '2px 8px' }}>Zaloguj</Link>
            )}
          </div>
        </div>
      </header>
      <main id="main">{children}</main>
      <footer style={{
        borderTop: '1px solid var(--border)', padding: 'var(--space-4) var(--space-5)',
        color: 'var(--text-3)', fontSize: 12,
      }}>
        <div style={{ maxWidth: 1400, margin: '0 auto' }}>
          Zakład Sztucznej Inteligencji, Wydział Elektroniki i Technik Informacyjnych ·{' '}
          <Link to="/docs">kontrakt optymalizatora</Link> ·{' '}
          <a href="/api/openapi">dokumentacja API</a>
        </div>
      </footer>
    </div>
  );
}
