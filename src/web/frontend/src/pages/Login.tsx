import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api';
import type { User } from '../types';

export function Login({ onUserChange }: { onUserChange: (user: User | null) => void }) {
  const navigate = useNavigate();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [organisation, setOrganisation] = useState('');
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null); setNotice(null); setBusy(true);
    try {
      if (mode === 'login') {
        const response = await api.post<{ user: User }>('/api/auth/login', { email, password });
        onUserChange(response.user);
        navigate('/runs?mine=1');
      } else {
        await api.post('/api/auth/register', {
          email, password, associated_organisation: organisation, join_reason: reason,
        });
        setNotice('Konto założone. Czeka na zatwierdzenie przez administratora.');
        setMode('login');
      }
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page" style={{ maxWidth: 460 }}>
      <div className="page-head">
        <h1>{mode === 'login' ? 'Logowanie' : 'Rejestracja'}</h1>
        <p className="small">
          Przeglądanie rankingu i wyników nie wymaga konta. Konto jest potrzebne,
          żeby zgłosić własny optymalizator.
        </p>
      </div>

      {notice && <div className="note note-accent" style={{ marginBottom: 16 }}>{notice}</div>}
      {error && <div className="note note-error" style={{ marginBottom: 16 }}>{error}</div>}

      <form onSubmit={submit} className="panel">
        <div className="panel-body">
          <div className="field">
            <label htmlFor="email">Adres e-mail</label>
            <input id="email" type="email" required autoComplete="username"
                   style={{ width: '100%' }} value={email}
                   onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="password">Hasło{mode === 'register' ? ' (min. 8 znaków)' : ''}</label>
            <input id="password" type="password" required style={{ width: '100%' }}
                   autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                   value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          {mode === 'register' && (
            <>
              <div className="field">
                <label htmlFor="org">Instytucja</label>
                <input id="org" style={{ width: '100%' }} value={organisation}
                       onChange={(e) => setOrganisation(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="reason">Powód dołączenia</label>
                <textarea id="reason" rows={3} value={reason}
                          onChange={(e) => setReason(e.target.value)} />
              </div>
            </>
          )}
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <button className="primary" type="submit" disabled={busy}>
              {mode === 'login' ? 'Zaloguj' : 'Załóż konto'}
            </button>
            <button type="button" className="subtle"
                    onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null); }}>
              {mode === 'login' ? 'Nie mam konta' : 'Mam już konto'}
            </button>
          </div>
        </div>
      </form>

      <div className="note" style={{ marginTop: 'var(--space-4)' }}>
        <p className="small" style={{ marginBottom: 0 }}>
          Logowanie przez Google i Microsoft działa w dotychczasowym panelu i jest
          przeniesione do control plane, ale wymaga publicznego adresu zwrotnego,
          którego to środowisko nie ma — dlatego nie jest tu włączone.
        </p>
      </div>
    </div>
  );
}
