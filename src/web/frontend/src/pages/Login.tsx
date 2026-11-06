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
        setNotice('Account created. Awaiting administrator approval.');
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
        <h1>{mode === 'login' ? 'Sign in' : 'Create an account'}</h1>
        <p className="small">
          You can browse the leaderboard and results without an account. An account is required
          to submit your own optimizer.
        </p>
      </div>

      {notice && <div className="note note-accent" style={{ marginBottom: 16 }}>{notice}</div>}
      {error && <div className="note note-error" style={{ marginBottom: 16 }}>{error}</div>}

      <form onSubmit={submit} className="panel">
        <div className="panel-body">
          <div className="field">
            <label htmlFor="email">Email address</label>
            <input id="email" type="email" required autoComplete="username"
                   style={{ width: '100%' }} value={email}
                   onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="password">Password{mode === 'register' ? ' (min. 8 characters)' : ''}</label>
            <input id="password" type="password" required style={{ width: '100%' }}
                   autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                   value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          {mode === 'register' && (
            <>
              <div className="field">
                <label htmlFor="org">Organization</label>
                <input id="org" style={{ width: '100%' }} value={organisation}
                       onChange={(e) => setOrganisation(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="reason">Reason for joining</label>
                <textarea id="reason" rows={3} value={reason}
                          onChange={(e) => setReason(e.target.value)} />
              </div>
            </>
          )}
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <button className="primary" type="submit" disabled={busy}>
              {mode === 'login' ? 'Log in' : 'Create account'}
            </button>
            <button type="button" className="subtle"
                    onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null); }}>
              {mode === 'login' ? 'I do not have an account' : 'I already have an account'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
