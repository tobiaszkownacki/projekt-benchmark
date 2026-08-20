import { useCallback, useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import { api } from './api';
import { Layout } from './components/Layout';
import { useTaskEvents } from './hooks/useEvents';
import type { User } from './types';

import { Overview } from './pages/Overview';
import { Leaderboard } from './pages/Leaderboard';
import { Runs } from './pages/Runs';
import { RunDetail } from './pages/RunDetail';
import { RunFiles } from './pages/RunFiles';
import { Submit } from './pages/Submit';
import { Docs } from './pages/Docs';
import { Compare } from './pages/Compare';
import { Admin } from './pages/Admin';
import { AdminQueue } from './pages/AdminQueue';
import { Login } from './pages/Login';
import { NotFound } from './pages/NotFound';

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    api.get<{ user: User | null }>('/api/auth/me')
      .then((response) => setUser(response.user))
      .catch(() => setUser(null));
  }, []);

  // A state change anywhere bumps one counter, and pages that care about live
  // status take it as a dependency. Cheaper and far less error-prone than a
  // per-run subscription, at a scale where the whole fleet is a hundred users.
  const onChange = useCallback(() => setRevision((value) => value + 1), []);
  useTaskEvents(onChange);

  return (
    <Layout user={user} onUserChange={setUser}>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/leaderboard" element={<Leaderboard />} />
        <Route path="/runs" element={<Runs user={user} revision={revision} />} />
        <Route path="/runs/:taskId" element={<RunDetail user={user} revision={revision} />} />
        <Route path="/runs/:taskId/files" element={<RunFiles />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/submit" element={<Submit user={user} />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="/admin" element={<Admin user={user} />} />
        <Route path="/admin/queue" element={<AdminQueue user={user} revision={revision} />} />
        <Route path="/login" element={<Login onUserChange={setUser} />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  );
}
