import { Link } from 'react-router-dom';
import { api } from '../api';
import { useApi } from '../hooks/useApi';
import { ErrorNote, Loading } from '../components/Empty';
import { duration, integer, timestamp } from '../components/format';
import type { User } from '../types';

interface Pending {
  id: string; email: string; display_name: string | null;
  associated_organisation: string | null; join_reason: string | null; created_at: string;
}
interface Account {
  id: string; email: string; role: string; display_name: string | null;
  associated_organisation: string | null; is_active: boolean;
  created_at: string; last_login_at: string | null; runs: number;
}
interface BudgetRow {
  id: string; email: string; display_name: string | null; role: string;
  runs: number; failed: number; active: number; today: number;
  gradients: number; samples: number; compute_seconds: number;
}
interface SubmissionRow {
  submission_id: string; display_name: string; kind: string; builtin_name: string | null;
  status: string; family: string | null; source_sha256: string | null;
  created_at: string; submitter: string;
}

export function Admin({ user }: { user: User | null }) {
  const users = useApi<{ pending: Pending[]; users: Account[] }>(
    user?.is_admin ? '/api/admin/users' : null,
  );
  const budget = useApi<{ rows: BudgetRow[]; daily_limit: number }>(
    user?.is_admin ? '/api/admin/budget' : null,
  );
  const submissions = useApi<{ submissions: SubmissionRow[] }>(
    user?.is_admin ? '/api/admin/submissions' : null,
  );

  if (!user?.is_admin) {
    return (
      <div className="page">
        <div className="note note-warning">This page is available to administrators only.</div>
      </div>
    );
  }

  async function approve(id: string) {
    await api.post('/api/admin/users/approve', { user_id: id });
    users.reload();
  }

  return (
    <div className="page page-wide">
      <div className="page-head">
        <div className="spread">
          <div>
            <h1>Admin panel</h1>
            <p>Account approvals, budget usage, and submissions.</p>
          </div>
          <Link className="button" to="/admin/queue">Queue and cluster status →</Link>
        </div>
      </div>

      {users.loading && <Loading what="konta" />}
      {users.error && <ErrorNote error={users.error} />}

      <div className="section">
        <div className="section-head">
          <h2>Pending accounts</h2>
          <span className="small muted mono">{users.data?.pending.length ?? 0}</span>
        </div>
        {users.data && users.data.pending.length === 0 ? (
          <div className="empty">No pending accounts.</div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Email</th><th>Name</th><th>Organization</th>
                  <th>Reason for joining</th><th>Created</th><th />
                </tr>
              </thead>
              <tbody>
                {users.data?.pending.map((row) => (
                  <tr key={row.id}>
                    <td className="mono small">{row.email}</td>
                    <td className="small">{row.display_name ?? 'N/A'}</td>
                    <td className="small">{row.associated_organisation ?? 'N/A'}</td>
                    <td className="small muted">{row.join_reason ?? 'N/A'}</td>
                    <td className="mono small">{timestamp(row.created_at)}</td>
                    <td>
                      <button onClick={() => approve(row.id)}>Approve</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-head">
          <h2>Budget usage by user</h2>
          <span className="small muted">
            daily limit: <span className="mono">{budget.data?.daily_limit}</span>
          </span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>User</th><th>Role</th>
                <th className="num">Runs</th><th className="num">Failed</th>
                <th className="num">Active</th><th className="num">Today</th>
                <th className="num">Gradient evaluations</th><th className="num">Processed samples</th>
                <th className="num">Compute time</th>
              </tr>
            </thead>
            <tbody>
              {budget.data?.rows.map((row) => (
                <tr key={row.id}>
                  <td className="mono small">{row.display_name ?? row.email}</td>
                  <td className="small">{row.role}</td>
                  <td className="num">{integer(row.runs)}</td>
                  <td className="num">{integer(row.failed)}</td>
                  <td className="num">{integer(row.active)}</td>
                  <td className="num">
                    {row.today > (budget.data?.daily_limit ?? 3)
                      ? <span style={{ color: 'var(--warning)' }}>{row.today}</span>
                      : row.today}
                  </td>
                  <td className="num">{integer(row.gradients)}</td>
                  <td className="num">{integer(row.samples)}</td>
                  <td className="num small">{duration(row.compute_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note" style={{ marginTop: 'var(--space-3)' }}>
          The queue determines order, not fairness. This table shows actual consumption
          in the units tracked by the evaluator. Wall-clock time comes last because it
          is not a comparison metric.
        </div>
      </div>

      <div className="section">
        <div className="section-head"><h2>Submissions</h2></div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Name</th><th>Type</th><th>Family</th><th>Status</th>
                <th>Submitter</th><th>Source hash</th><th>Created</th><th />
              </tr>
            </thead>
            <tbody>
              {submissions.data?.submissions.map((row) => (
                <tr key={row.submission_id}>
                  <td className="mono small">{row.display_name}</td>
                  <td className="small">{row.kind}</td>
                  <td className="small">{row.family ?? 'N/A'}</td>
                  <td className="small">{row.status}</td>
                  <td className="mono small">{row.submitter}</td>
                  <td className="tiny mono dim">
                    {row.source_sha256 ? row.source_sha256.slice(0, 12) : 'N/A'}
                  </td>
                  <td className="mono small">{timestamp(row.created_at)}</td>
                  <td>
                    {row.status !== 'rejected' && (
                      <button className="subtle tiny" onClick={async () => {
                        await api.post(`/api/admin/submissions/${row.submission_id}/revoke`);
                        submissions.reload();
                      }}>Revoke</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
