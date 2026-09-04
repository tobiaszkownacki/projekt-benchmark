import { useCallback, useEffect, useState } from 'react';
import { api, ApiError } from '../api';

interface State<T> { data: T | null; error: ApiError | Error | null; loading: boolean }

/** Fetch on mount and whenever `path` changes, with the stale response of an
 *  overtaken request discarded rather than allowed to win the race. */
export function useApi<T>(path: string | null, deps: unknown[] = []): State<T> & { reload: () => void } {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: path !== null });
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (path === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let current = true;
    setState((previous) => ({ ...previous, loading: true }));
    api
      .get<T>(path)
      .then((data) => { if (current) setState({ data, error: null, loading: false }); })
      .catch((error) => { if (current) setState({ data: null, error, loading: false }); });
    return () => { current = false; };
    // `deps` is spread deliberately: callers pass a revision counter so a
    // live status change refetches without threading a callback through.
  }, [path, nonce, ...deps]);

  return { ...state, reload };
}
