import { useEffect, useRef } from 'react';

export interface TaskChange {
  task_id: string;
  status: string;
  artifact: string | null;
  submitted_by: string;
}

/** One EventSource for the whole application.
 *
 *  Opened once at the root and shared, rather than per component: a connection
 *  per mounted card multiplies open streams by the size of the page for no
 *  benefit, and each one costs a Postgres LISTEN fan-out on the server. */
export function useTaskEvents(onChange: (change: TaskChange) => void): void {
  const handler = useRef(onChange);
  handler.current = onChange;

  useEffect(() => {
    let source: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      if (closed) return;
      source = new EventSource('/api/events');
      source.addEventListener('task_changed', (event) => {
        try {
          handler.current(JSON.parse((event as MessageEvent).data));
        } catch {
          // A malformed frame is not worth tearing the stream down for.
        }
      });
      source.onerror = () => {
        source?.close();
        // EventSource reconnects on its own, but not after the server closes
        // the stream cleanly, which is what happens on a redeploy.
        if (!closed) retry = setTimeout(connect, 5000);
      };
    };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      source?.close();
    };
  }, []);
}
