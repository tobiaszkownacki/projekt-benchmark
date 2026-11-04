import { Link } from 'react-router-dom';

export function NotFound() {
  return (
    <div className="page">
      <div className="page-head"><h1>Page not found</h1></div>
      <p className="muted">
        This URL does not match any resource. If you followed a link to a run, you may not have access to it. 
        Results are visible to the owner and administrators.
      </p>
      <Link className="button" to="/">Return to the home page</Link>
    </div>
  );
}
