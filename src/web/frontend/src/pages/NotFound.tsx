import { Link } from 'react-router-dom';

export function NotFound() {
  return (
    <div className="page">
      <div className="page-head"><h1>Nie ma takiej strony</h1></div>
      <p className="muted">
        Adres nie odpowiada żadnemu zasobowi. Jeśli trafiłeś tu z linku do runu,
        możliwe że nie masz do niego dostępu — wyniki są widoczne dla właściciela
        i administratorów.
      </p>
      <Link className="button" to="/">Wróć na stronę główną</Link>
    </div>
  );
}
