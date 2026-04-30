import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import PhantomAdvisorBrief from './PhantomAdvisorBrief.jsx'

function NavBar({ currentHash }) {
  const links = [
    { hash: '', label: 'Phantom-SoM Brief', desc: 'Advisor meeting (2026-04-30)' },
    { hash: '#weekly', label: 'Weekly Report', desc: '4-23 archive' },
  ];
  return (
    <div className="bg-slate-900 text-white px-6 py-2 text-sm flex items-center gap-4 border-b border-slate-700">
      <span className="font-semibold text-slate-300">📊 P79 Dashboard</span>
      <div className="flex gap-2">
        {links.map((l) => {
          const active = (currentHash === l.hash) || (l.hash === '' && currentHash !== '#weekly');
          return (
            <a
              key={l.hash}
              href={l.hash || '#'}
              className={`px-3 py-1 rounded transition-colors ${
                active ? 'bg-indigo-600 text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
              }`}
              title={l.desc}
            >
              {l.label}
            </a>
          );
        })}
      </div>
      <span className="ml-auto text-xs text-slate-400 font-mono">
        URL hash: <code>{currentHash || '(default)'}</code>
      </span>
    </div>
  );
}

function Router() {
  const [hash, setHash] = useState(() => window.location.hash);

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash);
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // Default → PhantomAdvisorBrief (current focus, advisor meeting deck)
  // #weekly → original weekly App
  const isWeekly = hash === '#weekly';
  return (
    <div>
      <NavBar currentHash={hash} />
      {isWeekly ? <App /> : <PhantomAdvisorBrief />}
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Router />
  </StrictMode>,
)
