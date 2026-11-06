// The saved theme before the first paint (no dark flash on the light theme). A classic script every page loads in
// its <head>, before any stylesheet applies.
try {
  if (JSON.parse(localStorage.getItem('zvs.theme') || '"dark"') === 'light') document.documentElement.dataset.theme = 'light';
} catch {}
