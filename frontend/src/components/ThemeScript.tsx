/**
 * ThemeScript - Inline script to initialize theme before React hydration
 * Prevents flash of wrong theme by applying the correct class immediately
 */
export function ThemeScript() {
  // This script runs inline before React hydrates, preventing FOUC
  const themeScript = `
(function() {
  try {
    var theme = localStorage.getItem('atlas-theme');
    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    // Default to LIGHT for first-time visitors; only honor system dark when explicitly selected.
    var shouldBeDark = theme === 'dark' || (theme === 'system' && prefersDark);
    if (shouldBeDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  } catch (e) {}
})();
`;

  return (
    <script
      dangerouslySetInnerHTML={{ __html: themeScript }}
      suppressHydrationWarning
    />
  );
}
