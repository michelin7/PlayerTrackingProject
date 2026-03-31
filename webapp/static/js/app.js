// HockeyVision — global JS utilities

document.addEventListener('DOMContentLoaded', () => {
    // Smooth transitions for page loads
    document.body.style.opacity = '0';
    requestAnimationFrame(() => {
        document.body.style.transition = 'opacity 0.3s ease';
        document.body.style.opacity = '1';
    });

    // Active nav link highlighting
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});

// Utility: format seconds to MM:SS
function formatTime(seconds) {
    if (!seconds || seconds < 0) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// Utility: format number with locale
function formatNumber(n) {
    if (n == null) return '—';
    return Math.round(n).toLocaleString('ru-RU');
}
