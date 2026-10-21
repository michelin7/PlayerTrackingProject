// HockeyVision — global JS utilities

document.addEventListener('DOMContentLoaded', () => {
    // Smooth transitions for page loads
    document.body.style.opacity = '0';
    requestAnimationFrame(() => {
        document.body.style.transition = 'opacity 0.3s ease';
        document.body.style.opacity = '1';
    });

    // Sidebar toggle — remembers open/closed per viewer.
    const SIDEBAR_KEY = 'psf_sidebar_open';
    const toggle = document.getElementById('menuToggle');
    const backdrop = document.getElementById('sidebarBackdrop');
    const isMobile = () => window.matchMedia('(max-width: 768px)').matches;

    function setSidebar(open) {
        document.body.classList.toggle('sidebar-open', open);
        if (toggle) toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        try { localStorage.setItem(SIDEBAR_KEY, open ? '1' : '0'); } catch (e) {}
    }

    // Default: open on desktop, closed on mobile; honor saved preference.
    let saved = null;
    try { saved = localStorage.getItem(SIDEBAR_KEY); } catch (e) {}
    setSidebar(saved === null ? !isMobile() : saved === '1');

    if (toggle) toggle.addEventListener('click', () =>
        setSidebar(!document.body.classList.contains('sidebar-open')));
    if (backdrop) backdrop.addEventListener('click', () => setSidebar(false));
    // On mobile, tapping a link closes the overlay.
    document.querySelectorAll('.side-link').forEach(link =>
        link.addEventListener('click', () => { if (isMobile()) setSidebar(false); }));
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

// --- Shift list grouped by the configured squad -----------------------------
// One long alphabetical column says nothing about who plays with whom. Given the
// shift rows and the config's pairs/lines, this lays them out the way a coach
// reads a bench: per team, ALL defence pairs first, then ALL forward lines, then
// anyone the config does not place. Shared by review, result and live control so
// the three lists cannot drift apart.
window.PSF_LINES = (function () {
    const numkey = n => { const i = parseInt(n, 10); return isNaN(i) ? [1, String(n)] : [0, i]; };
    const byNumber = (a, b) => {
        const [ka, va] = numkey(a.number), [kb, vb] = numkey(b.number);
        return ka - kb || (va < vb ? -1 : va > vb ? 1 : 0);
    };

    // -> [{team, kind: 'defense'|'forward'|'rest', idx, rows:[...]}, ...]
    function group(rows, lines) {
        const byTeam = {};
        for (const r of rows || []) (byTeam[r.team] = byTeam[r.team] || []).push(r);
        const out = [];
        for (const team of Object.keys(byTeam).sort()) {
            const cfg = (lines || {})[team] || {};
            // A number is unique within a team, so it is a safe key; taking a row
            // out of the pool also stops a player listed twice in the config from
            // appearing twice on screen.
            const pool = new Map(byTeam[team].map(r => [String(r.number), r]));
            for (const kind of ['defense', 'forward']) {
                (cfg[kind] || []).forEach((nums, i) => {
                    const got = [];
                    for (const n of nums) {
                        const r = pool.get(String(n));
                        if (r) { got.push(r); pool.delete(String(n)); }
                    }
                    if (got.length) out.push({team, kind, idx: i + 1, rows: got});
                });
            }
            const rest = [...pool.values()].sort(byNumber);
            if (rest.length) out.push({team, kind: 'rest', idx: 0, rows: rest});
        }
        return out;
    }

    // "Пара 1" / "Звено 2" / "Вне состава"
    function title(sec) {
        const t = (k, v) => (window.PSF_I18N ? window.PSF_I18N.t(k, v) : k);
        if (sec.kind === 'rest') return t('sh_rest');
        return t(sec.kind === 'defense' ? 'sh_pair' : 'sh_line', {n: sec.idx});
    }

    // Builds the whole list: a team block, then its pair/line headers, then the
    // rows as the calling page draws them.
    function render(sections, rowHtml, colorOf) {
        let out = '', team = null;
        for (const sec of sections) {
            if (sec.team !== team) {
                team = sec.team;
                out += `<div class="sh-team"><span class="sh-dot" style="background:${colorOf(team)}"></span>${team}</div>`;
            }
            out += `<div class="sh-group">${title(sec)}</div>` + sec.rows.map(rowHtml).join('');
        }
        return out;
    }

    return {group, title, render};
})();


// A jersey number is PICKED from the configured squad, never typed: a number
// that is not in the roster cannot be matched to a player afterwards, and a
// typo quietly creates a phantom. Shared by Live control and review.
window.PSF_ROSTER = (function () {
    const t = (k, v) => (window.PSF_I18N ? window.PSF_I18N.t(k, v) : k);

    function numbers(roster, team) {
        const squad = (roster || {})[team] || {};
        return Object.keys(squad).sort((a, b) => (parseInt(a, 10) || 0) - (parseInt(b, 10) || 0));
    }

    // Fills a <select> with the team's squad. A current number that is NOT in
    // the squad stays selectable and is marked as such — the run already carries
    // it, and silently dropping it would rewrite the operator's own work.
    function fill(sel, roster, team, current) {
        if (!sel) return;
        const squad = (roster || {})[team] || {};
        const cur = (current === null || current === undefined) ? '' : String(current);
        const empty = !Object.keys(squad).length;
        const out = [`<option value="">${t(empty ? 'num_no_squad' : 'num_pick')}</option>`];
        if (cur && !(cur in squad))
            out.push(`<option value="${cur}">#${cur} — ${t('num_off_squad')}</option>`);
        for (const n of numbers(roster, team))
            out.push(`<option value="${n}">#${n} ${squad[n] || ''}</option>`);
        sel.innerHTML = out.join('');
        sel.value = cur;
    }

    // Team changed: same number if that team has it, nothing otherwise.
    function refill(sel, roster, team) {
        const keep = sel ? sel.value : '';
        fill(sel, roster, team, ((roster || {})[team] || {})[keep] !== undefined ? keep : '');
    }

    return {fill, refill, numbers};
})();
