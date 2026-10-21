// Self-check for the on-ice concurrency sweep used by the review timeline.
// Keep this copy in sync with onIceStats() there. Run: node tools/test_onicestats.mjs
const ON_ICE_LIMIT = 6;
function onIceStats(rows) {
    const byTeam = {};
    for (const r of rows)
        for (const iv of (r.intervals || []))
            (byTeam[r.team] = byTeam[r.team] || []).push(iv);
    const out = {};
    for (const team in byTeam) {
        const ev = [];
        for (const [s, e] of byTeam[team]) { ev.push([s, 1], [e, -1]); }
        ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        let cur = 0, max = 0, overStart = null, overSec = 0;
        const wins = [];
        for (const [t, d] of ev) {
            const prev = cur; cur += d;
            if (prev <= ON_ICE_LIMIT && cur > ON_ICE_LIMIT) overStart = t;
            if (prev > ON_ICE_LIMIT && cur <= ON_ICE_LIMIT && overStart != null) {
                wins.push([overStart, t, prev]); overSec += t - overStart; overStart = null;
            }
            if (cur > max) max = cur;
        }
        out[team] = { max, overSec: Math.round(overSec), wins };
    }
    return out;
}
const row = (team, ivs) => ({ team, intervals: ivs });
const eq = (a, b, m) => { if (a !== b) { console.error(`FAIL ${m}: ${a} !== ${b}`); process.exit(1); } };

// 1) Exactly 6 overlapping players -> within limit, no violation.
let s = onIceStats(Array.from({length: 6}, () => row('A', [[0, 10]])));
eq(s.A.max, 6, 'max=6'); eq(s.A.overSec, 0, '6 is within limit');

// 2) 7 overlapping for 10s -> violation of 10s, max 7.
s = onIceStats(Array.from({length: 7}, () => row('A', [[0, 10]])));
eq(s.A.max, 7, 'max=7'); eq(s.A.overSec, 10, '7 over limit 10s');

// 3) Touching intervals (end==start) must NOT count as overlap.
s = onIceStats([...Array.from({length: 6}, () => row('A', [[0, 5]])), row('A', [[5, 10]])]);
eq(s.A.max, 6, 'touching not overlapping'); eq(s.A.overSec, 0, 'no false violation');

// 4) Teams are independent.
s = onIceStats([...Array.from({length: 7}, () => row('A', [[0, 4]])), row('B', [[0, 100]])]);
eq(s.A.max, 7, 'A over'); eq(s.B.max, 1, 'B fine'); eq(s.B.overSec, 0, 'B no violation');

console.log('onIceStats: all checks passed');
