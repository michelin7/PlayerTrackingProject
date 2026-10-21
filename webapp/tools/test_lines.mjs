// Self-check for the grouped shift list: defence pairs first, then forward
// lines, per team, with anyone the config does not place at the end.
// Run: node tools/test_lines.mjs
globalThis.window = {}; globalThis.document = { addEventListener: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.window.PSF_I18N = { t: (k, v) => (v ? `${k}:${v.n}` : k) };
await import('../static/js/app.js');
const L = window.PSF_LINES;
const eq = (a, b, m) => { const x = JSON.stringify(a), y = JSON.stringify(b);
    if (x !== y) throw new Error(`${m}\n got ${x}\nwant ${y}`); };
const shape = secs => secs.map(s => [s.team, s.kind, s.idx, s.rows.map(r => r.number)]);

const lines = {
    Away: {defense: [['94','4'], ['3','48']], forward: [['21','17'], ['29','95']]},
    Home: {defense: [['24','44']],            forward: [['10','14','12']]},
};
const rows = n => n.map(([team, number]) => ({team, number}));

// all D-pairs before all forward lines, team by team
eq(shape(L.group(rows([['Home','10'],['Away','17'],['Away','94'],['Home','24'],
                       ['Away','4'],['Home','14']]), lines)),
   [['Away','defense',1,['94','4']], ['Away','forward',1,['17']],
    ['Home','defense',1,['24']],     ['Home','forward',1,['10','14']]],
   'pairs must come before lines, within each team');

// config order wins over numeric order
eq(shape(L.group(rows([['Away','48'],['Away','95'],['Away','3']]), lines)),
   [['Away','defense',2,['3','48']], ['Away','forward',2,['95']]],
   'a pair keeps the order the config lists it in');

// a player with ice time but no place in the config still shows up, last
eq(shape(L.group(rows([['Away','77'],['Away','94'],['Away','1']]), lines)),
   [['Away','defense',1,['94']], ['Away','rest',0,['1','77']]],
   'unplaced players go last, sorted by number');

// no config at all -> exactly the flat list we had before
eq(shape(L.group(rows([['Home','9'],['Away','3'],['Home','10']]), {})),
   [['Away','rest',0,['3']], ['Home','rest',0,['9','10']]],
   'without a config the list stays flat');

// a player listed twice in the config must not be drawn twice
const dup = {Away: {defense: [['94']], forward: [['94','21']]}};
eq(shape(L.group(rows([['Away','94'],['Away','21']]), dup)),
   [['Away','defense',1,['94']], ['Away','forward',1,['21']]],
   'a duplicated player appears once, in its first group');

// empty groups are dropped, not rendered as blank headers
eq(shape(L.group(rows([['Home','14']]), lines)),
   [['Home','forward',1,['14']]], 'a pair with nobody on ice is not shown');

// titles come from i18n with the group number
eq(L.title({kind:'defense', idx:2}), 'sh_pair:2', 'pair title');
eq(L.title({kind:'forward', idx:3}), 'sh_line:3', 'line title');
eq(L.title({kind:'rest'}), 'sh_rest', 'rest title');

console.log('grouped shift list: all checks passed');
