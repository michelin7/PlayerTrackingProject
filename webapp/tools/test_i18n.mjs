// Self-check for the language switch. A page that re-renders translated markup
// has to translate it, so its onChange listener calls applyLang again. Without
// a guard that recursed until the web view gave up: the result page, which
// rebuilds a video element per player on every pass, froze the whole window.
import assert from 'node:assert';
import fs from 'node:fs';

global.localStorage = {
    _s: {},
    getItem(k) { return this._s[k] ?? null; },
    setItem(k, v) { this._s[k] = v; },
};
global.document = {
    documentElement: {},
    querySelectorAll: () => [],
    querySelector: () => null,
    getElementById: () => null,
    addEventListener: () => {},
};
global.window = {};
eval(fs.readFileSync(new URL('../static/js/i18n.js', import.meta.url), 'utf8'));
const I = global.window.PSF_I18N;

let calls = 0;
I.onChange(() => { calls++; I.applyLang(I.getLang()); });   // what result.html does

I.applyLang('ru');
assert.strictEqual(calls, 1, 'the listener ran once, not until the stack blew');
assert.strictEqual(I.t('nav_results'), 'Результаты');

I.applyLang('en');
assert.strictEqual(calls, 2, 'a real switch still notifies');
assert.strictEqual(I.getLang(), 'en');

console.log('language switch OK');
