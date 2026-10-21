// Self-check for the squad picker: a number is picked from the roster, never typed.
import assert from 'node:assert';
import fs from 'node:fs';

// minimal DOM stand-in for the <select> the helper fills
class Opt { constructor(v, t) { this.value = v; this.text = t; } }
class Sel {
    constructor() { this.options = []; this._v = ''; }
    set innerHTML(html) {
        this.options = [...html.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)]
            .map(m => new Opt(m[1], m[2]));
    }
    set value(v) { this._v = this.options.some(o => o.value === String(v)) ? String(v) : ''; }
    get value() { return this._v; }
}
global.window = {PSF_I18N: {t: k => k}};
const src = fs.readFileSync(new URL('../static/js/app.js', import.meta.url), 'utf8');
eval(src.slice(src.indexOf('window.PSF_ROSTER')));
const R = global.window.PSF_ROSTER;
const roster = {Home: {'10': 'Ivanov', '9': 'Petrov', '77': 'Sidorov'}, Away: {'22': 'Smith'}};

const s = new Sel();
R.fill(s, roster, 'Home', '10');
assert.deepStrictEqual(s.options.map(o => o.value), ['', '9', '10', '77'], 'sorted numerically');
assert.strictEqual(s.value, '10');

// a number the run already carries but the config does not: kept, and marked
R.fill(s, roster, 'Home', '99');
assert.strictEqual(s.value, '99');
assert.ok(s.options[1].text.includes('num_off_squad'), s.options[1].text);

// switching team offers THAT team's squad, and drops a number it does not have
R.fill(s, roster, 'Home', '10');
R.refill(s, roster, 'Away');
assert.deepStrictEqual(s.options.map(o => o.value), ['', '22']);
assert.strictEqual(s.value, '', 'a number of the other team is not carried over');

// no config at all: nothing to pick, and the page says so
R.fill(s, {}, 'Home', '');
assert.strictEqual(s.options.length, 1);
assert.ok(s.options[0].text.includes('num_no_squad'));

console.log('roster picker OK');
