"""The arena panel, injected into the page you are already scrolling.

Same layout language as the T-Rex arena (`laya_mlx/trex/app.py`): the thing being watched on
top, every number about it underneath, one window. Here the window is the browser itself and
the arena is a fixed strip along the bottom, with `document.body` padded so the panel never
covers the post you are reading.

THE NO-NAVIGATION RULE APPLIES HERE TOO: nothing in this module drives the page. It paints,
it listens, it queues. Clicks are *received*, never issued.

Two structural choices, both load-bearing rather than stylistic:

* The panel lives in an **open shadow root**. `browser.py`'s MutationObserver watches
  `document` with `subtree: true`, and it only forgives the single attribute `data-laya-i`.
  Shadow trees are invisible to it, so repainting the panel several times a second cannot
  keep resetting `window.__layaMut` and starve `settle()`. Shadow scoping also means
  LinkedIn's global CSS cannot reach in and reshape the panel.
* `update()` never builds DOM. `install()` builds every node once from the rubric; each frame
  only writes text node values, bar widths and one SVG `d` attribute, and only where the value
  actually changed. `mark()` is likewise idempotent, which matters because the virtualised
  feed re-renders the same card constantly and every repaint would otherwise be a mutation.

## The `payload` dict

One frame of the whole dashboard. Every key is optional; anything missing renders as an em
dash rather than throwing, because a frame arrives before Jev has answered far more often
than not.

```python
{
  # The post the rubric table is describing. None between posts.
  "post": {
      "urn": "urn:li:activity:7100000000000000000",
      "author": "Jane Roe",
      "permalink": "https://www.linkedin.com/feed/update/urn:li:activity:71000.../",
      "state": "scored",        # scored | pending | dropped | sponsored — tints the title rule
  },

  # question name -> per-engine cell. Keys must be rubric question names; unknown names are
  # ignored, absent names render as em dashes. `jev: None` is the normal case (every Nth post).
  #   score / noul questions read "value" (noul also reads "label": "true"/"false")
  #   choice questions read "label"
  "rubric": {
      "specificity":   {"laya": {"value": 2.7}, "jev": {"value": 2.1}},
      "actionability": {"laya": {"value": 1.4}, "jev": {"value": 1.9}},
      "hook_type":     {"laya": {"label": "contrarian"}, "jev": {"label": "curiosity_gap"}},
      "takes_a_position": {"laya": {"value": 1.0, "label": "true"}, "jev": None},
  },

  "latency":  {"laya": 14.0, "jev": 310.0},          # ms for THIS post; None where unknown
  "counters": {"seen": 214, "scored": 198, "dropped": 16, "sponsored": 3},
  "budget":   {"spent": 0.41, "cap": 2.00, "exhausted": False},   # USD
  "agreement": 0.62,        # 0..1 across the session, or None before the first Jev pair

  # Predicted-vs-actual engagement. None when the corpus has no mined pattern yet, which is
  # the day-one state, not an error — the panel then shows `note` instead.
  "band": {"predicted": 180.0, "low": 120.0, "high": 260.0, "actual": 210.0},

  "paused": False,
  "note": "no pattern mined yet",     # free text on the footer rule

  # Stage 2 charts. Lists of numbers, one entry per scored post, oldest first. None entries
  # are gaps (a post Jev did not score). Omit the key entirely and the charts stay empty.
  "series": {
      "laya": [2.1, 2.4, ...],        # mean rubric score per post
      "jev":  [None, None, 2.0, ...],
      "lat_laya": [14.0, 13.0, ...],  # ms
      "lat_jev":  [None, None, 305.0, ...],
      "agreement": [0.5, 0.55, ...],  # running rate, 0..1
  },
}
```

`poll_clicks` returns the drained queue, oldest first:

```python
[{"urn": "urn:li:activity:71000...", "ts": 1758572400123.0, "source": "card"},
 {"urn": "urn:li:activity:71000...", "ts": 1758572401456.0, "source": "save"},
 {"urn": None,                       "ts": 1758572402789.0, "source": "pause",
  "paused": True}]
```

`source` is `card` (the user clicked a post), `save` (the ★ SAVE button, always the post
currently in the table) or `pause` (the PAUSE button, which carries the new `paused` state and
no urn). The queue is bounded at 200 and drops oldest, so a driver that stops polling cannot
grow the page's memory without bound.
"""

from ..rubric import RUBRIC_VERSION, post_questions

# Serialised into the page by Playwright, so it takes its whole world as one argument and
# leaves exactly one global behind. Returns false when a panel is already installed.
PANEL_JS = r"""
(spec) => {
  if (window.__laya) return false;
  const A = 'data-laya-overlay';
  const DASH = '—';

  const mk = (tag, cls) => {
    const n = document.createElement(tag);
    n.setAttribute(A, '');
    if (cls) n.className = cls;
    return n;
  };
  const txt = (parent, value) => {
    const t = document.createTextNode(value === undefined ? DASH : value);
    parent.appendChild(t);
    return t;
  };
  const setT = (node, value) => {
    const s = (value === null || value === undefined) ? DASH : String(value);
    if (node.nodeValue !== s) node.nodeValue = s;
  };
  const setW = (node, pct) => {
    const s = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
    if (node.style.width !== s) node.style.width = s;
  };
  const setA = (node, name, value) => {
    if (node.getAttribute(name) !== value) node.setAttribute(name, value);
  };
  const num = (v, digits) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(digits) : null;

  const CSS = `
    :host { contain: layout style; }
    .panel {
      box-sizing: border-box; height: 100%; display: flex; flex-direction: column;
      padding: 10px 22px 8px; gap: 8px; overflow: hidden;
      background: var(--page); color: var(--text);
      border-top: 1px solid var(--rule);
      font: 11px/1.35 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-variant-numeric: tabular-nums;
      --page:#ffffff; --ink:#202124; --text:#5f6368; --faint:#9aa0a6;
      --rule:#e2e4e8; --track:#f1f3f4; --warn:#c5221f;
      --laya:#2a78d6; --jev:#eb6834;
    }
    @media (prefers-color-scheme: dark) {
      .panel {
        --page:#1b1f23; --ink:#e8eaed; --text:#9aa0a6; --faint:#6b7176;
        --rule:#2f343a; --track:#262b30; --warn:#f28b82;
        --laya:#6ea8f0; --jev:#f59266;
      }
    }
    .row { display: flex; align-items: baseline; gap: 14px; }
    .grow { flex: 1 1 auto; min-width: 0; }
    .title { color: var(--ink); letter-spacing: .10em; font-size: 11px; }
    .who {
      color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .cap { color: var(--faint); letter-spacing: .10em; font-size: 10px; }
    button {
      pointer-events: auto; cursor: pointer; font: inherit; letter-spacing: .10em;
      color: var(--ink); background: var(--page); border: 1px solid var(--rule);
      padding: 3px 10px;
    }
    button:hover { border-color: var(--ink); }
    hr { border: 0; border-top: 1px solid var(--rule); margin: 0; }
    hr.state { border-top-color: var(--rule); }
    hr.state[data-state="pending"]   { border-top-color: var(--faint); }
    hr.state[data-state="dropped"]   { border-top-color: var(--warn); }
    hr.state[data-state="sponsored"] { border-top-color: var(--warn); }
    hr.state[data-state="scored"]    { border-top-color: var(--ink); }

    .rubric { display: grid; grid-template-columns: 1fr 1fr; column-gap: 34px; row-gap: 1px; }
    .r {
      display: grid; align-items: center;
      grid-template-columns: 112px 1fr 34px 1fr 34px; column-gap: 8px; height: 17px;
    }
    .q { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .t { position: relative; height: 8px; background: var(--track); }
    .b { display: block; height: 100%; width: 0%; }
    .b.laya { background: var(--laya); }
    .b.jev  { background: var(--jev); }
    .lb {
      position: absolute; inset: -3px 0 auto 0; font-style: normal; color: var(--ink);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .r[data-kind="choice"] .t { background: transparent; }
    .chip { display: inline-block; width: 7px; height: 7px; }
    .chip.laya { background: var(--laya); }
    .chip.jev  { background: var(--jev); }
    .v { color: var(--ink); text-align: right; }

    .grid { display: grid; grid-template-columns: repeat(7, 1fr); column-gap: 18px; }
    .tile { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .tile .k { color: var(--faint); letter-spacing: .10em; font-size: 10px; }
    .tile .n { color: var(--ink); font-size: 15px; }
    .tile .s { color: var(--text); font-size: 10px; }

    .charts { display: grid; grid-template-columns: repeat(3, 1fr); column-gap: 34px;
              flex: 1 1 auto; min-height: 0; }
    .chart { display: flex; flex-direction: column; gap: 3px; min-height: 0; }
    .chart svg { width: 100%; flex: 1 1 auto; min-height: 0; }
    .chart path { fill: none; stroke-width: 1.4; vector-effect: non-scaling-stroke; }
    .chart .axis { stroke: var(--rule); stroke-width: 1; vector-effect: non-scaling-stroke; }
    .foot { color: var(--faint); display: flex; gap: 14px; }
    .foot .warn { color: var(--warn); }
  `;

  const host = mk('div');
  host.id = '__laya-overlay';
  Object.assign(host.style, {
    position: 'fixed', left: '0', right: '0', bottom: '0', height: '40vh',
    zIndex: '2147483647', pointerEvents: 'none', margin: '0',
  });
  const root = host.attachShadow({ mode: 'open' });
  const style = mk('style');
  style.textContent = CSS;
  root.appendChild(style);

  const panel = mk('div', 'panel');
  root.appendChild(panel);

  // -- title row -----------------------------------------------------------------------
  const head = mk('div', 'row');
  const title = mk('span', 'title');
  txt(title, 'LIVE FEED · Signal');
  const who = mk('span', 'who grow');
  const whoT = txt(who, undefined);
  const btnSave = mk('button');
  txt(btnSave, '★ SAVE');
  const btnPause = mk('button');
  const pauseT = txt(btnPause, 'PAUSE');
  head.append(title, who, btnSave, btnPause);
  const rule = mk('hr', 'state');
  setA(rule, 'data-state', 'idle');
  panel.append(head, rule);

  // -- rubric table --------------------------------------------------------------------
  const cap = mk('div', 'row');
  const capL = mk('span', 'cap');
  txt(capL, 'CURRENT POST · RUBRIC v' + spec.rubric_version);
  const capR = mk('span', 'cap grow');
  capR.style.textAlign = 'right';
  for (const engine of ['laya', 'jev']) {
    capR.append(mk('i', 'chip ' + engine));
    txt(capR, ' ' + engine.toUpperCase() + '   ');
  }
  cap.append(capL, capR);

  const table = mk('div', 'rubric');
  const rows = {};
  for (const q of spec.questions) {
    const r = mk('div', 'r');
    setA(r, 'data-kind', q.kind);
    const name = mk('span', 'q');
    txt(name, q.name);
    const cells = {};
    for (const engine of ['laya', 'jev']) {
      const track = mk('span', 't');
      const bar = mk('i', 'b ' + engine);
      const label = mk('em', 'lb');
      const labelT = txt(label, '');
      track.append(bar, label);
      const value = mk('span', 'v');
      const valueT = txt(value, undefined);
      r.append(track, value);
      cells[engine] = { bar, labelT, valueT };
    }
    r.insertBefore(name, r.firstChild);
    table.appendChild(r);
    rows[q.name] = { q, cells };
  }
  panel.append(cap, table);

  // -- stat grid -----------------------------------------------------------------------
  const grid = mk('div', 'grid');
  const tiles = {};
  const TILES = [
    ['seen', 'SEEN'], ['scored', 'SCORED'], ['dropped', 'DROPPED'],
    ['sponsored', 'SPONSORED'], ['latency', 'ANSWER TIME'], ['cost', 'API COST'],
    ['agreement', 'AGREEMENT'],
  ];
  for (const [key, label] of TILES) {
    const tile = mk('div', 'tile');
    const k = mk('span', 'k');
    txt(k, label);
    const n = mk('span', 'n');
    const nT = txt(n, undefined);
    const s = mk('span', 's');
    const sT = txt(s, '');
    tile.append(k, n, s);
    grid.appendChild(tile);
    tiles[key] = { nT, sT, s };
  }
  panel.append(mk('hr'), grid);

  // -- charts --------------------------------------------------------------------------
  const CHARTS = [
    ['scores', 'SCORES OVER SESSION', ['laya', 'jev']],
    ['latency', 'LATENCY PER POST, MS', ['lat_laya', 'lat_jev']],
    ['agreement', 'AGREEMENT', ['agreement']],
  ];
  const SVG = 'http://www.w3.org/2000/svg';
  const svgEl = (tag) => {
    const n = document.createElementNS(SVG, tag);
    n.setAttribute(A, '');
    return n;
  };
  const charts = mk('div', 'charts');
  const plots = {};
  for (const [key, label, series] of CHARTS) {
    const box = mk('div', 'chart');
    const k = mk('span', 'cap');
    txt(k, label);
    const svg = svgEl('svg');
    svg.setAttribute('viewBox', '0 0 100 30');
    svg.setAttribute('preserveAspectRatio', 'none');
    const axis = svgEl('path');
    axis.setAttribute('class', 'axis');
    axis.setAttribute('d', 'M0 29.5 L100 29.5');
    svg.appendChild(axis);
    const paths = {};
    for (const name of series) {
      const p = svgEl('path');
      p.setAttribute('d', '');
      p.setAttribute('stroke', name.endsWith('jev') ? 'var(--jev)' : 'var(--laya)');
      svg.appendChild(p);
      paths[name] = p;
    }
    const foot = mk('span', 'cap');
    const footT = txt(foot, '');
    box.append(k, svg, foot);
    charts.appendChild(box);
    plots[key] = { paths, series, footT };
  }
  panel.append(charts);

  // -- footer --------------------------------------------------------------------------
  const foot = mk('div', 'foot');
  const bandS = mk('span');
  const bandT = txt(bandS, '');
  const noteS = mk('span', 'grow');
  const noteT = txt(noteS, '');
  foot.append(bandS, noteS);
  panel.append(mk('hr'), foot);

  // -- rails, in the light DOM so they can ride real post cards -------------------------
  const railCSS = document.createElement('style');
  railCSS.setAttribute(A, '');
  railCSS.textContent = `
    [data-laya-rail] {
      position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
      pointer-events: none; z-index: 2147483646;
    }
    [data-laya-rail="laya"]      { background: #2a78d6; }
    [data-laya-rail="both"]      { background: linear-gradient(#2a78d6 50%, #eb6834 50%); }
    [data-laya-rail="pending"]   { background: #9aa0a6; }
    [data-laya-rail="dropped"]   { background: #c5221f; }
    [data-laya-rail="sponsored"] { background: repeating-linear-gradient(
                                     #c5221f 0 4px, transparent 4px 8px); }
    [data-laya-rail="saved"]     { background: #202124; width: 5px; }
  `;
  document.head.appendChild(railCSS);

  // -- click queue ---------------------------------------------------------------------
  const queue = [];
  const CARD = '[data-laya-i],[data-urn],[data-id]';
  const push = (entry) => {
    queue.push(entry);
    if (queue.length > 200) queue.splice(0, queue.length - 200);
  };
  const urnOf = (node) => node.getAttribute('data-laya-i')
    || node.getAttribute('data-urn') || node.getAttribute('data-id');

  let currentUrn = null;
  let paused = false;
  let prevPadding = '';

  const onClick = (event) => {
    const node = event.target instanceof Element ? event.target.closest(CARD) : null;
    if (!node) return;
    const urn = urnOf(node);
    if (urn) push({ urn, ts: Date.now(), source: 'card' });
  };
  document.addEventListener('click', onClick, true);

  btnSave.addEventListener('click', (event) => {
    event.stopPropagation();
    push({ urn: currentUrn, ts: Date.now(), source: 'save' });
  });
  btnPause.addEventListener('click', (event) => {
    event.stopPropagation();
    paused = !paused;
    setT(pauseT, paused ? 'RESUME' : 'PAUSE');
    push({ urn: null, ts: Date.now(), source: 'pause', paused });
  });

  // -- chart geometry ------------------------------------------------------------------
  const ceiling = (values, floor) => {
    let top = floor;
    for (const v of values) if (typeof v === 'number' && isFinite(v) && v > top) top = v;
    const mag = Math.pow(10, Math.floor(Math.log10(top)));
    return Math.ceil(top / mag) * mag;
  };
  const trace = (values, top) => {
    if (!values || values.length < 2) return '';
    const step = 100 / (values.length - 1);
    let d = '';
    let pen = 'M';
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (typeof v !== 'number' || !isFinite(v)) continue;  // Jev's every-Nth gaps join up.
      const y = 29.5 - 29 * Math.max(0, Math.min(v, top)) / top;
      d += pen + (i * step).toFixed(2) + ' ' + y.toFixed(2) + ' ';
      pen = 'L';
    }
    return d.trim();
  };

  const CELL = (side) => (side && typeof side === 'object') ? side : null;

  window.__laya = {
    host,
    root,
    update(payload) {
      payload = payload || {};
      const post = payload.post || null;
      currentUrn = post ? (post.urn || null) : null;
      setT(whoT, post ? [post.author, post.permalink].filter(Boolean).join('  ·  ') : null);
      setA(rule, 'data-state', (post && post.state) || 'idle');
      if ('paused' in payload) {
        paused = !!payload.paused;
        setT(pauseT, paused ? 'RESUME' : 'PAUSE');
      }

      const rubric = payload.rubric || {};
      for (const name in rows) {
        const { q, cells } = rows[name];
        const given = rubric[name] || {};
        for (const engine of ['laya', 'jev']) {
          const cell = CELL(given[engine]);
          const ui = cells[engine];
          if (!cell) {
            setW(ui.bar, 0);
            setT(ui.labelT, '');
            setT(ui.valueT, null);
            continue;
          }
          if (q.kind === 'choice') {
            setW(ui.bar, 0);
            setT(ui.labelT, cell.label || DASH);
            setT(ui.valueT, '');
          } else if (q.kind === 'noul') {
            const v = typeof cell.value === 'number' ? cell.value : null;
            setW(ui.bar, v === null ? 0 : v * 100);
            setT(ui.labelT, '');
            setT(ui.valueT, cell.label ? (cell.label === 'true' ? 'yes' : 'no')
                                       : (v === null ? null : (v >= 0.5 ? 'yes' : 'no')));
          } else {
            const v = typeof cell.value === 'number' ? cell.value : null;
            setW(ui.bar, v === null ? 0 : (v / q.ceiling) * 100);
            setT(ui.labelT, '');
            setT(ui.valueT, num(v, 1));
          }
        }
      }

      const c = payload.counters || {};
      setT(tiles.seen.nT, c.seen);
      setT(tiles.scored.nT, c.scored);
      setT(tiles.dropped.nT, c.dropped);
      setT(tiles.sponsored.nT, c.sponsored);

      const lat = payload.latency || {};
      const laya = num(lat.laya, 0);
      const jev = num(lat.jev, 0);
      setT(tiles.latency.nT, laya === null ? null : laya + ' ms');
      setT(tiles.latency.sT, jev === null ? 'jev ' + DASH : 'jev ' + jev + ' ms');

      const b = payload.budget || {};
      const spent = num(b.spent, 2);
      const cap2 = num(b.cap, 2);
      setT(tiles.cost.nT, spent === null ? null : '$' + spent);
      setT(tiles.cost.sT, b.exhausted ? 'budget exhausted'
                                      : (cap2 === null ? '' : 'of $' + cap2));
      const costClass = b.exhausted ? 's warn' : 's';
      if (tiles.cost.s.className !== costClass) tiles.cost.s.className = costClass;

      const agree = num(payload.agreement === null || payload.agreement === undefined
                        ? undefined : payload.agreement * 100, 0);
      setT(tiles.agreement.nT, agree === null ? null : agree + '%');

      const series = payload.series || {};
      const tops = {
        scores: ceiling([].concat(series.laya || [], series.jev || []), 3),
        latency: ceiling([].concat(series.lat_laya || [], series.lat_jev || []), 10),
        agreement: 1,
      };
      for (const key in plots) {
        const plot = plots[key];
        for (const name of plot.series) setA(plot.paths[name], 'd', trace(series[name], tops[key]));
        setT(plot.footT, key === 'agreement' ? '0–1' : '0–' + tops[key]);
      }

      const band = payload.band || null;
      if (band) {
        const lo = num(band.low, 0), hi = num(band.high, 0);
        const pred = num(band.predicted, 0), actual = num(band.actual, 0);
        setT(bandT, 'PRED ' + (pred === null ? DASH : pred)
                  + (lo === null || hi === null ? '' : ' [' + lo + '–' + hi + ']')
                  + '  ·  ACTUAL ' + (actual === null ? DASH : actual));
      } else {
        setT(bandT, '');
      }
      setT(noteT, payload.note || '');
    },

    mark(urn, state) {
      const q = JSON.stringify(String(urn));
      const card = document.querySelector(
        '[data-laya-i=' + q + '],[data-urn=' + q + '],[data-id=' + q + ']');
      if (!card) return false;
      let rail = card.firstElementChild;
      if (!rail || !rail.hasAttribute('data-laya-rail')) {
        if (getComputedStyle(card).position === 'static') card.style.position = 'relative';
        rail = mk('div');
        rail.setAttribute('data-laya-rail', state);
        card.insertBefore(rail, card.firstChild);
        return true;
      }
      setA(rail, 'data-laya-rail', state);
      return true;
    },

    drain() {
      return queue.splice(0, queue.length);
    },

    teardown() {
      document.removeEventListener('click', onClick, true);
      for (const rail of document.querySelectorAll('[data-laya-rail]')) rail.remove();
      railCSS.remove();
      host.remove();
      document.body.style.paddingBottom = prevPadding;
      delete window.__laya;
    },
  };

  prevPadding = document.body.style.paddingBottom;
  document.body.style.paddingBottom = '42vh';
  document.body.appendChild(host);
  return true;
}
"""


def _spec() -> dict:
    """The rubric, flattened to what the renderer needs: name, kind, and a bar ceiling."""
    questions = []
    for name, q in post_questions().items():
        kind = q["type"]
        if kind == "score":
            ceiling = len(q["criteria"]) - 1
        elif kind == "noul":
            ceiling = 1
        else:
            ceiling = None
        questions.append({"name": name, "kind": kind, "ceiling": ceiling})
    return {"questions": questions, "rubric_version": RUBRIC_VERSION}


def install(page) -> None:
    """Inject the panel and its styles. Calling it twice is a no-op, not a second panel.

    Called every tick rather than once, because `page.evaluate` does not survive navigation:
    the session starts on a blank tab and you navigate to your own feed, which wipes the panel
    and `window.__laya` with it. The cheap existence check keeps the repeat call free.
    """
    if page.evaluate("() => !!window.__laya"):
        return
    page.evaluate(PANEL_JS, _spec())


def update(page, payload: dict) -> None:
    """Push one frame. Writes text, bar widths and path data in place; builds no nodes."""
    page.evaluate("(p) => window.__laya.update(p)", payload)


def mark(page, urn: str, state: str) -> None:
    """Paint the thin rail on one post card. Idempotent, so repaints cost no DOM mutation.

    `state` is one of laya, both, pending, dropped, sponsored, saved. A urn that is not on
    screen right now is not an error -- the feed is virtualised and the card will come back.
    """
    page.evaluate("(a) => window.__laya.mark(a[0], a[1])", [urn, state])


def poll_clicks(page) -> list[dict]:
    """Drain the page-side click queue. Returns oldest first; empty when nothing was clicked."""
    return page.evaluate("() => window.__laya.drain()")


def teardown(page) -> None:
    """Remove the panel, the rails and the body padding. Safe when nothing was installed."""
    page.evaluate("() => { if (window.__laya) window.__laya.teardown(); }")
