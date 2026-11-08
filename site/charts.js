/* Interactive charts for the findings, drawn from data/charts.json (written by kh_figures.py from the same
   computations as the paper figures) and data/districts.geojson. Uses the page's css(), tr(), esc(), showTip(), hideTip(). */
(function(){
const {css, tr, esc, showTip, hideTip} = window.__site;
const COL = {phys: '#2160a8', dig: '#159a7a', pov: '#9c4a1f'};          // validated: CVD dE >= 10.4, normal dE >= 20.8 on the page surface
const MUTED = '#a39d91';
let CH = null;
const MINUS = s => String(s).replace(/-/g, '−');
const pctTxt = v => MINUS((v > 0 ? '+' : '') + Math.round(v) + '%');
const pTxt = p => (p < 0.001 ? '<0.001' : p.toFixed(3));
const row = (model, v) => CH.coef.find(r => r.model === model && r.var === v);
const strong = r => r.p_ri < .05 && (r.ci_hi < 0 || r.ci_lo > 0);

function svgOf(id, W, H){
  const s = d3.select('#' + id); s.selectAll('*').remove();
  return s.attr('viewBox', `0 0 ${W} ${H}`);
}
function axisStyle(g){ g.selectAll('text').attr('font-size', 11.5).style('fill', css('--ink-3')); g.select('.domain').attr('stroke', css('--rule')); g.selectAll('line').attr('stroke', css('--rule')); return g; }
function grid(g, y, x0, x1, ticks){ g.append('g').selectAll('line').data(ticks).join('line').attr('x1', x0).attr('x2', x1).attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', css('--rule')).attr('stroke-dasharray', '2 3'); }

/* ---------- decile gradient panels ---------- */
function gradPanels(id, panels, W, H){
  const svg = svgOf(id, W, H), n = panels.length, gap = 26, M = {l: 46, r: 12, t: 30, b: 44};
  const pw = (W - M.l - M.r - gap * (n - 1)) / n;
  const cap = Math.min(6, d3.max(panels.flatMap(p => p.g.hi)) * 1.05);
  const y = d3.scaleLinear().domain([0, cap]).nice().range([H - M.b, M.t]);
  panels.forEach((p, i) => {
    const x0 = M.l + i * (pw + gap), g = svg.append('g');
    const xs = p.g.x.concat(p.g.fx), x = d3.scaleLinear().domain(d3.extent(xs)).nice().range([x0, x0 + pw]);
    grid(g, y, x0, x0 + pw, y.ticks(4));
    axisStyle(g.append('g').attr('transform', `translate(0,${H - M.b})`).call((p.xticks ? d3.axisBottom(x).tickValues(p.xticks).tickFormat(p.xfmt) : d3.axisBottom(x).ticks(5)).tickSizeOuter(0)));
    if (i === 0) axisStyle(g.append('g').attr('transform', `translate(${x0},0)`).call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))).select('.domain').remove();
    g.append('text').attr('x', x0).attr('y', 14).attr('font-size', 12.5).attr('font-weight', 600).style('fill', css('--ink')).text(p.title);
    g.append('text').attr('x', x0 + pw / 2).attr('y', H - 8).attr('text-anchor', 'middle').attr('font-size', 11.5).style('fill', css('--ink-3')).text(p.xlabel);
    g.append('path').datum(p.g.fx.map((v, k) => [v, p.g.fy[k]]).filter(d => d[1] <= cap)).attr('d', d3.line().x(d => x(d[0])).y(d => y(d[1])))
      .attr('fill', 'none').attr('stroke', p.color).attr('stroke-width', 2).attr('opacity', .9);
    const last = p.g.fx.length - 1;
    g.append('text').attr('x', x0 + pw).attr('y', M.t + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('font-weight', 600).style('fill', css('--ink-2'))
      .text(MINUS(`${p.est.pct > 0 ? '+' : ''}${Math.round(p.est.pct)}% ${p.per}`));
    p.g.x.forEach((xv, k) => {
      const lo = p.g.lo[k], hi = p.g.hi[k], clipped = hi > cap;
      g.append('line').attr('x1', x(xv)).attr('x2', x(xv)).attr('y1', y(lo)).attr('y2', y(Math.min(hi, cap))).attr('stroke', MUTED).attr('stroke-width', 1.5);
      if (clipped) g.append('path').attr('d', d3.symbol(d3.symbolTriangle, 22)()).attr('transform', `translate(${x(xv)},${y(cap) + 3})`).attr('fill', MUTED);
      g.append('circle').attr('cx', x(xv)).attr('cy', y(p.g.rate[k])).attr('r', 4.6).attr('fill', p.color).attr('stroke', css('--surface')).attr('stroke-width', 2)
        .on('mousemove', ev => showTip(`<b>${tr('Decile', '10분위')} ${k + 1}</b><span class="k">${tr('Projects per 100,000 people', '인구 10만 명당 사업')}</span> ${p.g.rate[k].toFixed(2)}<br><span class="k">95% CI</span> ${lo.toFixed(2)}–${hi.toFixed(2)}`, ev))
        .on('mouseleave', hideTip);
    });
  });
  svg.append('text').attr('transform', `translate(13,${(M.t + H - M.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11.5).style('fill', css('--ink-3'))
    .text(tr('Projects per 100,000 people', '인구 10만 명당 사업 수'));
}

/* ---------- coefficient panels ---------- */
function coefChart(id, groups, W, shared = false){
  const rh = 25, M = {l: 238, r: 128, t: 6, b: 34}, head = 26;
  const H = M.t + M.b + d3.sum(groups, g => g.rows.length * rh + (g.title ? head : 6));
  const svg = svgOf(id, W, H); let yy = M.t;
  const all = groups.flatMap(g => g.rows.map(r => r.d || row(r.m, r.v)).filter(Boolean));
  groups.forEach((g, gi) => {
    const rows = g.rows.map(r => ({...r, d: r.d || row(r.m, r.v)})).filter(r => r.d);
    const src = shared ? all : rows.map(r => r.d), lo = d3.min(src, r => r.ci_lo), hi = d3.max(src, r => r.ci_hi);
    const x = d3.scaleLinear().domain([Math.min(lo, 0), Math.max(hi, 0)]).nice().range([M.l, W - M.r]);
    if (g.title) { svg.append('text').attr('x', 0).attr('y', yy + 16).attr('font-size', 12.5).attr('font-weight', 600).style('fill', g.color).text(g.title); yy += head; }
    const top = yy, bot = yy + rows.length * rh;
    svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', top).attr('y2', bot).attr('stroke', css('--ink-3')).attr('stroke-width', 1);
    rows.forEach((r, i) => {
      const cy = top + i * rh + rh / 2, c = strong(r.d) ? g.color : MUTED, main = r.main ?? (i === 0 && !shared) ?? false;
      svg.append('rect').attr('x', 0).attr('y', cy - rh / 2).attr('width', W).attr('height', rh).attr('fill', i % 2 ? 'transparent' : css('--tint')).attr('opacity', .45);
      svg.append('text').attr('x', M.l - 12).attr('y', cy + 4).attr('text-anchor', 'end').attr('font-size', 12).style('fill', css('--ink-2')).attr('font-weight', main ? 600 : 400).text(r.label);
      svg.append('line').attr('x1', x(r.d.ci_lo)).attr('x2', x(r.d.ci_hi)).attr('y1', cy).attr('y2', cy).attr('stroke', c).attr('stroke-width', 2).attr('stroke-linecap', 'round');
      svg.append(main ? 'rect' : 'circle').call(s => main
        ? s.attr('x', x(r.d.pct) - 5).attr('y', cy - 5).attr('width', 10).attr('height', 10).attr('transform', `rotate(45 ${x(r.d.pct)} ${cy})`)
        : s.attr('cx', x(r.d.pct)).attr('cy', cy).attr('r', 4.6)).attr('fill', c).attr('stroke', css('--surface')).attr('stroke-width', 2)
        .on('mousemove', ev => showTip(`<b>${esc(r.label)}</b>${pctTxt(r.d.pct)} <span class="k">95% CI</span> ${pctTxt(r.d.ci_lo)} ~ ${pctTxt(r.d.ci_hi)}<br><span class="k">p RI</span> ${pTxt(r.d.p_ri)} · <span class="k">${tr('p cluster', 'p 군집')}</span> ${pTxt(r.d.p_cluster)}`, ev))
        .on('mouseleave', hideTip);
      svg.append('text').attr('x', W - M.r + 14).attr('y', cy + 4).attr('font-size', 11.5).style('fill', css('--ink-2')).style('font-family', css('--f-mono'))
        .text(`${pctTxt(r.d.pct)}  p ${pTxt(r.d.p_ri)}`);
    });
    yy = bot;
    if (!shared || gi === groups.length - 1) {
      axisStyle(svg.append('g').attr('transform', `translate(0,${yy})`).call(d3.axisBottom(x).ticks(5).tickFormat(v => pctTxt(v)).tickSizeOuter(0)));
      yy += 30;
    }
    yy += 4;
  });
  svg.attr('viewBox', `0 0 ${W} ${yy + 4}`);
}

/* ---------- which matters more ---------- */
function importance(id){
  const W = 760, H = 196, M = {l: 132, r: 54, t: 30, b: 26}, gap = 70, pw = (W - M.l - M.r - gap) / 2;
  const svg = svgOf(id, W, H);
  const order = ['physical access', 'digital access', 'poverty', 'night lights'];
  const lab = {'physical access': tr('Physical access', '물리적 접근성'), 'digital access': tr('Digital access', '디지털 접근성'), 'poverty': tr('Poverty', '빈곤'), 'night lights': tr('Night lights', '야간 불빛')};
  const col = {'physical access': COL.phys, 'digital access': COL.dig, 'poverty': COL.pov, 'night lights': COL.pov};
  const I = order.map(k => CH.importance.find(r => r.measure === k));
  const y = d3.scaleBand().domain(order).range([M.t, H - M.b]).padding(.32);
  [['shapley_share', tr('Share of explained variation (Shapley)', '설명력 몫 (Shapley)'), [0, 100], v => Math.round(v) + '%'],
   ['cv_gain', tr('Out-of-sample gain in fit', '표본 밖 적합도 개선'), [0, 30], v => MINUS((v > 0 ? '+' : '') + v.toFixed(1) + '%')]].forEach(([k, title, dom, fmt], j) => {
    const x0 = M.l + j * (pw + gap), x = d3.scaleLinear().domain(dom).range([x0, x0 + pw]);
    svg.append('text').attr('x', x0).attr('y', 14).attr('font-size', 12.5).attr('font-weight', 600).style('fill', css('--ink')).text(title);
    grid(svg, v => v, 0, 0, []);
    axisStyle(svg.append('g').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(v => v + '%').tickSizeOuter(0)));
    I.forEach(r => {
      const c = r.measure === 'physical access' ? col[r.measure] : MUTED, v = r[k];
      svg.append('rect').attr('x', x(0)).attr('y', y(r.measure)).attr('width', Math.max(1, x(Math.max(v, 0)) - x(0))).attr('height', y.bandwidth()).attr('rx', 3).attr('fill', c)
        .on('mousemove', ev => showTip(`<b>${lab[r.measure]}</b><span class="k">${tr('Shapley share', 'Shapley 몫')}</span> ${Math.round(r.shapley_share)}%<br><span class="k">${tr('Out-of-sample gain', '표본 밖 개선')}</span> ${MINUS((r.cv_gain > 0 ? '+' : '') + r.cv_gain.toFixed(1))}% (sd ${r.cv_sd.toFixed(1)})`, ev))
        .on('mouseleave', hideTip);
      if (k === 'cv_gain') svg.append('line').attr('x1', x(Math.max(0, v - r.cv_sd))).attr('x2', x(v + r.cv_sd)).attr('y1', y(r.measure) + y.bandwidth() / 2).attr('y2', y(r.measure) + y.bandwidth() / 2).attr('stroke', css('--ink-3')).attr('stroke-width', 1.2);
      svg.append('text').attr('x', x(Math.max(v, 0) + (k === 'cv_gain' ? r.cv_sd : 0)) + 6).attr('y', y(r.measure) + y.bandwidth() / 2 + 4).attr('font-size', 11.5).style('fill', css('--ink-2')).text(fmt(v));
      if (j === 0) svg.append('text').attr('x', M.l - 10).attr('y', y(r.measure) + y.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).style('fill', css('--ink-2')).text(lab[r.measure]);
    });
  });
}

/* ---------- district types ---------- */
function types(id){
  const W = 660, H = 300, M = {l: 46, r: 14, t: 18, b: 62}, svg = svgOf(id, W, H);
  const lab = {capital: tr('Phnom Penh', '프놈펜'), town: tr('Provincial towns', '도청 소재지'), rural1: tr('Rural: closest third', '농촌: 가까운 1/3'), rural2: tr('Rural: middle third', '농촌: 중간 1/3'), rural3: tr('Rural: farthest third', '농촌: 먼 1/3')};
  const x = d3.scaleBand().domain(CH.types.map(d => d.key)).range([M.l, W - M.r]).padding(.38);
  const cap = 10, y = d3.scaleLinear().domain([0, cap]).range([H - M.b, M.t]);
  grid(svg, y, M.l, W - M.r, y.ticks(5));
  axisStyle(svg.append('g').attr('transform', `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0))).select('.domain').remove();
  CH.types.forEach(d => {
    const urban = d.key === 'capital' || d.key === 'town', c = urban ? css('--ink') : MUTED, cx = x(d.key) + x.bandwidth() / 2;
    svg.append('path').attr('d', `M${x(d.key)},${y(0)} V${y(d.rate) + 3} q0,-3 3,-3 H${x(d.key) + x.bandwidth() - 3} q3,0 3,3 V${y(0)} Z`).attr('fill', c)
      .on('mousemove', ev => showTip(`<b>${lab[d.key]}</b><span class="k">${tr('Projects per 100,000 people', '인구 10만 명당 사업')}</span> ${d.rate.toFixed(2)}<br><span class="k">95% CI</span> ${d.lo.toFixed(2)}–${d.hi.toFixed(2)}<br><span class="k">${tr('Districts', '군')}</span> ${d.n} · <span class="k">${tr('median travel time', '이동시간 중앙값')}</span> ${d.tt} ${tr('min', '분')}`, ev))
      .on('mouseleave', hideTip);
    svg.append('line').attr('x1', cx).attr('x2', cx).attr('y1', y(d.lo)).attr('y2', y(Math.min(d.hi, cap))).attr('stroke', css('--ink-3')).attr('stroke-width', 1.3);
    if (d.hi > cap) svg.append('path').attr('d', d3.symbol(d3.symbolTriangle, 22)()).attr('transform', `translate(${cx},${y(cap) + 3})`).attr('fill', css('--ink-3'));
    svg.append('text').attr('x', x(d.key) + x.bandwidth() + 4).attr('y', y(d.rate) + 4).attr('font-size', 12).attr('font-weight', 600).style('fill', css('--ink')).text(d.rate.toFixed(2));
    svg.append('text').attr('x', cx).attr('y', H - M.b + 18).attr('text-anchor', 'middle').attr('font-size', 11.5).style('fill', css('--ink-2')).text(lab[d.key]);
    svg.append('text').attr('x', cx).attr('y', H - M.b + 34).attr('text-anchor', 'middle').attr('font-size', 10.5).style('fill', css('--ink-3')).text(`${d.n} ${tr('districts', '개 군')} · ${d.tt} ${tr('min', '분')}`);
  });
  svg.append('text').attr('transform', `translate(13,${(M.t + H - M.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11.5).style('fill', css('--ink-3')).text(tr('Projects per 100,000 people', '인구 10만 명당 사업 수'));
}

/* ---------- remote, unconnected, poor ---------- */
function overlap(id, geo){
  const d = geo.features.map(f => f.properties).filter(p => p.tt_town_dry > 0 && p.phone_pct != null && p.dep_z != null && p.pop_2019 > 1000 && !/Lake/i.test(p.name));
  const W = 760, H = 290, M = {l: 48, r: 12, t: 30, b: 44}, gap = 44, pw = (W - M.l - M.r - gap) / 2, svg = svgOf(id, W, H);
  const x = x0 => d3.scaleLog().domain([3, 700]).range([x0, x0 + pw]);
  const r = d3.scaleSqrt().domain([0, d3.max(d, p => p.N)]).range([2.5, 13]);
  const corr = (a, b) => { const ma = d3.mean(a), mb = d3.mean(b); return d3.sum(a, (v, i) => (v - ma) * (b[i] - mb)) / Math.sqrt(d3.sum(a, v => (v - ma) ** 2) * d3.sum(b, v => (v - mb) ** 2)); };
  const dots = [];
  [['phone_pct', tr('Digital access: households with a phone (%)', '디지털 접근성: 휴대전화 보유 가구 (%)'), COL.dig, v => v.toFixed(1) + '%'],
   ['dep_z', tr('Poverty: census deprivation (SD)', '빈곤: 센서스 박탈 (SD)'), COL.pov, v => v.toFixed(2)]].forEach(([k, title, c, f], j) => {
    const x0 = M.l + j * (pw + gap), X = x(x0), y = d3.scaleLinear().domain(d3.extent(d, p => p[k])).nice().range([H - M.b, M.t]);
    grid(svg, y, x0, x0 + pw, y.ticks(4));
    axisStyle(svg.append('g').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(X).tickValues([10, 30, 100, 300]).tickFormat(d3.format('~g')).tickSizeOuter(0)));
    axisStyle(svg.append('g').attr('transform', `translate(${x0},0)`).call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))).select('.domain').remove();
    svg.append('text').attr('x', x0).attr('y', 14).attr('font-size', 12.5).attr('font-weight', 600).style('fill', css('--ink')).text(title);
    svg.append('text').attr('x', x0 + pw / 2).attr('y', H - 8).attr('text-anchor', 'middle').attr('font-size', 11.5).style('fill', css('--ink-3')).text(tr('travel time to the nearest town (min, log scale)', '가장 가까운 도시까지 이동시간 (분, 로그)'));
    const rr = corr(d.map(p => Math.log(p.tt_town_dry)), d.map(p => p[k]));
    svg.append('text').attr('x', x0 + pw).attr('y', M.t + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('font-weight', 600).style('fill', css('--ink-2')).text(MINUS(`r = ${rr.toFixed(2)}`));
    dots.push(svg.append('g').selectAll('circle').data(d.slice().sort((a, b) => b.N - a.N)).join('circle').attr('cx', p => X(p.tt_town_dry)).attr('cy', p => y(p[k])).attr('r', p => r(p.N))
      .attr('fill', c).attr('fill-opacity', .28).attr('stroke', c).attr('stroke-width', 1)
      .on('mousemove', (ev, p) => { hl(p.unit); showTip(`<b>${esc(p.name)}</b><span class="k">${tr('Travel time', '이동시간')}</span> ${Math.round(p.tt_town_dry)} ${tr('min', '분')}<br><span class="k">${tr('Phones', '휴대전화')}</span> ${p.phone_pct.toFixed(1)}% · <span class="k">${tr('Deprivation', '박탈')}</span> ${p.dep_pct.toFixed(1)}%<br><span class="k">${tr('Projects', '사업')}</span> ${p.N.toFixed(1)}`, ev); })
      .on('mouseleave', () => { hl(null); hideTip(); }));
  });
  function hl(u){ dots.forEach(s => s.attr('fill-opacity', p => u == null ? .28 : p.unit === u ? .9 : .08).attr('stroke-opacity', p => u == null || p.unit === u ? 1 : .2)); }
}

/* ---------- Phnom Penh or the nearest town (spec 9-33) ---------- */
const WORK = {'NR48 Koh Kong bridges': ['NR48 Koh Kong bridges', 'NR48 꼬꽁 교량'], 'Sekong bridge': ['Sekong bridge', '세콩교'], 'Prek Tamak bridge': ['Prek Tamak bridge', '쁘렉따막교'],
  'NR78 Ratanakiri paving': ['NR78 paving', 'NR78 포장'], 'Stung Treng bridge': ['Stung Treng bridge', '스퉁트렝교'], 'Neak Loeung bridge': ['Neak Loeung bridge', '네악루엉교']};
const PROV = {'Prey Veng': ['Prey Veng', '프레이벵'], 'Ratanakiri': ['Ratanakiri', '라타나끼리'], 'Svay Rieng': ['Svay Rieng', '스바이리엥']};
const wl = k => tr(...(WORK[k] || [k, k])), pl = k => tr(...(PROV[k] || [k, k]));
function capitalChart(id){
  const rowsDef = [['all districts', 'zone', tr('All 141 districts', '141개 군 전체'), true], ['Phnom Penh khans excluded', 'zone', tr('Without Phnom Penh', '프놈펜 제외'), true],
    ['donors EU/US', 'zone', tr('EU and US donors', '유럽·미국 공여자')], ['donors China', 'zone', tr('China', '중국')], ['sector social', 'zone', tr('Social sectors', '사회 분야')]];
  const R = rowsDef.map(([s, fe, label, main]) => ({...CH.capital.find(r => r.sample === s && r.fe === fe), label, main})).filter(r => r.cv_town != null);
  const W = 720, rh = 46, M = {l: 168, r: 70, t: 34, b: 34}, H = M.t + M.b + R.length * rh, svg = svgOf(id, W, H);
  const x = d3.scaleLinear().domain([-10, 45]).range([M.l, W - M.r]), CAP = css('--ink-2');
  svg.append('g').selectAll('line').data(x.ticks(6)).join('line').attr('x1', d => x(d)).attr('x2', d => x(d)).attr('y1', M.t - 6).attr('y2', H - M.b).attr('stroke', css('--rule')).attr('stroke-dasharray', d => d === 0 ? null : '2 3');
  axisStyle(svg.append('g').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(v => pctTxt(v)).tickSizeOuter(0)));
  [[COL.phys, tr('Travel time to the nearest town', '가장 가까운 도시까지 이동시간')], [CAP, tr('Travel time to Phnom Penh', '프놈펜까지 이동시간')]].forEach(([c, t], i) => {
    const lx = M.l + i * 250; svg.append('rect').attr('x', lx).attr('y', 6).attr('width', 12).attr('height', 12).attr('rx', 3).attr('fill', c);
    svg.append('text').attr('x', lx + 18).attr('y', 16).attr('font-size', 12).style('fill', css('--ink-2')).text(t);
  });
  R.forEach((r, i) => {
    const y0 = M.t + i * rh;
    if (i % 2 === 0) svg.append('rect').attr('x', 0).attr('y', y0).attr('width', W).attr('height', rh).attr('fill', css('--tint')).attr('opacity', .45).lower();
    svg.append('text').attr('x', M.l - 12).attr('y', y0 + rh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('font-weight', r.main ? 600 : 400).style('fill', css('--ink-2')).text(r.label);
    [['town', COL.phys, 9], ['capital', CAP, 25]].forEach(([k, c, off]) => {
      const v = r['cv_' + k], sd = r['cv_' + k + '_sd'], yy = y0 + off - 4;
      svg.append('rect').attr('x', x(Math.min(0, v))).attr('y', yy).attr('width', Math.max(1, Math.abs(x(v) - x(0)))).attr('height', 12).attr('rx', 3).attr('fill', c)
        .on('mousemove', ev => showTip(`<b>${esc(r.label)}</b><span class="k">${k === 'town' ? tr('Nearest town', '가까운 도시') : tr('Phnom Penh', '프놈펜')}</span> ${MINUS((v > 0 ? '+' : '') + v.toFixed(1))}% (sd ${sd.toFixed(1)})<br><span class="k">Shapley</span> ${Math.round(r['shapley_' + k])}%`, ev))
        .on('mouseleave', hideTip);
      svg.append('line').attr('x1', x(v - sd)).attr('x2', x(v + sd)).attr('y1', yy + 6).attr('y2', yy + 6).attr('stroke', css('--ink-3')).attr('stroke-width', 1.1);
      svg.append('text').attr('x', x(Math.max(v + sd, 0)) + 6).attr('y', yy + 10).attr('font-size', 11.5).style('fill', css('--ink-2')).style('font-family', css('--f-mono')).text(MINUS((v > 0 ? '+' : '') + v.toFixed(1) + '%'));
    });
  });
}

/* ---------- where the works cut travel time (spec 9-34) ---------- */
function worksMap(id, geo){
  const W = 760, H = 560, svg = svgOf(id, W, H), G = CH.wgrid;
  const proj = d3.geoMercator().fitExtent([[10, 10], [W - 190, H - 10]], geo), path = d3.geoPath(proj);
  svg.append('g').selectAll('path').data(geo.features).join('path').attr('d', path).attr('fill', css('--tint')).attr('stroke', css('--rule')).attr('stroke-width', .6);
  const ramp = d3.scaleSequentialLog(d3.interpolateRgb(css('--surface'), COL.phys)).domain([3, 240]).clamp(true);
  const h = G.step / 2, g = svg.append('g');
  G.cells.forEach(([lon, lat, v, w]) => {
    const [x0, y0] = proj([lon - h, lat + h]), [x1, y1] = proj([lon + h, lat - h]);
    g.append('rect').attr('x', x0).attr('y', y0).attr('width', x1 - x0 + .3).attr('height', y1 - y0 + .3).attr('fill', ramp(v))
      .on('mousemove', ev => showTip(`<b>${wl(G.works[w])}</b>${tr('Without it, Phnom Penh is', '없으면 프놈펜까지')} <b>+${v}</b> ${tr('min away', '분 더 걸림')}`, ev)).on('mouseleave', hideTip);
  });
  svg.append('g').selectAll('path').data(geo.features).join('path').attr('d', path).attr('fill', 'none').attr('stroke', css('--surface')).attr('stroke-width', .5).attr('opacity', .7).style('pointer-events', 'none');
  const pp = proj([104.928, 11.556]);
  svg.append('circle').attr('cx', pp[0]).attr('cy', pp[1]).attr('r', 5).attr('fill', css('--ink')).attr('stroke', css('--surface')).attr('stroke-width', 2);
  svg.append('text').attr('x', pp[0] - 9).attr('y', pp[1] + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('font-weight', 600).style('fill', css('--ink')).text(tr('Phnom Penh', '프놈펜'));
  const dx = {'NR48 Koh Kong bridges': [10, 20, 'start'], 'Sekong bridge': [10, 20, 'start'], 'Prek Tamak bridge': [12, -6, 'start'], 'NR78 Ratanakiri paving': [10, -14, 'start'], 'Stung Treng bridge': [-12, 4, 'end'], 'Neak Loeung bridge': [12, 16, 'start']};
  CH.wsites.forEach(s => {
    const [x, y] = proj([s.lon, s.lat]), [ox, oy, an] = dx[s.work] || [10, 0, 'start'];
    svg.append('path').attr('d', d3.symbol(d3.symbolDiamond, 70)()).attr('transform', `translate(${x},${y})`).attr('fill', css('--surface')).attr('stroke', css('--ink')).attr('stroke-width', 1.6);
    const t = svg.append('text').attr('x', x + ox).attr('y', y + oy).attr('text-anchor', an).attr('font-size', 12).style('fill', css('--ink')).style('paint-order', 'stroke').attr('stroke', css('--surface')).attr('stroke-width', 3.5);
    t.append('tspan').attr('font-weight', 600).text(wl(s.work)); t.append('tspan').attr('dx', 5).style('fill', css('--ink-3')).text(s.year);
  });
  const lx = W - 178, ly = 40, lw = 14, lh = 180, lg = svg.append('g');
  lg.append('text').attr('x', lx).attr('y', ly - 14).attr('font-size', 12).attr('font-weight', 600).style('fill', css('--ink')).text(tr('Minutes added to Phnom Penh', '프놈펜까지 늘어나는 시간'));
  lg.append('text').attr('x', lx).attr('y', ly).attr('font-size', 11).style('fill', css('--ink-3')).text(tr('if the work did not exist', '공사가 없다면'));
  lg.append('text').attr('x', lx).attr('y', ly + 14).attr('font-size', 11).style('fill', css('--ink-3')).text(tr('(a bridge becomes a 60-min ferry)', '(다리는 60분 나룻배로)'));
  const ys = d3.scaleLog().domain([3, 240]).range([ly + 26 + lh, ly + 26]);
  d3.range(0, 1.0001, .02).forEach(t => { const v = 3 * Math.pow(80, t); lg.append('rect').attr('x', lx).attr('y', ys(v * Math.pow(80, .02))).attr('width', lw).attr('height', Math.abs(ys(v) - ys(v * Math.pow(80, .02))) + .5).attr('fill', ramp(v)); });
  axisStyle(lg.append('g').attr('transform', `translate(${lx + lw},0)`).call(d3.axisRight(ys).tickValues([5, 15, 30, 60, 120, 240]).tickFormat(d3.format('~g')).tickSizeOuter(0))).select('.domain').remove();
}

/* ---------- treated provinces, year by year (spec 9-34) ---------- */
function worksPanels(id){
  const P = CH.wpanel, n = P.treated.length, W = 780, H = 250, M = {l: 40, r: 10, t: 46, b: 34}, gap = 22, pw = (W - M.l - M.r - gap * (n - 1)) / n, svg = svgOf(id, W, H);
  const yMax = Math.max(4, d3.max(P.treated.flatMap(t => t.y)), d3.max(P.control)), y = d3.scaleLinear().domain([0, yMax]).nice().range([H - M.b, M.t]);
  const coef = m => (CH.wcoef || []).find(r => r.model === m);
  P.treated.forEach((t, i) => {
    const x0 = M.l + i * (pw + gap), x = d3.scaleBand().domain(P.years).range([x0, x0 + pw]).padding(.18), g = svg.append('g'), cx = yr => x(yr) + x.bandwidth() / 2;
    g.append('rect').attr('x', x(t.year - 5)).attr('y', M.t).attr('width', x(t.year + 5) + x.bandwidth() - x(t.year - 5)).attr('height', H - M.b - M.t).attr('fill', css('--tint')).attr('opacity', .6);
    grid(g, y, x0, x0 + pw, y.ticks(4));
    g.append('line').attr('x1', cx(t.year)).attr('x2', cx(t.year)).attr('y1', M.t - 4).attr('y2', H - M.b).attr('stroke', css('--ink')).attr('stroke-dasharray', '3 3');
    g.append('text').attr('x', cx(t.year) + 4).attr('y', M.t + 8).attr('font-size', 11).style('fill', css('--ink-2')).text(tr('opened', '개통') + ' ' + t.year);
    t.y.forEach((v, k) => {
      const yr = P.years[k];
      if (v > 0) g.append('path').attr('d', `M${x(yr)},${y(0)} V${y(v) + 2} q0,-2 2,-2 H${x(yr) + x.bandwidth() - 2} q2,0 2,2 V${y(0)} Z`).attr('fill', yr === t.year ? MUTED : COL.phys);
      g.append('rect').attr('x', x(yr)).attr('y', M.t).attr('width', x.bandwidth()).attr('height', H - M.b - M.t).attr('fill', 'transparent')
        .on('mousemove', ev => showTip(`<b>${pl(t.province)} · ${yr}</b><span class="k">${tr('Activities', '활동')}</span> ${v.toFixed(2)}<br><span class="k">${tr('Control provinces, mean', '대조 주 평균')}</span> ${P.control[k].toFixed(2)}`, ev)).on('mouseleave', hideTip);
    });
    g.append('path').datum(P.control.map((v, k) => [P.years[k], v])).attr('d', d3.line().x(d => cx(d[0])).y(d => y(d[1]))).attr('fill', 'none').attr('stroke', css('--ink-3')).attr('stroke-width', 1.6).style('pointer-events', 'none');
    axisStyle(g.append('g').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).tickValues(P.years.filter(v => v % 5 === 0)).tickSizeOuter(0)));
    if (i === 0) axisStyle(g.append('g').attr('transform', `translate(${x0},0)`).call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))).select('.domain').remove();
    g.append('text').attr('x', x0).attr('y', 14).attr('font-size', 12.5).attr('font-weight', 600).style('fill', css('--ink')).text(pl(t.province));
    const c = coef('event: ' + t.work);
    g.append('text').attr('x', x0).attr('y', 30).attr('font-size', 11.5).style('fill', css('--ink-3'))
      .text(`${wl(t.work)} · ${tr('5 yrs before', '전 5년')} ${c ? c.treated_pre.toFixed(1) : ''} → ${tr('after', '후')} ${c ? c.treated_post.toFixed(1) : ''}`);
  });
  svg.append('text').attr('transform', `translate(12,${(M.t + H - M.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11.5).style('fill', css('--ink-3')).text(tr('Activities per year', '연간 활동 수'));
}

/* ---------- draw all ---------- */
function drawAll(geo){
  const PAIR = 'pair: physical access + digital access (r = -0.62)';
  const sd = tr('per SD', '/ SD'), dbl = tr('per doubling', '/ 두 배');
  if (document.getElementById('ch-travel')) gradPanels('ch-travel', [{g: CH.grad.travel, est: row('physical access alone', 'reach'), color: COL.phys, title: tr('Physical access: travel time to the nearest town', '물리적 접근성: 가장 가까운 도시까지 이동시간'),
    xlabel: tr('travel time relative to the province average', '주 평균 대비 이동시간'), xticks: [-2, -1, 0, 1, 2], xfmt: v => ({'-2': '¼×', '-1': '½×', '0': tr('avg', '평균'), '1': '2×', '2': '4×'})[v], per: dbl}], 660, 290);
  if (CH.grad.phone && document.getElementById('ch-digital')) gradPanels('ch-digital', [{g: CH.grad.phone, est: row('digital access alone', 'phone'), color: COL.dig, title: tr('Digital access: phone ownership 2019', '디지털 접근성: 2019년 휴대전화 보유율'), xlabel: tr('phone ownership vs province mean (SD)', '주 평균 대비 보유율 (SD)'), per: sd}], 660, 290);
  if (CH.grad.deprivation && document.getElementById('ch-need')) gradPanels('ch-need', [
    {g: CH.grad.deprivation, est: row('poverty alone', 'deprivation'), color: COL.pov, title: tr('Poverty: census deprivation 2019', '빈곤: 2019년 센서스 박탈'), xlabel: tr('deprivation vs province mean (SD)', '주 평균 대비 박탈 (SD)'), per: sd},
    {g: CH.grad.lights, est: row('need check: night lights per person 2015', 'lights'), color: COL.pov, title: tr('Economic activity: night lights 2015', '경제활동: 2015년 야간 불빛'), xlabel: tr('night lights vs province mean (SD)', '주 평균 대비 야간 불빛 (SD)'), per: sd}], 780, 290);
  coefChart('ch-robust', [
    {color: COL.phys, rows: [{m: 'physical access alone', v: 'reach', main: true, label: tr('Main estimate (141 districts)', '주 추정치 (141개 군)')}]},
    {title: tr('Robustness', '견고성 점검'), color: COL.phys, rows: [
      {m: 'R1a geocodes as published (no reclassification)', v: 'reach', label: tr('Geocodes as published', '지오코딩 원자료 그대로')},
      {m: 'R1b worst case: province-level geocodes to remotest district', v: 'reach', label: tr('Reclassified projects in remotest district', '재분류 사업을 가장 외진 군에')},
      {m: 'R2a province-only projects added at remotest district', v: 'reach', label: tr('+ province-only projects, remotest', '+ 주 단위 사업, 가장 외진 군')},
      {m: 'R2b province-only projects added at best-reached district', v: 'reach', label: tr('+ province-only projects, best-reached', '+ 주 단위 사업, 가장 가까운 군')},
      {m: 'R3 Phnom Penh excluded', v: 'reach', label: tr('Phnom Penh excluded', '프놈펜 제외')},
      {m: 'R4 Weiss et al. (2018) travel time to cities', v: 'reach_weiss', label: tr('Weiss et al. (2018) map', 'Weiss et al. (2018) 지도')}]},
    {title: tr('Donor groups', '공여자 그룹'), color: COL.phys, rows: [
      {m: 'donors: EU/US', v: 'reach', label: tr('EU and US donors', '유럽·미국')}, {m: 'donors: World Bank', v: 'reach', label: tr('World Bank (12 projects)', '세계은행 (12건)')}, {m: 'donors: China', v: 'reach', label: tr('China', '중국')}]}], 720, true);
  coefChart('ch-coef3', [
    {title: tr('Physical access (travel time ×2)', '물리적 접근성 (이동시간 ×2)'), color: COL.phys, rows: [
      {m: 'physical access alone', v: 'reach', label: tr('Main estimate', '주 추정치')}, {m: PAIR, v: 'reach', label: tr('Holding digital access fixed', '디지털 접근성 고정')},
      {m: 'R4 Weiss et al. (2018) travel time to cities', v: 'reach_weiss', label: tr('Weiss et al. (2018) map', 'Weiss et al. (2018) 지도')}, {m: 'rural districts: physical access alone', v: 'reach', label: tr('Rural districts only', '농촌 군만')}]},
    {title: tr('Digital access (+1 SD)', '디지털 접근성 (+1SD)'), color: COL.dig, rows: [
      {m: 'digital access alone', v: 'phone', label: tr('Main estimate: phones 2019', '주 추정치: 2019 휴대전화')}, {m: 'digital check: phone ownership 2008', v: 'phone_2008', label: tr('Phones 2008 (before the projects)', '2008 휴대전화 (사업 이전)')},
      {m: 'digital check: Ookla mobile speed', v: 'speed', label: tr('Ookla mobile speed', 'Ookla 모바일 속도')}, {m: PAIR, v: 'phone', label: tr('Holding physical access fixed', '물리적 접근성 고정')},
      {m: 'rural districts: digital access alone', v: 'phone', label: tr('Rural districts only', '농촌 군만')}]},
    {title: tr('Poverty and economic activity (+1 SD)', '빈곤과 경제활동 (+1SD)'), color: COL.pov, rows: [
      {m: 'poverty alone', v: 'deprivation', label: tr('Main estimate: deprivation 2019', '주 추정치: 2019 박탈')}, {m: 'poverty check: 2008 index', v: 'dep_2008', label: tr('Deprivation 2008', '2008 박탈')},
      {m: 'poverty check: without water (piped only)', v: 'dep_nowater', label: tr('Deprivation without water', '식수 제외 박탈')}, {m: 'poverty check: Meta RWI', v: 'poverty', label: tr('Meta wealth index (negative)', 'Meta 부 지수 (음수)')},
      {m: 'need check: night lights per person 2015', v: 'lights', label: tr('Night lights 2015 (richer = more)', '2015 야간 불빛 (밝을수록 부유)')}, {m: 'rural districts: poverty alone', v: 'deprivation', label: tr('Rural districts only', '농촌 군만')}]}], 720);
  if (CH.importance && document.getElementById('ch-importance')) importance('ch-importance');
  types('ch-types');
  if (geo && document.getElementById('ch-overlap')) overlap('ch-overlap', geo);
  if (CH.capital && document.getElementById('ch-capital')) capitalChart('ch-capital');
  if (CH.designs && document.getElementById('ch-designs')) {
    const dz = m => CH.designs.find(r => r.model === m);
    coefChart('ch-designs', [
      {title: tr('Only adjacent districts compared', '맞닿은 군끼리만 비교'), color: COL.phys, rows: [
        {d: dz('travel time x2, neighbour pairs'), main: true, label: tr('Travel time ×2, 361 neighbour pairs', '이동시간 ×2, 이웃 군 361쌍')}]},
      {title: tr('Distance to a town, or the roads?', '도시까지 거리인가, 길인가?'), color: COL.phys, rows: [
        {d: dz('straight-line distance x2 (detour fixed)'), label: tr('Straight-line distance ×2', '직선거리 ×2')},
        {d: dz('road detour x2 (distance fixed)'), label: tr('Road detour ×2 (minutes per km)', '도로 우회 ×2 (km당 분)')}]},
      {title: tr('Type of project', '사업 유형'), color: COL.phys, rows: [
        {d: dz('field projects alone, travel time x2'), label: tr('Field projects (water, schools, farms)', '현장 사업 (수자원, 학교, 농업)')},
        {d: dz('institutional projects alone, travel time x2'), label: tr('Institutional projects', '기관 사업')},
        {d: dz('infrastructure projects alone, travel time x2'), label: tr('Infrastructure (23 projects)', '인프라 (23건)')}]},
      {title: tr('Donor', '공여자'), color: COL.phys, rows: [
        {d: dz('EU/US alone, travel time x2'), label: tr('EU and US donors', '유럽·미국')},
        {d: dz('China alone, travel time x2'), label: tr('China', '중국')}]}], 720, true);
  }
  if (CH.hubs && document.getElementById('ch-hubs')) {
    const hz = (t, s) => CH.hubs.find(r => r.test === t && r.sample === s);
    coefChart('ch-hubs', [
      {title: tr('Travel time ×2, districts compared within a province', '이동시간 ×2, 같은 주 안의 군 비교'), color: COL.phys, rows: [
        {d: hz('travel time x2', 'all districts'), main: true, label: tr('All 141 districts', '전체 141개 군')},
        {d: hz('travel time x2', 'rural districts only'), label: tr('112 rural districts only', '농촌 112개 군만')}]},
      {title: tr('Travel time ×2, adjacent districts only', '이동시간 ×2, 맞닿은 군끼리만'), color: COL.phys, rows: [
        {d: hz('travel time x2, neighbour pairs', 'all pairs'), label: tr('All 361 neighbour pairs', '이웃 군 361쌍 전체')},
        {d: hz('travel time x2, neighbour pairs', 'pairs with a hub'), label: tr('104 pairs with a hub', '거점이 낀 104쌍')},
        {d: hz('travel time x2, neighbour pairs', 'rural-rural pairs'), label: tr('257 rural–rural pairs', '농촌–농촌 257쌍')}]},
      {title: tr('Distance and road detour, in one model', '직선거리와 도로 우회, 한 모형'), color: COL.phys, rows: [
        {d: hz('straight-line distance x2', 'all districts'), label: tr('Distance ×2, all districts', '거리 ×2, 전체')},
        {d: hz('road detour x2', 'all districts'), label: tr('Detour ×2, all districts', '우회 ×2, 전체')},
        {d: hz('straight-line distance x2', 'rural districts only'), label: tr('Distance ×2, rural only', '거리 ×2, 농촌만')},
        {d: hz('road detour x2', 'rural districts only'), label: tr('Detour ×2, rural only', '우회 ×2, 농촌만')}]}], 720, true);
  }
  if (geo && CH.wgrid && document.getElementById('ch-works')) worksMap('ch-works', geo);
  if (CH.wpanel && document.getElementById('ch-events')) worksPanels('ch-events');
}
window.drawCharts = function(geo){
  if (CH) return drawAll(geo);
  // no-cache: revalidate with the server, because charts.json changes more often than this script's version tag
  fetch('data/charts.json', {cache: 'no-cache'}).then(r => r.json()).then(j => { CH = j; drawAll(geo); }).catch(e => console.error('charts.json', e));
};
if (window.__site.geo()) window.drawCharts(window.__site.geo());
})();
