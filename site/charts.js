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
  const all = groups.flatMap(g => g.rows.map(r => row(r.m, r.v)).filter(Boolean));
  groups.forEach((g, gi) => {
    const rows = g.rows.map(r => ({...r, d: row(r.m, r.v)})).filter(r => r.d);
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
}
window.drawCharts = function(geo){
  if (CH) return drawAll(geo);
  fetch('data/charts.json').then(r => r.json()).then(j => { CH = j; drawAll(geo); }).catch(e => console.error('charts.json', e));
};
if (window.__site.geo()) window.drawCharts(window.__site.geo());
})();
