import {config} from '../config.js';
import {buildRequest, runSimulation} from './api.js';
import {el, empty, plainLanguage, percent, number, list, isNumber, clamp} from './format.js';
import {PersonaPanel} from './personas.js';
import {PriceChart, renderComposition, renderBuying, renderSentiment} from './charts.js';

const $ = id => document.getElementById(id);
const form = $('simulation-form');
const results = $('results');
const personaPanel = new PersonaPanel($('persona-area'));
const priceChart = new PriceChart($('price-chart'), $('price-legend'), $('price-points'));
const demo = config.mode === 'demo';
let busy = false;
let hasResults = false;

$('mode-tag').textContent = demo ? '' : 'Live simulation';
$('mode-tag').hidden = demo;
$('form-note').textContent = demo ? '' : 'AI responses use your connected backend. Charges may apply.';
$('form-note').hidden = demo;
$('advanced-toggle').addEventListener('click', () => {
  const open = $('advanced-options').hidden;
  $('advanced-options').hidden = !open;
  $('advanced-toggle').setAttribute('aria-expanded', String(open));
  $('advanced-toggle').firstChild.textContent = open ? 'Fewer options ' : 'More options ';
});

function renderRecommendations(recommendations) {
  const variants = {
    VIABLE: {title: 'Looks viable', state: 'positive', symbol: '✓'},
    'MODERATE RISK': {title: 'Some risk to consider', state: 'moderate', symbol: '!'},
    'HIGH RISK - PIVOT REQUIRED': {title: 'High risk. Time to adjust.', state: 'negative', symbol: '↗'},
  };
  const verdict = variants[recommendations.verdict_badge] || {title: recommendations.verdict_badge || 'No verdict returned', state:'unknown', symbol:'—'};
  const container = $('verdict-content');
  container.parentElement.dataset.state = verdict.state;
  container.replaceChildren(el('div','verdict-symbol',verdict.symbol), el('h2','verdict-title',verdict.title), el('p','verdict-caption','Based on the returned simulation, not a forecast of real-world sales.'));
  $('executive-summary').textContent = plainLanguage(recommendations.executive_summary);
  $('recommended-pivot').textContent = plainLanguage(recommendations.recommended_pivot);
  const uplift = $('post-adjustment');uplift.replaceChildren();
  if (isNumber(recommendations.simulated_post_pivot_uplift)) uplift.append(el('strong','',percent(recommendations.simulated_post_pivot_uplift)),el('span','','estimated purchase intent after adjustments'));
  const themes = list(recommendations.clustered_objections);
  const concerns = $('concerns-list');concerns.replaceChildren();
  if (!themes.length) empty(concerns,'No concern themes were returned.');
  themes.forEach(theme => {
    const row=el('article','concern-row');const overview=el('div');
    overview.append(el('h3','theme-label',plainLanguage(theme.theme)));
    const track=el('div','bar-track');const fill=el('span','bar-fill');fill.style.width=`${isNumber(theme.frequency_pct)?clamp(theme.frequency_pct):0}%`;track.append(fill);
    const numbers=el('div','concern-numbers');numbers.append(el('span','','Estimated share'),el('strong','',percent(theme.frequency_pct)));
    overview.append(track,numbers);
    const quotes=el('ul','concern-quotes');list(theme.representative_quotes).forEach(quote=>quotes.append(el('li','',plainLanguage(quote))));
    if(!quotes.children.length) quotes.append(el('li','','No example quotes were returned.'));
    row.append(overview,quotes);concerns.append(row);
  });
}

function renderResult(data, request) {
  const stats=data.statistics;
  renderComposition($('composition-chart'),stats.cohort_breakdown);
  const counts = Object.values(stats.sentiment_breakdown || {});
  const returnedCount = counts.length && counts.every(value => Number.isInteger(value) && value >= 0)
    ? counts.reduce((sum, value) => sum + value, 0) : null;
  const reportCount = returnedCount ?? (demo ? null : request.sample_size);
  $('respondent-count').textContent = number(reportCount);
  const countNote = $('respondent-note');
  countNote.textContent = reportCount !== null && reportCount !== request.sample_size
    ? (demo ? `Saved example: ${reportCount} responses. Your selected ${request.sample_size} respondents require a connected API to generate a new report.`
      : `You requested ${request.sample_size} respondents; the service returned ${reportCount} responses.`)
    : returnedCount === null && !demo ? 'Requested sample size; the service did not return a response total.' : '';
  countNote.hidden = demo || !countNote.textContent;
  personaPanel.render(data.personas);
  renderRecommendations(data.recommendations);
  renderBuying($('buying-chart'),stats.cohort_breakdown);
  renderSentiment($('sentiment-chart'),stats.sentiment_breakdown);
  $('mean-intent').textContent=percent(stats.mean_buy_intent_pct);
  $('sentiment-score').textContent=isNumber(stats.net_sentiment_score)?`${stats.net_sentiment_score>0?'+':''}${number(stats.net_sentiment_score,2)}`:'—';
  $('result-context').textContent=demo
    ? ''
    : `RESULTS FOR · ${request.raw_query}`;
  $('result-context').hidden = demo;
  results.hidden=false;
  priceChart.render(stats.psm_curves,stats.psm_intersections,demo?7:request.override_price);
  hasResults=true;
}

async function submit(request) {
  if(busy) return;
  busy=true;
  $('submit-button').disabled=true;
  $('submit-label').textContent=demo?'Loading…':'Simulating…';
  results.setAttribute('aria-busy','true');
  $('request-error').hidden=true;
  $('request-status').textContent=demo?'':'Waiting for your simulation. This can take a little while.';
  try {
    const data=await runSimulation(request,config);
    renderResult(data,request);
    $('request-status').textContent=demo
      ? ''
      : 'Your simulation is ready.';
  } catch(error) {
    $('request-error').textContent=error.message;
    $('request-error').hidden=false;
    $('request-status').textContent=hasResults?'The earlier report remains below. No new result was received.':'';
  } finally {
    busy=false;
    $('submit-button').disabled=false;
    $('submit-label').textContent='Simulate';
    results.setAttribute('aria-busy','false');
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  if(!form.reportValidity() || busy) return;
  try { submit(buildRequest(form)); }
  catch(error) {$('request-error').textContent=error.message;$('request-error').hidden=false;}
});

// Only the saved fixture is loaded automatically; API mode never runs on page load.
if(demo) submit(buildRequest(form));
