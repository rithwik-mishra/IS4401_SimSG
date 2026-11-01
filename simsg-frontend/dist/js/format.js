export const isNumber = value => typeof value === 'number' && Number.isFinite(value);
export const number = (value, digits = 0) => isNumber(value) ? value.toLocaleString('en-SG', {maximumFractionDigits: digits, minimumFractionDigits: digits}) : '—';
export const money = value => isNumber(value) ? `S$${number(value, 2)}` : '—';
export const percent = (value, digits = 0) => isNumber(value) ? `${number(value, digits)}%` : '—';
export const clamp = value => Math.max(0, Math.min(100, value));
export const list = value => Array.isArray(value) ? value : [];
export const text = value => typeof value === 'string' && value.trim() ? value : 'Not provided';

// Display-only glossary. Does not rewrite numbers, verdicts, or conclusions.
export function plainLanguage(value) {
  return text(value)
    .replace(/Van Westendorp(?: Price Sensitivity Meter)?/gi, 'price-reaction analysis')
    .replace(/Point of Marginal Cheapness/gi, 'lower end of the price range')
    .replace(/Point of Marginal Expensiveness/gi, 'upper end of the price range')
    .replace(/Optimal Price Point/gi, 'reference price')
    .replace(/Indifference Price Point/gi, 'price where cheap and expensive ratings are equal')
    .replace(/\s*\((?:OPP|PMC|PME|IPP|PSM)\)/g, '')
    .replace(/\bOPP\b/g, 'reference price')
    .replace(/\bPMC\b/g, 'lower price reference')
    .replace(/\bPME\b/g, 'upper price reference')
    .replace(/\bIPP\b/g, 'equal-rating price')
    .replace(/\bPSM\b/g, 'price reactions');
}

export function el(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) element.textContent = String(content);
  return element;
}

export function empty(container, message = 'No information returned for this section.') {
  container.replaceChildren(el('p', 'empty-state', message));
}

export const GROUPS = [
  {name: 'Senior / Retiree', short: 'Seniors / retirees', icon: '👓', colour: '#8c79b4'},
  {name: 'Working-Class', short: 'Working class', icon: '💼', colour: '#6b9a90'},
  {name: 'Suburban Family', short: 'Suburban families', icon: '🏡', colour: '#d6ad69'},
  {name: 'Young Adult / Student', short: 'Young adults', icon: '🎓', colour: '#8fa861'},
  {name: 'Affluent Private Property', short: 'Private property', icon: '🏙️', colour: '#7899bf'},
];

export function groupInfo(name) {
  return GROUPS.find(group => group.name === name) || {name: text(name), short: text(name), icon: '👤', colour: '#94a18d'};
}

export function dwellingLabel(value) {
  const labels = {'1-2 room': '1–2 room HDB', '3 room': '3 room HDB', '4 room': '4 room HDB', '5 room/exec': '5 room / executive HDB', condo: 'Condominium', landed: 'Landed home'};
  return labels[value] || text(value);
}

export function sentimentInfo(value) {
  const match = String(value || '').toLowerCase();
  if (match === 'positive') return {icon: '♥️', label: 'Positive'};
  if (match === 'negative') return {icon: '💔', label: 'Negative'};
  if (match === 'neutral') return {icon: '🤍', label: 'Neutral'};
  return {icon: '—', label: 'Not provided'};
}
