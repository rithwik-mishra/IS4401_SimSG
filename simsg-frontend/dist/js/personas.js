import {isNumber, el, empty, groupInfo, dwellingLabel, sentimentInfo, money, percent, text, plainLanguage, list} from './format.js';

const ageLabel = age => isNumber(age) ? `${age}yr` : '—';

export class PersonaPanel {
  constructor(root) {
    this.root = root;
    this.cards = root.querySelector('#persona-cards');
    this.detail = root.querySelector('#persona-detail');
    this.personas = [];
    this.active = -1;
    this.pinned = false;
    this.timer = null;
    this.root.addEventListener('pointerenter', () => clearTimeout(this.timer));
    this.root.addEventListener('pointerleave', () => {
      this.timer = setTimeout(() => {
        if (!this.pinned && !this.root.contains(document.activeElement)) this.close();
      }, 170);
    });
    this.root.addEventListener('focusout', event => {
      if (!this.root.contains(event.relatedTarget) && !this.pinned) this.close();
    });
    this.root.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        const card = this.cards.children[this.active];
        card?.focus();
        this.close();
      }
    });
  }

  render(personas) {
    this.close();
    this.personas = list(personas);
    this.cards.replaceChildren();
    if (!this.personas.length) return empty(this.cards, 'No individual profiles were returned.');
    this.personas.forEach((persona, index) => {
      const group = groupInfo(persona.demographic_cohort);
      const sentiment = sentimentInfo(persona.sentiment);
      const card = el('button', 'persona-card');
      card.type = 'button';
      card.style.setProperty('--group-color', group.colour);
      card.setAttribute('aria-expanded', 'false');
      card.setAttribute('aria-controls', 'persona-detail');
      card.setAttribute('aria-label', `${group.name}, ${persona.age ?? 'age not provided'}, ${text(persona.gender)}, ${dwellingLabel(persona.dwelling_type)}. ${sentiment.label}. Show profile details.`);
      const top = el('div', 'persona-top');
      const icon = el('span', 'persona-icon', group.icon);
      icon.setAttribute('aria-hidden', 'true');
      const feeling = el('span', 'persona-sentiment', sentiment.icon);
      feeling.setAttribute('role', 'img');
      feeling.setAttribute('aria-label', sentiment.label);
      top.append(icon, feeling);
      card.append(top, el('span', 'persona-name', group.name), el('span', 'persona-preview', `${ageLabel(persona.age)}, ${text(persona.gender)}`), el('span', 'persona-preview', dwellingLabel(persona.dwelling_type)));
      card.addEventListener('pointerenter', event => {
        clearTimeout(this.timer);
        if (event.pointerType !== 'touch' && !this.pinned) this.show(index);
      });
      card.addEventListener('focus', () => { if (!this.pinned) this.show(index); });
      card.addEventListener('click', () => {
        if (this.pinned && this.active === index) this.close();
        else { this.show(index); this.pinned = true; }
      });
      this.cards.append(card);
    });
  }

  show(index) {
    const persona = this.personas[index];
    if (!persona) return;
    this.active = index;
    [...this.cards.children].forEach((card, i) => card.setAttribute('aria-expanded', String(i === index)));
    this.detail.replaceChildren();
    const heading = el('div', 'detail-heading');
    const title = el('div');
    title.append(el('h3', '', text(persona.demographic_cohort)), el('p', '', `${ageLabel(persona.age)}, ${text(persona.gender)} · ${dwellingLabel(persona.dwelling_type)}`));
    const close = el('button', 'detail-close', '×');
    close.type = 'button'; close.setAttribute('aria-label', 'Close persona details');
    close.addEventListener('click', () => { this.cards.children[this.active]?.focus(); this.close(); });
    heading.append(title, close);
    const metrics = el('div', 'detail-metrics');
    const values = [
      ['Monthly household income', money(persona.monthly_income_sgd)],
      ['Daily dining budget', money(persona.daily_dining_budget_sgd)],
      ['Purchase intent', percent(persona.buy_probability_pct)],
      ['Price ceiling', money(persona.price_ceiling_sgd)],
    ];
    values.forEach(([label, value]) => { const cell = el('div', 'detail-metric'); cell.append(el('span', '', label), el('strong', '', value)); metrics.append(cell); });
    this.detail.append(heading, metrics, el('blockquote', 'persona-quote', `“${plainLanguage(persona.quote)}”`));
    if (persona.primary_objection) {
      const objection = el('p', 'detail-objection'); objection.append(el('strong', '', 'Main hesitation: '), document.createTextNode(plainLanguage(persona.primary_objection))); this.detail.append(objection);
    }
    const thresholds = el('div', 'thresholds');
    [['too_cheap', 'Too cheap'], ['cheap', 'Good value'], ['expensive', 'Getting expensive'], ['too_expensive', 'Too expensive']].forEach(([key,label]) => {
      if (persona[key] === null || persona[key] === undefined) return;
      const item = el('span'); item.append(document.createTextNode(`${label} `), el('b', '', money(persona[key]))); thresholds.append(item);
    });
    if (thresholds.children.length) this.detail.append(thresholds);
    this.detail.hidden = false;
  }

  close() {
    clearTimeout(this.timer);
    this.active = -1; this.pinned = false;
    [...this.cards.children].forEach(card => card.setAttribute('aria-expanded', 'false'));
    this.detail.hidden = true;
  }
}
