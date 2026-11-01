import {el, empty, groupInfo, money, percent, number, isNumber, list, clamp} from './format.js';

const NS = 'http://www.w3.org/2000/svg';
function svgElement(tag, attrs = {}, content) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key, String(value)));
  if (content !== undefined) node.textContent = content;
  return node;
}

export function renderComposition(container, rows) {
  const groups = list(rows).slice().sort((a, b) => (isNumber(b.population_share_pct) ? b.population_share_pct : -Infinity) - (isNumber(a.population_share_pct) ? a.population_share_pct : -Infinity));
  if (!groups.length) return empty(container);
  const wrapper = el('div', 'composition-rows');
  const maximum = Math.max(1, ...groups.map(group => isNumber(group.population_share_pct) ? group.population_share_pct : 0));
  groups.forEach(group => {
    const meta = groupInfo(group.cohort_name);
    const row = el('div', 'composition-row'); row.style.setProperty('--group-color', meta.colour);
    const name = el('span', 'composition-name'); name.append(el('i', 'colour-dot'), document.createTextNode(meta.short)); name.title = meta.name;
    const track = el('div', 'bar-track'); const bar = el('span', 'bar-fill');
    bar.style.width = `${isNumber(group.population_share_pct) ? Math.max(0, group.population_share_pct) / maximum * 100 : 0}%`; track.append(bar);
    row.append(name, track, el('span', 'count-label', percent(group.population_share_pct)));
    row.title = `${meta.name}: ${percent(group.population_share_pct)} of the panel`;
    wrapper.append(row);
  });
  container.replaceChildren(wrapper);
}

export function renderBuying(container, rows) {
  const groups = list(rows).slice().sort((a, b) => (isNumber(b.mean_buy_intent_pct) ? b.mean_buy_intent_pct : -Infinity) - (isNumber(a.mean_buy_intent_pct) ? a.mean_buy_intent_pct : -Infinity));
  if (!groups.length) return empty(container);
  const wrapper = el('div', 'buying-rows');
  groups.forEach(group => {
    const meta = groupInfo(group.cohort_name);
    const row = el('div', 'buying-row'); row.style.setProperty('--group-color', meta.colour);
    const label = el('div', 'buying-label'); const text = el('div');
    text.append(document.createTextNode(meta.short), el('small', '', `${number(group.count)} in this group`)); label.append(el('i','colour-dot'), text);
    const track = el('div', 'bar-track'); const fill = el('span','bar-fill');
    fill.style.width = `${isNumber(group.mean_buy_intent_pct) ? clamp(group.mean_buy_intent_pct) : 0}%`;track.append(fill);
    row.append(label,track,el('span','buying-value',percent(group.mean_buy_intent_pct,1))); wrapper.append(row);
  });
  const scale = el('div','buying-scale'); [0,25,50,75,100].forEach(value=>scale.append(el('span','',`${value}%`)));
  const details = el('details','group-ceilings'); details.append(el('summary','','View price ceilings and group shares'));
  const grid = el('div','group-ceilings-grid');
  groups.forEach(group => {const item=el('div'); item.append(el('p','',groupInfo(group.cohort_name).short), el('strong','',money(group.median_price_ceiling_sgd)),el('p','',`${percent(group.population_share_pct,2)} of the panel`));grid.append(item);});
  details.append(grid); container.replaceChildren(wrapper,scale,details);
}

export function renderSentiment(container, counts) {
  const entries = Object.entries(counts || {}).filter(([,value]) => isNumber(value) && value >= 0);
  if (!entries.length) return empty(container);
  const total = entries.reduce((sum,[,value]) => sum+value,0);
  const colours = {Positive:'#6e9b7e',Neutral:'#c6ae77',Negative:'#c98478'};
  const track = el('div','sentiment-track'); track.setAttribute('role','img');track.setAttribute('aria-label',entries.map(([name,count])=>`${name}: ${count}`).join('; '));
  const legend = el('div','sentiment-legend');
  entries.forEach(([name,count]) => {
    const share = total > 0 ? count/total*100 : 0;
    const colour = colours[name] || '#a4b29c';
    if (count > 0) { const segment = el('div','sentiment-segment',share>14?`${name} ${number(share)}%`:'');segment.style.width=`${share}%`;segment.style.setProperty('--sentiment-color',colour);segment.title=`${name}: ${count} (${number(share,1)}%)`;track.append(segment); }
    const item=el('div','sentiment-item');item.style.setProperty('--sentiment-color',colour);item.append(el('i'),el('span','',name),el('strong','',`${count} · ${number(share)}%`));legend.append(item);
  });
  container.replaceChildren(track,legend);
}

const SERIES=[
  {key:'too_cheap',label:'So cheap, quality feels doubtful',colour:'#91a8c4',dash:''},
  {key:'cheap',label:'Good value',colour:'#91a46f',dash:''},
  {key:'expensive',label:'Getting expensive',colour:'#d7b570',dash:'6 3'},
  {key:'too_expensive',label:'Too expensive to consider',colour:'#c68f83',dash:'2 4'},
];
const PRICE_POINTS=[
  {key:'pmc',label:'Lower end of the range',short:'Lower end',note:'Returned lower price reference'},
  {key:'opp',label:'Reference price',short:'Reference',note:'A starting point for price exploration'},
  {key:'ipp',label:'Cheap and expensive ratings are equal',short:'Equal ratings',note:'The two groups have the same share'},
  {key:'pme',label:'Upper end of the range',short:'Upper end',note:'Returned upper price reference'},
];

export class PriceChart {
  constructor(container, legend, points) {
    this.container=container;this.legend=legend;this.points=points;
    this.rows=[];this.values={};this.visible=new Set(SERIES.map(s=>s.key));this.inputPrice=null;
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(container);
    SERIES.forEach(series=>{
      const button=el('button','legend-button');button.type='button';button.setAttribute('aria-pressed','true');button.style.setProperty('--series-color',series.colour);
      button.append(el('i','legend-line'),document.createTextNode(series.label));
      button.addEventListener('click',()=>{this.visible.has(series.key)?this.visible.delete(series.key):this.visible.add(series.key);button.setAttribute('aria-pressed',String(this.visible.has(series.key)));this.draw();});legend.append(button);
    });
  }

  render(rows,values,inputPrice=null) {
    this.rows=list(rows).filter(row=>isNumber(row.price)).slice().sort((a,b)=>a.price-b.price);
    this.values=values || {};this.inputPrice=isNumber(inputPrice)?inputPrice:null;
    this.points.replaceChildren();
    PRICE_POINTS.forEach(point=>{const item=el('div','price-point');item.append(el('span','price-point-label',point.label),el('strong','',money(this.values[point.key])),el('small','',point.note));this.points.append(item);});
    this.draw();
  }

  draw() {
    const width=this.container.clientWidth;
    if (!width) return;
    if (!this.rows.length) return empty(this.container,'No price reactions were returned.');
    const height=330,m={left:47,right:18,top:36,bottom:48};
    const refs=Object.values(this.values).filter(isNumber);
    const prices=[...this.rows.map(row=>row.price),...refs,...(this.inputPrice===null?[]:[this.inputPrice])];
    let low=Math.min(...prices),high=Math.max(...prices);if(low===high){low=Math.max(0,low-1);high+=1;}
    const span=high-low;low=Math.max(0,low-span*.02);high+=span*.02;
    const x=value=>m.left+(value-low)/(high-low)*(width-m.left-m.right);
    const y=value=>height-m.bottom-clamp(value)/100*(height-m.top-m.bottom);
    const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,height,role:'img','aria-label':'Simulated price reactions, with returned price reference values. See the values beneath this chart.'});
    const add=(tag,attrs,content)=>{const node=svgElement(tag,attrs,content);svg.append(node);return node;};
    if(isNumber(this.values.pmc)&&isNumber(this.values.pme)&&this.values.pme>=this.values.pmc){add('rect',{x:x(this.values.pmc),y:m.top,width:x(this.values.pme)-x(this.values.pmc),height:height-m.top-m.bottom,fill:'#eef3e6',opacity:.85});}
    [0,25,50,75,100].forEach(value=>{add('line',{x1:m.left,x2:width-m.right,y1:y(value),y2:y(value),stroke:'#e9ede3','stroke-width':1});add('text',{x:m.left-12,y:y(value)+4,'text-anchor':'end'},`${value}%`);});
    const steps=width<500?3:6;
    for(let i=0;i<=steps;i++){const value=low+(high-low)*i/steps;add('text',{x:x(value),y:height-m.bottom+22,'text-anchor':i===0?'start':i===steps?'end':'middle'},`S$${number(value,1)}`);}
    add('text',{x:(m.left+width-m.right)/2,y:height-4,'text-anchor':'middle'},'Price per item');
    add('text',{x:m.left,y:16},'Share of simulated responses');
    SERIES.filter(series=>this.visible.has(series.key)).forEach(series=>{
      let path='',started=false;
      for(const row of this.rows){if(!isNumber(row[series.key])){started=false;continue;}path+=`${started?'L':'M'}${x(row.price)},${y(row[series.key])} `;started=true;}
      add('path',{d:path.trim(),fill:'none',stroke:series.colour,'stroke-width':2.5,'stroke-linejoin':'round','stroke-dasharray':series.dash});
    });
    if(this.inputPrice!==null){add('line',{x1:x(this.inputPrice),x2:x(this.inputPrice),y1:m.top,y2:height-m.bottom,stroke:'#7c8971','stroke-width':1,'stroke-dasharray':'5 5'});add('text',{x:Math.min(width-m.right,Math.max(m.left+70,x(this.inputPrice))),y:m.top-7,'text-anchor':'end'},`Your price ${money(this.inputPrice)}`);}
    // Values come directly from the response. No intersections are calculated here.
    [this.values.pmc,this.values.pme].filter(isNumber).forEach(value=>add('line',{x1:x(value),x2:x(value),y1:m.top,y2:height-m.bottom,stroke:'#bccca6','stroke-width':1}));
    const hover=svgElement('g',{visibility:'hidden'});svg.append(hover);
    const guide=svgElement('line',{y1:m.top,y2:height-m.bottom,stroke:'#91a281','stroke-width':1});hover.append(guide);
    const dots=SERIES.filter(series=>this.visible.has(series.key)).map(series=>{const dot=svgElement('circle',{r:4,fill:series.colour,stroke:'#fff','stroke-width':1.5});hover.append(dot);return {series,dot};});
    const tooltip=el('div','chart-tooltip');tooltip.hidden=true;tooltip.setAttribute('role','tooltip');
    let pinned=false;
    const show=event=>{
      const rect=svg.getBoundingClientRect();const px=Math.max(x(this.rows[0].price),Math.min(x(this.rows.at(-1).price),event.clientX-rect.left));
      const price=low+(px-m.left)/(width-m.left-m.right)*(high-low);
      const index=this.rows.findIndex(row=>row.price>=price);const b=this.rows[index<0?this.rows.length-1:index];const a=this.rows[Math.max(0,(index<0?this.rows.length-1:index)-1)];
      const fraction=b.price===a.price?0:(price-a.price)/(b.price-a.price);
      hover.setAttribute('visibility','visible');guide.setAttribute('x1',px);guide.setAttribute('x2',px);tooltip.replaceChildren(el('strong','',money(price)));
      dots.forEach(({series,dot})=>{if(!isNumber(a[series.key])||!isNumber(b[series.key])){dot.setAttribute('visibility','hidden');return;}const value=a[series.key]+fraction*(b[series.key]-a[series.key]);dot.setAttribute('visibility','visible');dot.setAttribute('cx',px);dot.setAttribute('cy',y(value));const row=el('div');row.append(el('span','',series.label),el('span','',percent(value,1)));tooltip.append(row);});
      tooltip.hidden=false;tooltip.style.left=`${Math.max(0,Math.min(width-tooltip.offsetWidth,px+15))}px`;tooltip.style.top='46px';
    };
    const hit=add('rect',{x:m.left,y:m.top,width:width-m.left-m.right,height:height-m.top-m.bottom,fill:'transparent'});
    hit.addEventListener('pointermove',event=>{if(!pinned)show(event);});hit.addEventListener('pointerleave',()=>{if(!pinned){hover.setAttribute('visibility','hidden');tooltip.hidden=true;}});hit.addEventListener('click',event=>{pinned=!pinned;show(event);});
    this.container.replaceChildren(svg,tooltip);
  }
}
