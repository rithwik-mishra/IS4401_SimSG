export interface PresetQuery {
  id: string;
  title: string;
  town: string;
  price: number;
  category: string;
  query: string;
}

export const PRESET_QUERIES: PresetQuery[] = [
  {
    id: 'amk-kopi',
    title: 'Artisanal Kopi Bar',
    town: 'Ang Mo Kio',
    price: 7.0,
    category: 'Beverage & Cafe',
    query: 'S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold brew and toast sets',
  },
  {
    id: 'tampines-bao',
    title: 'Salted Egg Bao Kiosk',
    town: 'Tampines',
    price: 4.5,
    category: 'Bakery / Snack',
    query: 'S$4.50 salted egg lava bao kiosk in Tampines Central with fresh organic soy milk sets',
  },
  {
    id: 'jurong-grain-bowl',
    title: 'High-Protein Grain Bowl',
    town: 'Jurong East',
    price: 13.5,
    category: 'Fast Casual Dining',
    query: 'S$13.50 high-protein keto grain bowl near Jurong East MRT with sous-vide chicken breast',
  },
  {
    id: 'bt-craft-beer',
    title: 'Craft Beer Microbrewery',
    town: 'Bukit Timah',
    price: 18.0,
    category: 'Bar & Alcohol',
    query: 'S$18 craft IPA microbrewery tasting room in Bukit Timah with artisanal sourdough pretzels',
  },
];

// Only the towns with local context passages in data/rag_corpus/layer2_corpus.json
export const PLANNING_AREAS = [
  { value: 'all', label: 'Use location in idea' },
  { value: 'Ang Mo Kio', label: 'Ang Mo Kio' },
  { value: 'Bukit Merah', label: 'Bukit Merah' },
  { value: 'Bukit Timah', label: 'Bukit Timah' },
  { value: 'Jurong East', label: 'Jurong East' },
  { value: 'Outram', label: 'Outram' },
  { value: 'Tampines', label: 'Tampines' },
];
