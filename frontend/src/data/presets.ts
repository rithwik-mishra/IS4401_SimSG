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

export const PLANNING_AREAS = [
  { value: 'all', label: 'Use location in idea' },
  { value: 'Ang Mo Kio', label: 'Ang Mo Kio' },
  { value: 'Tampines', label: 'Tampines' },
  { value: 'Jurong East', label: 'Jurong East' },
  { value: 'Bukit Timah', label: 'Bukit Timah' },
  { value: 'Bukit Merah', label: 'Bukit Merah' },
  { value: 'Outram', label: 'Outram' },
  { value: 'Bedok', label: 'Bedok' },
  { value: 'Bishan', label: 'Bishan' },
  { value: 'Bukit Batok', label: 'Bukit Batok' },
  { value: 'Bukit Panjang', label: 'Bukit Panjang' },
  { value: 'Choa Chu Kang', label: 'Choa Chu Kang' },
  { value: 'Clementi', label: 'Clementi' },
  { value: 'Geylang', label: 'Geylang' },
  { value: 'Hougang', label: 'Hougang' },
  { value: 'Kallang', label: 'Kallang' },
  { value: 'Marine Parade', label: 'Marine Parade' },
  { value: 'Novena', label: 'Novena' },
  { value: 'Pasir Ris', label: 'Pasir Ris' },
  { value: 'Punggol', label: 'Punggol' },
  { value: 'Queenstown', label: 'Queenstown' },
  { value: 'Sengkang', label: 'Sengkang' },
  { value: 'Serangoon', label: 'Serangoon' },
  { value: 'Toa Payoh', label: 'Toa Payoh' },
  { value: 'Woodlands', label: 'Woodlands' },
  { value: 'Yishun', label: 'Yishun' },
];
