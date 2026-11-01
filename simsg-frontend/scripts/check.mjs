import {readdir,readFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {validateResponse,buildRequest} from '../dist/js/api.js';
import {plainLanguage} from '../dist/js/format.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const files=(await readdir(path.join(root,'dist/js'))).filter(name=>name.endsWith('.js'));
for(const file of [...files.map(name=>'dist/js/'+name),'dist/config.js']){
  const checked=spawnSync(process.execPath,['--check',path.join(root,file)],{encoding:'utf8'});
  assert.equal(checked.status,0,checked.stderr);
}
const fixture=JSON.parse(await readFile(path.join(root,'dist/sample-output.json'),'utf8'));
assert.equal(validateResponse(fixture),fixture);
assert.throws(()=>validateResponse({personas:[]}),/missing/);
assert.equal(plainLanguage('Optimal Price Point (OPP): S$4.50'),'reference price: S$4.50');

// Request-contract checks: omitted optional values, explicit zero, and the AI/mock inversion.
const NativeFormData=globalThis.FormData;
globalThis.FormData=class {constructor(values){this.values=values;}get(key){return this.values[key]??null;}has(key){return Object.hasOwn(this.values,key);}};
try{
  assert.deepEqual(buildRequest({raw_query:'  Coffee in Tampines  ',sample_size:'30',use_ai:'on',override_price:'',override_town:'',seed:''}),{raw_query:'Coffee in Tampines',sample_size:30,use_mock:false});
  assert.deepEqual(buildRequest({raw_query:'Coffee',sample_size:'5',override_price:'0',override_town:'Outram',seed:'0'}),{raw_query:'Coffee',sample_size:5,use_mock:true,override_price:0,override_town:'Outram',seed:0});
  assert.throws(()=>buildRequest({raw_query:'Coffee',sample_size:'101'}),/5 and 100/);
  assert.throws(()=>buildRequest({raw_query:' ',sample_size:'30'}),/Describe/);
}finally{globalThis.FormData=NativeFormData;}
console.log(`Passed: ${files.length+1} JavaScript syntax checks, fixture shape, plain-language mapping, and API request contracts.`);
