import fs from 'fs';
const src=fs.readFileSync('stuck.mjs','utf8').split('const sim=createPhysics')[0];
fs.writeFileSync('stuck_head.mjs', src);
const { ctx } = await import('./flow.mjs').catch(e=>{throw e});
