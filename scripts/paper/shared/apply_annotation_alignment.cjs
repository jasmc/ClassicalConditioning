// Explicit candidate edit entry point. Verification uses the matching verify command.
if(process.argv.includes('--help')){console.log('Usage: node '+require('path').basename(__filename)+' candidate-directory browser-executable');process.exit(0);}
if(process.argv.includes('--verify')||process.argv.includes('--center-labels'))throw Error('Use the verification or centering entry point separately');
process.argv.splice(4,0,'--apply');
require('./verify_annotations.cjs');
