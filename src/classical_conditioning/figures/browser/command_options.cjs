// Pure command configuration; loading this module never launches a browser.
function parseMode(args, {allowCenter=false}={}) {
  const options=new Set(args), known=new Set(['--verify','--apply',...(allowCenter?['--center-labels']:[])]);
  for(const option of options)if(!known.has(option))throw Error('Unknown presentation option: '+option);
  if(options.has('--verify')&&(options.has('--apply')||options.has('--center-labels')))throw Error('Verification cannot be combined with artwork edits');
  if(options.has('--apply')&&options.has('--center-labels'))throw Error('Choose styling or centering separately');
  const centerOnly=options.has('--center-labels');
  return {verify:!options.has('--apply')||centerOnly,apply:options.has('--apply'),centerOnly};
}
function baseURL(args, fallback='http://127.0.0.1:8766') {
  const i=args.indexOf('--base-url');
  const value=i<0?fallback:args[i+1];
  if(!value||!/^https?:\/\//.test(value))throw Error('Provide an absolute --base-url');
  return value.replace(/\/$/,'');
}
module.exports={parseMode,baseURL};
