const sharp = require('C:/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const path = require('path');
(async () => {
  for (const name of ['legacy_style', 'pooled_style']) {
    await sharp(path.join(__dirname, 'references', name + '.svg'), {density: 144})
      .resize({width: name === 'legacy_style' ? 700 : 1600})
      .png().toFile(path.join(__dirname, 'references', name + '_preview.png'));
  }
})();
