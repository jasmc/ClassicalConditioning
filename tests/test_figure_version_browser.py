import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.reviews import library
from scripts.reviews.versions import Catalogue, SavedHTML, natural, scope_key, panel_views, sha, packed, figure_page
from scripts.reviews.ordered_library import ordered_groups, verify_package

SVG=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10"><path d="M0 0L20 10"/></svg>'
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')

class FigureVersions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);self.repo=base/'repo';self.outputs=base/'outputs';self.archive=base/'archive';self.stage=base/'stage'
        for p in [self.repo,self.outputs,self.archive,self.stage]:p.mkdir()
        self.scope=self.outputs/'figures/figure-01/assembly/versions/example';self.scope.mkdir(parents=True)
    def discover(self):return Catalogue(self.outputs,self.archive,self.repo,self.stage).discover()
    def catalogue(self):
        groups=[]
        for n in [4,2,1,3]:
            p=self.outputs/f'figures/figure-{n:02}/INDEX.html';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('Old navigation '+str(n))
            groups.append(dict(id=f'figure-{n:02}-folders',title=f'Figure {n}',context='',entries=[dict(id=f'fig{n}',title=f'Open Figure {n}',storage='outputs',path=str(p.relative_to(self.outputs)),sha256=sha(p.read_bytes()),kind='generated_navigation',description='Folders',status='Navigation')]))
        return dict(schema_version=1,version_browser=1,purpose='Figures and versions',reviewed='2026-10-27',groups=groups)
    def test_natural_and_scope_order(self):
        self.assertEqual(sorted(['v10','v2','v1'],key=natural),['v1','v2','v10'])
        scopes=['panels-c-f-i','panel-i','panel-a','assembly','panels-a-b-c']
        self.assertEqual(sorted(scopes,key=scope_key),['assembly','panel-a','panel-i','panels-a-b-c','panels-c-f-i'])
        self.assertEqual(sorted([12,'v2','02'],key=natural),['02',12,'v2'])
    def test_html_parser_handles_nested_svg_and_ignores_script_markup(self):
        d=SavedHTML('<h2>V2</h2><script id="fake">const x="<img src=\'bad\'>";</script><svg><svg><path/></svg></svg><figure><img src="x.png"><figcaption>Exact caption</figcaption></figure>')
        self.assertEqual(len(d.visuals),2);self.assertEqual(d.visuals[0]['raw'],b'<svg><svg><path/></svg></svg>')
        self.assertEqual(d.visuals[1]['caption'],'Exact caption')
    def test_same_picture_with_distinct_source_context_is_not_merged(self):
        for name in ['v2.svg','v10.svg']:(self.scope/name).write_bytes(SVG)
        (self.scope/'legend.md').write_text('Saved bundle legend')
        d=self.discover();rows=[r for r in d['versions'] if r['role']=='version']
        self.assertEqual(len(rows),2);self.assertNotEqual(rows[0]['id'],rows[1]['id'])
        self.assertEqual(rows[0]['artifact_sha256'],rows[1]['artifact_sha256']);self.assertEqual(rows[0]['legend']['binding'],'bundle legend')
    def test_html_data_images_zip_and_embedded_archive_are_accounted(self):
        data=base64.b64encode(PNG).decode();archive={'files':{'frozen.svg':{'base64':base64.b64encode(SVG).decode(),'sha256':sha(SVG)}}}
        (self.scope/'review.html').write_text('<h2>V2</h2><figure><img src="data:image/png;base64,'+data+'"><figcaption>Legend V2</figcaption></figure><script id="archive">'+json.dumps(archive)+'</script>')
        with zipfile.ZipFile(self.scope/'saved.zip','w') as z:z.writestr('v1.svg',SVG)
        d=self.discover();self.assertEqual(len(d['versions']),3)
        self.assertTrue(any(r['legend']['text']=='Legend V2' for r in d['versions']))
        self.assertTrue(any(r.get('role')=='archive artwork' for r in d['inventory']))
    def test_embedded_legend_and_explicit_image_context_are_retained(self):
        (self.scope/'figure.svg').write_bytes(SVG)
        (self.scope/'figure.html').write_text('<article><h2>Version one</h2><img src="figure.svg"><div class="legend"><p>Exact saved associated legend with source definitions.</p></div></article>')
        d=self.discover();rows=[r for r in d['versions'] if r['source']['path'].endswith('figure.svg')]
        self.assertEqual(rows[0]['legend']['text'],'Exact saved associated legend with source definitions.')
        self.assertIn('context_source',rows[0]['legend'])

    def test_legend_edits_are_detected_by_external_check(self):
        (self.scope/'v2.svg').write_bytes(SVG);legend=self.scope/'legend.md';legend.write_text('Saved legend')
        cat=self.catalogue();library.build(cat,self.outputs,self.archive,self.repo);legend.write_text('Changed legend')
        self.assertTrue(any(str(legend) in issue for issue in library.check(cat,self.repo,self.outputs,self.archive,external=True)['issues']))

    def test_native_svg_companion_uses_exact_embedded_svg_and_saved_legend(self):
        (self.scope/'figure-01-v11-editing.svg').write_bytes(b'<?xml version="1.0"?>'+SVG)
        (self.scope/'figure-01-v11-editing.html').write_text(SVG.decode()+'<div class="legend">Saved native SVG companion legend.</div>')
        d=self.discover();native=next(r for r in d['versions'] if r['source']['path'].endswith('figure-01-v11-editing.svg'))
        self.assertEqual(native['family'],'Presentation drafts');self.assertEqual(native['legend']['text'],'Saved native SVG companion legend.')

    def test_combined_historical_page_assigns_each_explicit_figure_heading(self):
        (self.scope/'combined.html').write_text('<h2>Figure 1 candidate</h2>'+SVG.decode()+'<h2>Figure 2 original</h2>'+SVG.decode())
        data=self.discover();self.assertEqual([r['figures'] for r in data['versions']],[[1],[2]])
        self.assertEqual(len({r['id'] for r in data['versions']}),2)

    def test_selected_preview_does_not_follow_subsequent_working_edits(self):
        art=self.scope/'working.svg';art.write_bytes(SVG);legend=self.scope/'legend.md';legend.write_text('Selected legend')
        state=self.repo/'docs/paper/figures/figure-01/current-state.json';state.parent.mkdir(parents=True)
        state.write_text(json.dumps(dict(figure_id='fig1',default_display='draft',draft=dict(working_svg=dict(path=str(art),sha256=sha(SVG)),legend=dict(path=str(legend),sha256=sha(legend.read_bytes()))))))
        d=self.discover();e=next(v for v in d['versions'] if v['selected'])
        self.assertTrue(e['snapshot_preview']);self.assertTrue(e['preview'].startswith('version-assets/'))
        art.write_bytes(b'manual working change');self.assertEqual((self.stage/e['preview']).read_bytes(),SVG)

    def test_corruption_of_prepared_preview_prevents_publication(self):
        from scripts.reviews import ordered_library
        (self.scope/'image.html').write_text('<img src="data:image/png;base64,'+base64.b64encode(PNG).decode()+'">')
        cat=self.catalogue();publish=ordered_library.publish_controls
        def corrupt(plan):
            payload=next((self.outputs/'review-library/version-assets').glob('*'));payload.write_bytes(b'concurrent edit')
            return publish(plan)
        with patch.object(ordered_library,'publish_controls',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'advanced'):library.build(cat,self.outputs,self.archive,self.repo)
        self.assertFalse((self.outputs/'START-HERE.html').exists())

    def test_new_retained_version_requires_and_survives_catalogue_refresh(self):
        (self.scope/'v1.svg').write_bytes(SVG);cat=self.catalogue();library.build(cat,self.outputs,self.archive,self.repo)
        new=self.scope/'v2.svg';new.write_bytes(SVG)
        self.assertTrue(any('Uninventoried' in issue and str(new) in issue for issue in library.check(cat,self.repo,self.outputs,self.archive,external=True)['issues']))
        library.build(cat,self.outputs,self.archive,self.repo)
        self.assertEqual(library.check(cat,self.repo,self.outputs,self.archive,external=True)['issues'],[])

    def test_corrupt_embedded_payload_fails(self):
        (self.scope/'bad.html').write_text('<script id="archive">'+json.dumps({'files':{'x.svg':{'base64':base64.b64encode(SVG).decode(),'sha256':'0'*64}}})+'</script>')
        with self.assertRaisesRegex(ValueError,'Corrupt embedded'):self.discover()
    def test_missing_legend_and_unavailable_image_are_visible_exceptions(self):
        (self.scope/'x.svg').write_bytes(SVG);(self.scope/'review.html').write_text('<img src="missing.png">')
        d=self.discover();self.assertEqual({r['reason'] for r in d['issues']},{'No saved legend association','Missing local image'})
        self.assertNotEqual(d['coverage']['status'],'complete')
    def test_explicit_panel_groups_are_isolated_without_other_panel_artwork(self):
        spec=dict(final_size_mm=[20,10],allocations=[dict(id='A',kind='panel',box_mm=[0,0,10,10],group_ids=['owned-A']),dict(id='B',kind='panel',box_mm=[10,0,10,10],group_ids=['owned-B'])])
        raw=('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10"><metadata id="layout-frames">'+json.dumps(spec)+'</metadata><defs><clipPath id="clip"><rect width="10" height="10"/></clipPath></defs><g transform="translate(1 0)"><g id="owned-A"><path id="a" d="M0 0L1 1"/></g><g id="owned-B"><path id="b"/></g></g></svg>').encode()
        views=panel_views(raw);self.assertEqual(len(views),2);self.assertIn(b'owned-A',views[0]['raw']);self.assertNotIn(b'owned-B',views[0]['raw']);self.assertIn(b'translate(1 0)',views[0]['raw']);self.assertIn(b'clipPath',views[0]['raw'])
        self.assertEqual(panel_views(SVG),[])
    def test_one_figure_section_and_panel_sort(self):
        cat=self.catalogue();cat['groups'].append(dict(id='figure-2-g',title='G',context='',entries=[dict(id='g',title='G guide',storage='outputs',path='figures/figure-02/panel-g/study.md',status='Study',description='Guide')]))
        groups=ordered_groups(cat);self.assertEqual([g['figure'] for g in groups],[1,2,3,4]);self.assertEqual(len(groups[1]['entries']),2)
    def test_guarded_build_preserves_art_and_uses_one_browser_per_figure(self):
        (self.scope/'v2.svg').write_bytes(SVG);(self.scope/'legend.md').write_text('Saved legend')
        cat=self.catalogue();report=library.build(cat,self.outputs,self.archive,self.repo)
        home=(self.outputs/'START-HERE.html').read_text();self.assertEqual([home.index('<h2>Figure '+str(n)+'</h2>') for n in [1,2,3,4]],sorted(home.index('<h2>Figure '+str(n)+'</h2>') for n in [1,2,3,4]))
        self.assertEqual((self.scope/'v2.svg').read_bytes(),SVG)
        page=(self.outputs/'figures/figure-01/INDEX.html').read_text();self.assertIn('version-catalogue',page);self.assertIn('art.replaceChildren()',page);self.assertNotIn('localStorage',page)
        self.assertEqual(library.check(cat,self.repo,self.outputs,self.archive,external=True)['issues'],[])
        (self.scope/'v2.svg').write_bytes(SVG.replace(b'20 10',b'30 10'))
        with self.assertRaisesRegex(ValueError,'Historical browser source changed'):library.build(cat,self.outputs,self.archive,self.repo)
    def test_manual_navigation_and_payload_edits_are_preserved(self):
        (self.scope/'a.html').write_text('<img src="data:image/png;base64,'+base64.b64encode(PNG).decode()+'">')
        cat=self.catalogue();report=library.build(cat,self.outputs,self.archive,self.repo)
        payload=self.outputs/report['payloads'][0]['path'];payload.write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError,'payload changed'):library.build(cat,self.outputs,self.archive,self.repo)
        self.assertEqual(payload.read_bytes(),b'edited')
    def test_direct_file_urls_and_joint_scope_links(self):
        p=self.outputs/'figures/figure-02/panels-b-e-h/versions/v2.svg';p.parent.mkdir(parents=True);p.write_bytes(SVG)
        data=self.discover();page=figure_page(2,data,self.outputs).decode();d=json.loads(page.split('id="version-catalogue">',1)[1].split('</script>',1)[0])
        self.assertTrue(any(s['id']=='panel-h' and s['related']==['panels-b-e-h'] for s in d['scopes']))
        self.assertFalse(d['versions'][0]['url'].startswith('/'));self.assertIn('panels-b-e-h',d['versions'][0]['url'])
    def test_unavailable_storage_and_source_symlink_fail(self):
        with self.assertRaisesRegex(ValueError,'Unavailable storage'):Catalogue(self.outputs,self.archive/'absent',self.repo,self.stage).discover()
        (self.scope/'bad.svg').symlink_to(self.repo/'missing')
        with self.assertRaisesRegex(ValueError,'symlink'):self.discover()

if __name__=='__main__':unittest.main()
