"""Operation isolation and provisional preview integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from classical_conditioning.external_artifacts import ResolutionSession, _read_json
from classical_conditioning.figure_workspace import sha
from classical_conditioning.figures.preview import write_preview, read_registry
from scripts.paper.shared.working_copy import display_target, begin


class EfficiencyTests(unittest.TestCase):
    def test_metadata_parsed_once_but_same_size_mutation_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'index.json';p.write_text('{"a":1}')
            with self.assertRaisesRegex(ValueError,'metadata changed'):
                with ResolutionSession():
                    self.assertEqual(_read_json(p), {'a':1})
                    p.write_text('{"a":2}')
                    self.assertEqual(_read_json(p), {'a':1})
            with ResolutionSession():self.assertEqual(_read_json(p), {'a':2})

    def test_display_does_not_consume_unrelated_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scope={}
            for key in ['review','working_svg','legend']:
                p=root/key;p.write_text(key);scope[key]={'path':str(p),'sha256':sha(p.read_bytes())}
            state=root/'state.json';state.write_text(json.dumps({'draft':scope,'default_display':'draft','preserved_figures':{'unavailable':{'path':'missing'}}}))
            def resolve(path,expected_sha256,**kw):
                p=Path(path)
                self.assertEqual(sha(p.read_bytes()),expected_sha256)
                return p
            with patch('scripts.paper.shared.working_copy.resolve_artifact',side_effect=resolve):
                target=display_target(state);self.assertEqual(target['svg'],scope['working_svg'])
                ticket=begin(root/'ticket.json',state)
                self.assertNotIn('missing',ticket['starting_hashes'])

    def test_preview_externalizes_registry_and_preserves_native_art(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_bytes(b'saved results')
            svg=b'<svg xmlns="http://www.w3.org/2000/svg"><path id="data" d="M0 0L1 1"/><metadata id="editable-copy-provenance">{"scientific_registry":{"data":{"units":"s"}}}</metadata></svg>'
            out=Path(d)/'preview'
            manifest=write_preview(svg,'Saved outcome (s).',out,inputs=[{'path':str(p),'sha256':sha(p.read_bytes())}],command='render saved')
            self.assertTrue(manifest.is_file())
            self.assertEqual(read_registry((out/'preview.svg').read_bytes(),out)['scientific_registry']['data']['units'],'s')
            self.assertIn('Provisional exploratory preview',(out/'preview.html').read_text())
            self.assertNotIn('**Provisional exploratory preview', (out/'preview.html').read_text())
            with self.assertRaises(FileExistsError):write_preview(svg,'legend',out,inputs=[],command='render')
            (out/'registry.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Registry hash mismatch'):read_registry((out/'preview.svg').read_bytes(),out)

    def test_bad_input_leaves_no_preview(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_bytes(b'changed');out=Path(d)/'preview'
            with self.assertRaisesRegex(ValueError,'input changed'):
                write_preview(b'<svg/>','legend',out,inputs=[{'path':str(p),'sha256':sha(b'original')}],command='render')
            self.assertFalse(out.exists())

    def test_presentation_declaration_protects_scientific_paths(self):
        from classical_conditioning.figures.presentation_edit import verify_declared_changes
        old=b'<svg xmlns="http://www.w3.org/2000/svg"><path id="data" d="M0 0L1 1"/><text id="title">Old</text></svg>'
        new=old.replace(b'>Old<',b'>New<')
        verify_declared_changes(old,new,[{'element_id':'title','text':True}])
        with self.assertRaisesRegex(ValueError,'attribute change'):
            verify_declared_changes(old,new.replace(b'M0 0L1 1',b'M0 0L2 2'),[{'element_id':'title','text':True}])

    def test_legend_only_publication_preserves_svg_and_never_reconstructs(self):
        from classical_conditioning.figures.presentation_edit import publish_presentation
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scope={}
            for key,contents in [('review',b'old html'),('working_svg',b'<svg xmlns="http://www.w3.org/2000/svg"/>'),('legend',b'old legend')]:
                p=root/key;p.write_bytes(contents);scope[key]={'path':str(p),'sha256':sha(contents)}
            state=root/'state.json';state.write_text(json.dumps({'draft':scope,'default_display':'draft'}))
            paths=[state,*[Path(r['path']) for r in scope.values()]]
            ticket=root/'ticket.json';ticket.write_text(json.dumps({'state_path':str(state),'starting_hashes':{str(p.resolve()):sha(p.read_bytes()) for p in paths}}))
            candidate=root/'new.md';candidate.write_text('New explanatory legend')
            target={'html':scope['review'],'svg':scope['working_svg'],'legend':scope['legend']}
            with patch('scripts.paper.shared.working_copy.display_target',return_value=target):
                result=publish_presentation(state,ticket,candidate_legend=candidate)
            self.assertFalse(result['scientific_reconstruction'])
            self.assertEqual(Path(scope['working_svg']['path']).read_bytes(),b'<svg xmlns="http://www.w3.org/2000/svg"/>')
            self.assertEqual(Path(scope['legend']['path']).read_text(),'New explanatory legend')

    def test_incremental_addition_does_not_walk_unrelated_roots(self):
        from scripts.reviews.incremental import build_changed
        from scripts.reviews.versions import figure_page
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);out=base/'outputs';archive=base/'archive';repo=base/'repo'
            for p in [out/'review-library',out/'figures/figure-01',archive,repo/'records/repository-archive']:p.mkdir(parents=True)
            model={'schema_version':1,'versions':[],'inventory':[],'issues':[],'documented_gaps':[],'coverage':{}}
            page=out/'figures/figure-01/INDEX.html';page.write_bytes(figure_page(1,model,out))
            home=out/'START-HERE.html';home.write_text('0 source-bound artwork views; 0 recorded coverage exceptions')
            mp=out/'review-library/versions.json';mp.write_text(json.dumps(model))
            gp=out/'review-library/guards.json';gp.write_text('[]')
            publication={'snapshot':'review-library/full','versions':{'path':'review-library/versions.json','sha256':sha(mp.read_bytes())},'source_guards':{'path':'review-library/guards.json','sha256':sha(gp.read_bytes())},'payloads':[],
                'generated_pages':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p.read_bytes())} for p in [page,home,mp]]}
            (out/'review-library/catalogue-publication.json').write_text(json.dumps(publication))
            (repo/'records/repository-archive/verified-artifact-locations.json').write_text('{"entries":[]}')
            bundle=out/'figures/figure-01/panel-a/versions/new';bundle.mkdir(parents=True)
            (bundle/'preview.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Saved plot</text></svg>')
            (bundle/'legend.md').write_text('A matching legend')
            with patch('scripts.reviews.versions.source_files',side_effect=AssertionError('global walk')):
                result=build_changed(out,archive,repo,[bundle],['figure-01'])
            self.assertEqual(result['added_versions'],1)
            self.assertFalse(result['coverage']['global_scan_performed'])
            self.assertEqual(len(json.loads(mp.read_bytes())['versions']),1)
            from scripts.reviews.versions import Catalogue, source_files
            stage=base/'full-check';stage.mkdir()
            full=Catalogue(out.resolve(),archive.resolve(),repo.resolve(),stage,load_history=False).discover(include_states=False,reconcile=False)
            recorded=json.loads(mp.read_bytes())
            identity=lambda v:(v['id'],v['artifact_sha256'],v['legend']['text'])
            self.assertEqual({identity(v) for v in recorded['versions']},{identity(v) for v in full['versions']})
            self.assertFalse(any('incremental-browser' in str(p) for _,p in source_files(out,archive)))


    def test_compact_packaging_removes_only_identical_font_rules(self):
        from classical_conditioning.figures.preview import compact_svg
        first='@font-face{font-family:A;src:url(data:font/ttf;base64,AAAA)}'
        other='@font-face{font-family:B;src:url(data:font/ttf;base64,BBBB)}'
        svg=('<svg xmlns="http://www.w3.org/2000/svg"><style>'+first+other+'</style><style>'+first+'</style><text>Editable</text></svg>').encode()
        packed,_=compact_svg(svg)
        self.assertEqual(packed.count(first.encode()),1)
        self.assertIn(other.encode(),packed)
        self.assertIn(b'Editable',packed)

    def test_atlas_preview_renders_only_requested_saved_fish(self):
        import pandas as pd
        from contextlib import nullcontext
        from scripts.paper.figure_03.supplement import build_atlas as atlas
        with tempfile.TemporaryDirectory() as d:
            saved=Path(d)/'saved';saved.mkdir();(saved/'classifiers/rule').mkdir(parents=True)
            paths=[saved/'trial-data.parquet',saved/'heatmaps.npz',saved/'fish-inventory.csv',saved/'classifiers/rule/results.parquet']
            for path in paths:path.write_bytes(path.name.encode())
            (saved/'render-inputs.json').write_text(json.dumps([{'path':str(p),'sha256':sha(p.read_bytes())} for p in paths]))
            inventory=pd.DataFrame({'fish_id':['one','two'],'condition_id':['control','delay']})
            trials=pd.DataFrame({'fish_id':['one','two'],'trial_number':[1,1]})
            with patch.object(atlas.pd,'read_parquet',return_value=trials),patch.object(atlas.pd,'read_csv',return_value=inventory),patch.object(atlas,'results_for',return_value={}),patch.object(atlas,'command_sources',return_value=[]),patch.object(atlas.np,'load',return_value=nullcontext({})),patch('classical_conditioning.figures.individual_atlas.fish_card',return_value=b'<svg xmlns="http://www.w3.org/2000/svg"/>') as card,patch.object(atlas,'classify',side_effect=AssertionError('classification forbidden')):
                manifests=atlas.preview_fish(saved,Path(d)/'task',['two'])
            self.assertEqual(card.call_count,1)
            self.assertEqual(len(manifests),1)
            self.assertEqual(json.loads(Path(manifests[0]).read_text())['parameters']['fish'],'two')
