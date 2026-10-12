"""Assembly checks use tiny synthetic SVGs; no real figures or pointer updates."""
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import assemble_svg_figure as assembly
from classical_conditioning import artifacts
from classical_conditioning.external_artifacts import resolve_artifact
from classical_conditioning.figures import assembly_sources
from classical_conditioning.figure_workspace import StaleFigureState


class AssemblyIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = self.root/'a.svg'; self.b = self.root/'b.svg'; self.alt = self.root/'candidate.svg'
        for path, color in ((self.a,'red'),(self.b,'blue'),(self.alt,'green')):
            path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect id="signal" width="10" height="10" fill="{color}"/></svg>')
        self.layout = self.root/'layout.json'
        self.data = {'canvas':[30,10], 'panels':[
            {'id':'A','source':self.a.name,'box':[0,0,10,10]},
            {'id':'B','source':self.b.name,'box':[20,0,10,10]}]}
        self.save_layout()
        self.output = self.root/'assembly.svg'; self.sidecar = self.root/'assembly.svg.json'

    def save_layout(self): self.layout.write_text(json.dumps(self.data))
    def render(self, **kwargs): return assembly.render(self.layout,self.output,**kwargs)

    def test_plan_is_no_write_and_does_not_need_exporter(self):
        before = {p.name:p.read_bytes() for p in self.root.iterdir()}
        result = subprocess.run([sys.executable,str(Path(assembly.__file__)),str(self.layout),'--plan'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'ready')
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_missing_source_and_nonfinite_geometry_are_blocked(self):
        self.a.unlink()
        report = assembly.preflight(self.layout,strict=True)
        self.assertFalse(report['can_render']); self.assertIn('unavailable',report['panels'][0]['source']['status'])
        self.data['canvas']=[float('nan'),10];self.save_layout()
        self.assertFalse(assembly.preflight(self.layout)['can_render'])
        self.assertFalse(self.output.exists())

    def test_hash_mismatch_fails_before_creating_outputs(self):
        self.data['panels'][0]['frozen_sha256']='0'*64;self.save_layout()
        report=assembly.preflight(self.layout)
        self.assertFalse(report['can_render']);self.assertIn('mismatch',str(report))
        with self.assertRaises(ValueError):self.render()
        self.assertFalse(self.output.exists())

    def test_archive_resolver_handles_windows_source_and_disconnected_archive(self):
        repo=self.root/'repo'; metadata=repo/'docs/maintenance/archive';metadata.mkdir(parents=True)
        archive=self.root/'ssd/archive';target=archive/'payload/reviews/a.svg';target.parent.mkdir(parents=True);target.write_bytes(self.a.read_bytes())
        (metadata/'transfer-inventory.json').write_text(json.dumps({'archive_root':str(archive),'files':[{'path':'reviews/a.svg','sha256':assembly.digest(self.a),'action':'archive_to_ssd'}]}))
        self.data['storage_root']='C:/Users/joaquim/Documents/ClassicalConditioning'
        self.data['panels']=[{'id':'A','source':'reviews/a.svg','box':[0,0,10,10],'frozen_sha256':assembly.digest(self.a)}];self.save_layout()
        with patch.object(assembly_sources,'resolve_artifact',functools.partial(resolve_artifact,repo=repo)):
            result=self.render();self.assertEqual(result['panels'][0]['source'],str(target.resolve()))
            target.unlink()
            self.assertFalse(assembly.preflight(self.layout)['can_render'])
        self.assertFalse((repo/'reviews').exists())

    def test_frozen_override_is_candidate_and_keeps_original_provenance(self):
        frozen=assembly.digest(self.a);self.data['panels'][0]['frozen_sha256']=frozen;self.save_layout()
        original_layout=self.layout.read_bytes(); self.render()
        result=self.render(source_overrides={'A':self.alt.name})
        panel=result['panels'][0]
        self.assertEqual(panel['original_source']['expected_sha256'],frozen)
        self.assertEqual(panel['sha256'],assembly.digest(self.alt))
        self.assertIn('candidate',panel['selection_status'])
        untouched=result['comparison']['panels'][1]
        self.assertTrue(untouched['unchanged']);self.assertTrue(untouched['embedded_payload_unchanged'])
        self.assertEqual(self.layout.read_bytes(),original_layout);self.assertEqual(assembly.digest(self.a),frozen)
        self.assertIn('--replace-panel-source',result['reproduction_command'])

    def test_replacement_rejects_unrequested_panel_change(self):
        self.render();original=(self.output.read_bytes(),self.sidecar.read_bytes())
        self.b.write_text(self.b.read_text().replace('blue','purple'))
        with self.assertRaisesRegex(ValueError,'Unrequested panel.*B'):
            self.render(source_overrides={'A':self.alt.name})
        self.assertEqual(original,(self.output.read_bytes(),self.sidecar.read_bytes()))

    def test_explicit_baseline_checks_saved_assembly(self):
        self.render();baseline=self.sidecar
        self.output.write_text('manual edit')
        with self.assertRaises(ValueError):
            assembly.render(self.layout,self.root/'next.svg',baseline_manifest=baseline)
        self.assertFalse((self.root/'next.svg').exists())

    def test_source_change_during_composition_preserves_outputs(self):
        self.render();before=(self.output.read_bytes(),self.sidecar.read_bytes());real=assembly._compose
        def change(*args):
            result=real(*args);self.a.write_text(self.a.read_text().replace('red','black'));return result
        with patch.object(assembly,'_compose',side_effect=change):
            with self.assertRaises(StaleFigureState):self.render()
        self.assertEqual(before,(self.output.read_bytes(),self.sidecar.read_bytes()))

    def test_manual_output_edit_during_composition_is_preserved(self):
        self.render();real=assembly._compose
        def edit(*args):
            result=real(*args);self.output.write_bytes(b'another editors work');return result
        with patch.object(assembly,'_compose',side_effect=edit):
            with self.assertRaises(StaleFigureState):self.render()
        self.assertEqual(self.output.read_bytes(),b'another editors work')

    def test_interrupted_pair_publication_restores_previous_pair(self):
        self.render();before=(self.output.read_bytes(),self.sidecar.read_bytes());real=artifacts.os.replace
        def interrupt(source,target):
            if Path(source).parent.name.startswith('.assembly-') and Path(source).name==self.sidecar.name:
                raise OSError('simulated interruption')
            return real(source,target)
        with patch.object(artifacts.os,'replace',side_effect=interrupt):
            with self.assertRaises(OSError):self.render(source_overrides={'A':self.alt.name})
        self.assertEqual(before,(self.output.read_bytes(),self.sidecar.read_bytes()))
        self.assertFalse(list(self.root.glob('.assembly-*')));self.assertFalse(Path(str(self.sidecar)+'.lock').exists())

    def test_saved_byte_verification_failure_rolls_back_pair(self):
        self.render();before=(self.output.read_bytes(),self.sidecar.read_bytes());real=artifacts.os.replace
        def corrupt(source,target):
            result=real(source,target)
            if Path(source).parent.name.startswith('.assembly-') and Path(source).name==self.output.name:
                self.output.write_text('corrupt saved bytes')
            return result
        with patch.object(artifacts.os,'replace',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'Saved assembly hash mismatch'):self.render(source_overrides={'A':self.alt.name})
        self.assertEqual(before,(self.output.read_bytes(),self.sidecar.read_bytes()))

    def test_missing_destination_cannot_fabricate_volume(self):
        destination=Path('/Volumes/UNAVAILABLE-ASSEMBLY-TEST/preview.svg')
        with patch.object(assembly.os.path,'ismount',return_value=False):
            with self.assertRaises(FileNotFoundError):assembly.render(self.layout,destination)
        self.assertFalse(destination.parent.exists())

    def test_managed_working_copy_uses_existing_guarded_workflow(self):
        repo=self.root/'repo';states=repo/'docs/analysis/figures/reviews/figure-01';states.mkdir(parents=True)
        (states/'current-state.json').write_text(json.dumps({'draft':{'working_svg':{'path':str(self.output)}}}))
        with patch.object(assembly,'REPO',repo):
            with self.assertRaisesRegex(ValueError,'managed by a current figure state'):self.render()
        self.assertFalse(self.output.exists())

    def test_incomplete_pair_and_input_overwrite_are_rejected(self):
        self.output.write_text('manual candidate without sidecar')
        with self.assertRaisesRegex(ValueError,'incomplete'):self.render()
        with self.assertRaisesRegex(ValueError,'bound input'):assembly.render(self.layout,self.a)
        self.assertIn('red',self.a.read_text())

    def test_cli_plan_checks_baseline_and_destination_without_writing(self):
        self.render();self.b.write_text(self.b.read_text().replace('blue','purple'))
        target=self.root/'next.svg'
        command=[sys.executable,str(Path(assembly.__file__)),str(self.layout),'--plan','--output',str(target),
                 '--baseline-manifest',str(self.sidecar),'--replace-panel-source','A='+self.alt.name]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertIn('Unrequested panel',str(json.loads(result.stdout)))
        self.assertFalse(target.exists())

    def test_baseline_panel_claims_match_saved_artwork(self):
        self.render();metadata=json.loads(self.sidecar.read_text())
        metadata['panels'][1]['panel_payload_sha256']='0'*64
        self.sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'provenance does not match'):
            self.render(source_overrides={'A':self.alt.name})

    def test_failed_optional_export_preserves_previous_pair(self):
        self.render();before=(self.output.read_bytes(),self.sidecar.read_bytes())
        with patch.object(assembly,'export_with_inkscape',side_effect=RuntimeError('export failed')):
            with self.assertRaises(RuntimeError):self.render(export_formats=('pdf',))
        self.assertEqual(before,(self.output.read_bytes(),self.sidecar.read_bytes()))
        self.assertFalse(self.output.with_suffix('.pdf').exists())

    def test_explicit_extra_export_participates_in_verified_transaction(self):
        def synthetic(svg,formats,**kwargs):
            for fmt in formats:svg.with_suffix('.'+fmt).write_bytes(b'synthetic export transaction fixture')
        with patch.object(assembly,'export_with_inkscape',side_effect=synthetic):
            result=self.render(export_formats=('pdf',))
        extra=result['extra_exports'][0]
        self.assertEqual(extra['sha256'],assembly.digest(self.output.with_suffix('.pdf')))
        self.assertEqual(Path(extra['path']),self.output.resolve().with_suffix('.pdf'))
        self.output.with_suffix('.pdf').write_bytes(b'another editors export')
        with self.assertRaisesRegex(ValueError,'unverified or changed'):
            self.render(export_formats=('pdf',))

    def test_requested_panel_geometry_change_preserves_other_panel_context(self):
        self.render();self.data['panels'][0]['box']=[0,0,12,10];self.save_layout()
        result=self.render(source_overrides={'A':self.alt.name})
        unchanged=next(p for p in result['comparison']['panels'] if p['id']=='B')
        self.assertTrue(unchanged['unchanged']);self.assertTrue(unchanged['clipping_unchanged'])

    def test_baseline_clipping_and_shared_context_claims_match_artwork(self):
        for attribute in ('clipping','shared-font'):
            with self.subTest(attribute=attribute):
                target=self.root/(attribute+'.svg')
                assembly.render(self.layout,target)
                metadata_path=target.with_suffix('.svg.json');metadata=json.loads(metadata_path.read_text())
                root=assembly.ET.parse(target).getroot()
                if attribute=='clipping':
                    node=root.find(f"./{assembly.q('defs')}/{assembly.q('clipPath')}[@id='clip-B']/{assembly.q('rect')}")
                    node.set('width','1')
                else:root.set('font-family','UnrecordedFont')
                target.write_bytes(assembly.ET.tostring(root))
                metadata['output_sha256']=assembly.digest(target);metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError,'clipping provenance|shared rendering context'):
                    assembly.render(self.layout,self.root/(attribute+'-candidate.svg'),source_overrides={'A':self.alt.name},baseline_manifest=metadata_path)

    def test_registered_frozen_bytes_cannot_be_overwritten_as_candidate(self):
        self.render();before=self.output.read_bytes()
        repo=self.root/'repo';records=repo/'records/frozen-analyses/figure-01/panel-a/version/manifests'
        records.mkdir(parents=True)
        (records/'freeze.json').write_text(json.dumps({'exports':[{'path':str(self.output),'sha256':assembly.digest(self.output)}]}))
        with patch.object(assembly,'REPO',repo):
            with self.assertRaisesRegex(ValueError,'accepted frozen bytes'):
                self.render(source_overrides={'A':self.alt.name})
        self.assertEqual(self.output.read_bytes(),before)

    def test_source_viewbox_and_raster_crop_errors_are_visible_in_plan(self):
        self.a.write_text(self.a.read_text().replace('0 0 10 10','0 0 -10 10'))
        self.assertFalse(assembly.preflight(self.layout)['can_render'])
        import base64
        png=self.root/'pixel.png'
        png.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lXcAAAAASUVORK5CYII='))
        self.data['panels']=[{'id':'A','source':png.name,'box':[0,0,10,10],'view_box':[0,0,1,1]}]
        self.save_layout();report=assembly.preflight(self.layout)
        self.assertFalse(report['can_render']);self.assertIn('requires an SVG',str(report))

    def test_second_publisher_cannot_replace_under_held_lock(self):
        self.render();before=(self.output.read_bytes(),self.sidecar.read_bytes())
        with assembly.state_lock(self.sidecar):
            with self.assertRaises(StaleFigureState):self.render(source_overrides={'A':self.alt.name})
        self.assertEqual(before,(self.output.read_bytes(),self.sidecar.read_bytes()))

    def test_cli_svg_build_needs_no_inkscape_and_requires_explicit_overwrite(self):
        args=[sys.executable,str(Path(assembly.__file__)),str(self.layout),'--output',str(self.output)]
        first=subprocess.run(args,capture_output=True,text=True);self.assertEqual(first.returncode,0,first.stderr)
        before=self.output.read_bytes()
        second=subprocess.run(args,capture_output=True,text=True);self.assertNotEqual(second.returncode,0)
        self.assertEqual(before,self.output.read_bytes())


if __name__=='__main__':unittest.main()
