"""Behavior preservation for the scoped maintenance successors; no paper data needed."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tests.characterization import folder_audit_contracts as original
from classical_conditioning.artifacts import write_parquet_table
from classical_conditioning.ingestion.protocol_audit import audit_fixed10_protocol
from classical_conditioning.operations.recovery import dispositions, STAGES
from classical_conditioning.operations.run_completion import check_run
from classical_conditioning.figures import individual_atlas as atlas

ROOT=Path(__file__).resolve().parents[1]

class PreservedContracts(unittest.TestCase):
    def test_protocol_audit_preserves_rows_errors_and_timing_decisions(self):
        cases=["Type Beg End\nCycle 30000 40000\nReinforcer 29000 29050\n",
               "Type Beg End\nCycle 10 9\nCycle 10 10010\nwrong row\nOther 1 2\n",
               "\ufeffType Beg End\n\nCycle a 20\nReinforcer 4 3\n"]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'protocol.txt'
            for content in cases:
                path.write_text(content)
                self.assertEqual(audit_fixed10_protocol(path),original.protocol(path))

    def test_recovery_preserves_all_prefix_and_failure_cases(self):
        for completed in range(len(STAGES)+1):
            values={s:[i<completed]*3 for i,s in enumerate(STAGES)}
            self.assertEqual(dispositions(values),original.dispositions(values))
        for values in [{**{s:[False]*3 for s in STAGES},STAGES[0]:[True,False,True]},
                       {**{s:[False]*3 for s in STAGES},STAGES[1]:[True]*3}]:
            errors=[]
            for call in [dispositions,original.dispositions]:
                with self.assertRaises(ValueError) as caught:call(values)
                errors.append(str(caught.exception))
            self.assertEqual(*errors)

    def test_completion_preserves_status_errors_and_verifier_calls(self):
        plan={'selected':[{'recording_id':'A'},{'recording_id':'B'}],'held':[{'recording_id':'held'}]}
        row=lambda rid:{'recording_id':rid,'status':'completed','lineage_verified':True,'stages':list(STAGES)}
        cases=[{'results':[row('A'),row('B')],'pending':[]},
               {'results':[row('A')],'pending':['B']},
               {'results':[row('A'),row('A')],'pending':['B']},
               {'results':[row('outside')],'pending':['A','B']},
               {'results':[{**row('A'),'lineage_verified':False},row('B')],'pending':[]},
               {'results':[row('A'),row('B')],'pending':[],'existing_files_changed':['old-file']}]
        for complete in cases:
            calls=[]
            def verifier(expected,result):calls.append((expected['recording_id'],result['recording_id']))
            before=original.check_run(deepcopy(plan),deepcopy(complete),verifier);expected_calls=calls[:];calls.clear()
            after=check_run(deepcopy(plan),deepcopy(complete),verifier)
            self.assertEqual(after,before);self.assertEqual(calls,expected_calls)
        def fail(*_):raise ValueError('lineage mismatch')
        self.assertEqual(check_run(plan,cases[0],fail),original.check_run(plan,cases[0],fail))

    def test_shared_and_compatibility_writers_preserve_bytes_and_receipts(self):
        modules=['analysis.cohort_outcomes','analysis.inference.fish_bootstrap',
                 'analysis.inference.fish_permutation','analysis.inference.mixed_effects',
                 'analysis.inference.model_input']
        calls=[write_parquet_table]+[importlib.import_module('classical_conditioning.'+m)._write_parquet for m in modules]
        frames=[pd.DataFrame({'value':[1.,np.nan,np.inf],'fish':['one','two','three']},index=pd.Index([8,4,3],name='discarded_index')),
                pd.DataFrame({'label':pd.Categorical(['a','b',None]),'count':pd.array([1,None,3],dtype='Int64')}),
                pd.DataFrame({'time':pd.to_datetime(['2026-01-01',None],utc=True)})]
        with tempfile.TemporaryDirectory() as folder:
            for frame in frames:
                before=Path(folder)/'before.parquet';expected=original._write_parquet(before,frame)
                for call in calls:
                    after=Path(folder)/'after.parquet';observed=call(after,frame)
                    self.assertEqual(before.read_bytes(),after.read_bytes())
                    self.assertEqual({k:v for k,v in expected.items() if k!='path'},
                                     {k:v for k,v in observed.items() if k!='path'})

class ImportAndStyleIsolation(unittest.TestCase):
    def test_atlas_import_preserves_caller_style_and_backend(self):
        code="""import classical_conditioning
import matplotlib as mpl
import matplotlib.pyplot
mpl.rcParams.update({'font.size':17,'axes.linewidth':2,'svg.hashsalt':'caller'})
before=dict(mpl.rcParams)
import classical_conditioning.figures.individual_atlas
assert dict(mpl.rcParams)==before
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_atlas_source_retains_lazy_window_reader_compatibility(self):
        from classical_conditioning.analysis.review import individual_atlas as source
        sentinel=object()
        with patch('classical_conditioning.figures.pooled_log_heatmap.read_windows',return_value=sentinel) as read:
            self.assertIs(source.read_windows('source',['column'],[(0,1)]),sentinel)
            read.assert_called_once_with('source',['column'],[(0,1)])

    def test_historical_style_preserves_svg_and_restores_caller_on_failure(self):
        def figure():
            fig,ax=plt.subplots(figsize=(2,1));ax.plot([0,1],[1,0]);ax.set_xlabel('Time (s)');return fig
        with mpl.rc_context(rc=original.LEGACY_RC):expected=original.svg_bytes(figure(),'test',50.8,25.4)
        with mpl.rc_context({'font.size':17,'svg.hashsalt':'caller'}):
            before=dict(mpl.rcParams)
            @atlas._legacy_atlas_style
            def render():return atlas.svg_bytes(figure(),'test',50.8,25.4)
            self.assertEqual(render(),expected);self.assertEqual(dict(mpl.rcParams),before)
            @atlas._legacy_atlas_style
            def fail():raise RuntimeError('render failure')
            with self.assertRaises(RuntimeError):fail()
            self.assertEqual(dict(mpl.rcParams),before)

    def test_bare_script_dependencies_are_bound_without_executing_them(self):
        from classical_conditioning import command_provenance as provenance
        from classical_conditioning.figure_workspace import Snapshot,StaleFigureState
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);commands=root/'scripts';commands.mkdir()
            entry=commands/'entry.py';helper=commands/'helper.py'
            entry.write_text('from helper import value\n');helper.write_text('raise RuntimeError("never import")\n')
            with patch.object(provenance,'REPOSITORY',root):
                provenance.command_sources.cache_clear()
                sources=provenance.command_sources(entry)
            provenance.command_sources.cache_clear()
            self.assertIn(helper.resolve(),sources);snapshot=Snapshot(sources)
            helper.write_text('raise RuntimeError("changed")\n')
            with self.assertRaises(StaleFigureState):snapshot.check()

    def test_maintenance_control_names_inside_archive_do_not_hide_artwork(self):
        from scripts.reviews.versions import source_files
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);outputs=base/'outputs';archive=base/'archive';maintenance=base/'maintenance'
            for p in [outputs,archive,maintenance]:p.mkdir()
            control=archive/'repository-history/new-maintenance';control.mkdir(parents=True)
            (control/'report.json').write_text('{}');(control/'supporting.svg').write_text('<svg/>')
            (maintenance/'task.log').write_text('outside the figure trees')
            observed={p for _,p in source_files(outputs,archive)}
            self.assertEqual(observed,{control/'report.json',control/'supporting.svg'})


class CandidateAndRoutingSafety(unittest.TestCase):
    def setUp(self):
        original=list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None),original))

    def test_candidate_validation_does_not_create_missing_roots(self):
        from classical_conditioning.operations.candidate_paths import candidate_directory, evidence_file
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder)
            target=parent/'new-candidate'
            self.assertEqual(candidate_directory(target),target.resolve())
            self.assertFalse(target.exists())
            with self.assertRaises(FileNotFoundError):candidate_directory(parent/'missing'/'child')
            with self.assertRaises(FileNotFoundError):evidence_file(parent/'missing.json')
            with self.assertRaises(ValueError):candidate_directory(ROOT/'outputs',existing=False)
            with self.assertRaises(ValueError):candidate_directory(parent/'Raw Data'/'candidate')
            with patch.object(Path,'is_mount',return_value=False):
                with self.assertRaises(FileNotFoundError):candidate_directory('/Volumes/Unavailable-audit-volume/candidate')
            self.assertFalse((parent/'missing').exists())

    def test_flat_compatibility_module_routes_to_the_scoped_implementation(self):
        code="""import scripts.process_trace10_additions as old
import scripts.workflows.trace.ingest_fixed10_additions as new
assert old is new
assert callable(old.protocol)
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_spacing_cli_passes_paths_without_overriding_driver_globals(self):
        code="""from pathlib import Path
from unittest.mock import patch
import sys
from scripts.paper.figure_02.trace10 import publish_spacing as entry
before={k:v for k,v in entry.finish.__dict__.items() if k in {'TEMP','FINAL','DEFAULT_FINAL'}}
with patch.object(entry.finish,'verify') as call, patch.object(sys,'argv',['command','verify','--candidate-dir','/explicit/candidate','--browser-evidence','/explicit/evidence.json']):
    entry.cli()
    call.assert_called_once_with(Path('/explicit/candidate'),browser_evidence=Path('/explicit/evidence.json'))
after={k:v for k,v in entry.finish.__dict__.items() if k in {'TEMP','FINAL','DEFAULT_FINAL'}}
assert before==after
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_changed_browser_report_aborts_before_publication(self):
        from scripts.paper.figure_01.trace10_example import publish as entry
        from classical_conditioning.figure_workspace import StaleFigureState
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);stage=root/'candidate';out=root/'output';stage.mkdir();out.mkdir()
            oldsvg=root/'old.svg';oldhtml=root/'old.html';oldsvg.write_bytes(b'original-svg');oldhtml.write_bytes(b'original-html')
            state={'draft':{'working_svg':{'path':str(oldsvg)},'review':{'path':str(oldhtml)}},'verification':{}}
            (stage/'ticket.json').write_text(json.dumps({'hashes':{},'state':state}))
            (stage/'candidate.svg').write_bytes(b'candidate-svg');(stage/'candidate.html').write_bytes(b'candidate-html')
            browser=stage/'browser.json';browser.write_text('{}')
            def measured(*_):
                browser.write_text('{"changed":true}')
                return {'status':'passed','issues':[]}
            def guarded_publish(_scope,_state,snapshot,**_):snapshot.check()
            with patch.object(entry,'OUT',out),patch.object(entry,'SCOPE',root/'state.json'),patch.object(entry,'command_sources',return_value=[]),patch.object(entry,'assess',side_effect=measured),patch.object(entry,'publish_state',side_effect=guarded_publish):
                with self.assertRaises(StaleFigureState):entry.publish('Synthetic unit-test review',stage)
            self.assertEqual(oldsvg.read_bytes(),b'original-svg')
            self.assertEqual(oldhtml.read_bytes(),b'original-html')

    def test_snapshot_extension_cannot_overwrite_an_earlier_identity(self):
        from classical_conditioning.figure_workspace import Snapshot,StaleFigureState
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'input';file.write_bytes(b'old')
            snapshot=Snapshot([file]);original=dict(snapshot.hashes)
            file.write_bytes(b'changed')
            with self.assertRaises(StaleFigureState):snapshot.include([file])
            self.assertEqual(snapshot.hashes,original)
            with self.assertRaises(StaleFigureState):snapshot.bind_hashes({'new-path':'new-hash',str(file.resolve()):'different'})
            self.assertEqual(snapshot.hashes,original)
            with self.assertRaises(StaleFigureState):snapshot.check()

    def test_workflow_facades_preserve_utf8_console_setup(self):
        import runpy
        for old,new in [('process_trace10_additions.py','ingest_fixed10_additions'),('resume_trace10_additions.py','resume_fixed10_additions')]:
            with patch.object(sys,'path',list(sys.path)):
                module=importlib.import_module('scripts.workflows.trace.'+new)
                with patch.object(module,'main') as call,patch.object(sys,'stdout') as output:
                    runpy.run_path(str(ROOT/'scripts'/old),run_name='__main__')
                    output.reconfigure.assert_called_once_with(encoding='utf-8')
                    call.assert_called_once_with()

    def test_guarded_facade_includes_the_extracted_helper(self):
        from classical_conditioning.command_provenance import command_sources
        command_sources.cache_clear()
        sources=command_sources(ROOT/'scripts/process_trace10_additions.py')
        self.assertIn(ROOT/'scripts/workflows/trace/ingest_fixed10_additions.py',sources)
        self.assertIn(ROOT/'src/classical_conditioning/ingestion/protocol_audit.py',sources)

    def test_authenticated_successor_paths_disable_checkout_conversion(self):
        routing=json.loads((ROOT/'records/repository-archive/folder-audit-follow-up-20261028/command-successors.json').read_text())
        paths=list(routing['moves'].values())+[
            'src/classical_conditioning/operations/candidate_paths.py',
            'src/classical_conditioning/operations/recovery.py',
            'src/classical_conditioning/operations/run_completion.py',
            'src/classical_conditioning/ingestion/protocol_audit.py']
        result=subprocess.run(['git','check-attr','-z','--stdin','text'],cwd=ROOT,
            input=b''.join(p.encode()+b'\0' for p in paths),capture_output=True,check=True)
        fields=result.stdout.split(b'\0')[:-1]
        self.assertTrue(all(fields[i+2]==b'unset' for i in range(0,len(fields),3)))

    def test_retired_plan_links_resolve_to_complete_preserved_accounts(self):
        from classical_conditioning.external_artifacts import resolve_artifact
        for name in ['13_SCRIPT_COMMAND_ORGANIZATION.md','15_REPOSITORY_FOLDER_AUDIT_FOLLOW_UP.md']:
            original = resolve_artifact('Plans/' + name, repo=ROOT)
            self.assertTrue(original.is_file())
            self.assertGreater(len(original.read_text()), 1000)

    def test_every_original_plan_block_survives_with_a_maintained_owner(self):
        import hashlib,re
        base=ROOT/'records/repository-history/2026-10-28-folder-audit-implementation'
        mapping=json.loads((base/'requirements-map.json').read_text())
        for source,expected in mapping['source_identities'].items():
            rows=[r for r in mapping['blocks'] if r['source']==source]
            original=ROOT/rows[0]['preserved_original'];raw=original.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),expected)
            blocks=[b for b in re.split(r'\n\s*\n',raw.decode()) if b.strip()]
            self.assertEqual(len(rows),len(blocks))
            for row,block in zip(rows,blocks):
                self.assertEqual(row['block_sha256'],hashlib.sha256(block.encode()).hexdigest())
                self.assertTrue(row['maintained_owners'])
                for owner in row['maintained_owners']:
                    from classical_conditioning.external_artifacts import resolve_artifact
                    self.assertTrue(resolve_artifact(owner,repo=ROOT).is_file(),owner)

if __name__=='__main__':unittest.main()
