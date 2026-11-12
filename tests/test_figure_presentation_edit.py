"""Native-scale evidence and conservative presentation publication; fixtures only."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
import xml.etree.ElementTree as ET

from classical_conditioning.figure_workspace import (
    StaleFigureState, measured_presentation, native_presentation_width, sha,
)
from classical_conditioning.figures import layout_frames as frames
from classical_conditioning.figures.presentation_edit import publish_presentation, verify_declared_changes
from scripts.paper.shared.working_copy import begin
from tests.test_layout_frames import browser


def artwork(width=136):
    # A nested half-size transform maps 12-pt source type to 6 pt at every width.
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 30">'
           '<g id="panel"><path id="data" d="M3 3L20 12" stroke="black" stroke-width=".352777777778"/>'
           '<g id="labels" transform="translate(5 5) scale(.5)">'
           '<text id="label" x="0" y="10" font-size="4.233333333333">Before</text>'
           '</g></g></svg>').encode()
    return frames.embed_spec(raw, dict(scope='figure', inset_mm=1, final_size_mm=[width, 30],
        allocations=[dict(id='panel', kind='panel', box_mm=[0, 0, width, 30], group_ids=['panel'])],
        background_ids=[]))


class PresentationEditTests(unittest.TestCase):
    def selected(self, root, width=136):
        scope = {}
        for key, data in [('review', b'previous HTML'), ('working_svg', artwork(width)), ('legend', b'Previous legend')]:
            path = root / key
            path.write_bytes(data)
            scope[key] = {'path': str(path), 'sha256': sha(data)}
        state = root / 'state.json'
        state.write_text(json.dumps({'figure_id': 'fixture', 'default_display': 'draft', 'draft': scope}))
        ticket = root / 'ticket.json'
        begin(ticket, state)
        return state, ticket, scope

    def candidate(self, root, width=136):
        svg = artwork(width).replace(b'>Before<', b'>After<')
        path = root / 'candidate.svg'
        path.write_bytes(svg)
        evidence = frames.assess(svg, browser(svg))
        self.assertEqual(evidence['status'], 'passed')
        evidence['browser'] = browser(svg)
        report = root / 'layout.json'
        report.write_text(json.dumps(evidence))
        return path, report

    def publish(self, state, ticket, candidate, evidence):
        return publish_presentation(state, ticket, candidate_svg=candidate, layout_evidence=evidence,
            changes=[{'element_id': 'label', 'text': True}],
            visual_review='Synthetic fixture only; not scientific artwork review')

    def test_publisher_measures_native_width_and_nested_type(self):
        for width in (89, 136, 183):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                state, ticket, scope = self.selected(root, width)
                old = state.read_bytes()
                candidate, evidence = self.candidate(root, width)
                result = self.publish(state, ticket, candidate, evidence)
                selected = json.loads(state.read_bytes())['draft']
                report = json.loads(Path(selected['effective_presentation']['path']).read_bytes())
                self.assertEqual(report['final_width_mm'], width)
                self.assertAlmostEqual(report['texts'][0]['font_height_pt'], 6)
                self.assertAlmostEqual(report['strokes'][0]['normal_width_pt'], 1)
                self.assertEqual(report['svg_sha256'], sha(candidate.read_bytes()))
                self.assertEqual((Path(result['history']) / 'state.json').read_bytes(), old)
                self.assertEqual(Path(scope['legend']['path']).read_bytes(), b'Previous legend')
                self.assertFalse(result['scientific_reconstruction'])

    def test_historical_default_and_explicit_width_still_work(self):
        svg = artwork(136)
        self.assertEqual(measured_presentation(svg)['final_width_mm'], 183)
        explicit = measured_presentation(svg, width_mm=68)
        self.assertEqual(explicit['final_width_mm'], 68)
        self.assertAlmostEqual(explicit['texts'][0]['font_height_pt'], 3)

    def test_native_dimensions_must_be_explicit_consistent_and_finite(self):
        for attr, value in [('width', None), ('height', None), ('width', '136px'),
                            ('width', '0mm'), ('width', 'nanmm'), ('width', 'infmm'),
                            ('width', '183mm'), ('viewBox', '0 0 0 30'), ('viewBox', '0 0 nan 30')]:
            with self.subTest(attr=attr, value=value):
                root = ET.fromstring(artwork())
                if value is None:
                    root.attrib.pop(attr)
                else:
                    root.set(attr, value)
                with self.assertRaises(ValueError):
                    native_presentation_width(ET.tostring(root))
        root = ET.fromstring(artwork())
        root.set('width', '183mm')
        root.set('viewBox', '0 0 183 30')
        with self.assertRaisesRegex(ValueError, 'frame final_size_mm'):
            native_presentation_width(ET.tostring(root))

    def test_text_positions_and_text_only_group_transforms_remain_supported(self):
        old = artwork()
        new = old.replace(b'translate(5 5) scale(.5)', b'translate(6 5) scale(.5)').replace(b'x="0"', b'x="1"')
        verify_declared_changes(old, new, [{'element_id': 'labels', 'attributes': ['transform']},
                                          {'element_id': 'label', 'attributes': ['x']}])

    def test_data_and_ancestor_coordinates_require_mapping_aware_route(self):
        for identity in ('panel', 'data'):
            with self.subTest(identity=identity):
                old = artwork()
                new = old.replace(('id="'+identity+'"').encode(), ('id="'+identity+'" transform="scale(1 2)"').encode())
                with self.assertRaisesRegex(ValueError, 'mapping-aware'):
                    verify_declared_changes(old, new, [{'element_id': identity, 'attributes': ['transform']}])
        for element in ('<image id="data" href="data:image/png;base64,AAAA"/>',
                        '<use id="data" href="#shape"/>'):
            old = ('<svg xmlns="http://www.w3.org/2000/svg">'+element+'</svg>').encode()
            new = old.replace(b'id="data"', b'id="data" x="20"')
            with self.assertRaisesRegex(ValueError, 'mapping-aware'):
                verify_declared_changes(old, new, [{'element_id': 'data', 'attributes': ['x']}])

    def test_stale_candidate_evidence_leaves_selected_bundle_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ticket, scope = self.selected(root)
            old = {p: p.read_bytes() for p in [state, *[Path(r['path']) for r in scope.values()]]}
            candidate, evidence = self.candidate(root)
            report = json.loads(evidence.read_bytes())
            report['artifact_sha256'] = 'stale'
            evidence.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'does not verify'):
                self.publish(state, ticket, candidate, evidence)
            self.assertTrue(all(p.read_bytes() == data for p, data in old.items()))
            self.assertFalse((root / 'presentation-history').exists())

    def test_stale_ticket_preserves_manual_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ticket, scope = self.selected(root)
            before = state.read_bytes()
            manual = Path(scope['working_svg']['path'])
            manual.write_bytes(manual.read_bytes().replace(b'Before', b'Manual'))
            legend = root / 'new.md'
            legend.write_text('Candidate legend')
            with self.assertRaises(StaleFigureState):
                publish_presentation(state, ticket, candidate_legend=legend)
            self.assertEqual(state.read_bytes(), before)
            self.assertIn(b'Manual', manual.read_bytes())

    def test_changed_registry_resource_is_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, _, scope = self.selected(root)
            sidecar = root / 'registry.json'
            sidecar.write_text('{}')
            svg = ET.fromstring(Path(scope['working_svg']['path']).read_bytes())
            ET.SubElement(svg, frames.SVG+'metadata', {'id': 'editable-copy-provenance'}).text = json.dumps(
                {'registry_ref': {'path': sidecar.name, 'sha256': sha(sidecar.read_bytes())}})
            raw = ET.tostring(svg)
            Path(scope['working_svg']['path']).write_bytes(raw)
            scope['working_svg']['sha256'] = sha(raw)
            state.write_text(json.dumps({'draft': scope, 'default_display': 'draft'}))
            ticket = root / 'resource-ticket.json'
            begin(ticket, state)
            before = state.read_bytes()
            sidecar.write_text('{"changed":true}')
            legend = root / 'new.md'
            legend.write_text('Candidate legend')
            with self.assertRaisesRegex(ValueError, 'Registry hash mismatch'):
                publish_presentation(state, ticket, candidate_legend=legend)
            self.assertEqual(state.read_bytes(), before)

    def test_failed_state_commit_rolls_back_selected_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ticket, scope = self.selected(root)
            old = {p: p.read_bytes() for p in [state, *[Path(r['path']) for r in scope.values()]]}
            legend = root / 'new.md'
            legend.write_text('Candidate legend')
            replace = os.replace
            def fail_commit(source, destination):
                if Path(destination).resolve() == state.resolve():
                    raise OSError('synthetic state commit failure')
                return replace(source, destination)
            with patch('classical_conditioning.figure_workspace.os.replace', side_effect=fail_commit):
                with self.assertRaisesRegex(OSError, 'state commit failure'):
                    publish_presentation(state, ticket, candidate_legend=legend)
            self.assertTrue(all(p.read_bytes() == data for p, data in old.items()))
            self.assertFalse(Path(str(state)+'.lock').exists())

    def test_legend_cli_never_imports_numerical_reconstruction(self):
        from scripts.paper.shared import build_legend_review as command
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ticket, scope = self.selected(root)
            legend = root / 'new.md'
            legend.write_text('New explanatory legend')
            argv = ['build_legend_review.py', '--state', str(state), '--baseline', str(ticket),
                    '--candidate-legend', str(legend), '--update-current']
            output = io.StringIO()
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules,
                    {'classical_conditioning.figures.aligned_example_pair': None}), redirect_stdout(output):
                command.main()
            self.assertFalse(json.loads(output.getvalue())['scientific_reconstruction'])
            self.assertEqual(Path(scope['working_svg']['path']).read_bytes(), artwork())

    def test_wrong_reconstruction_scope_rejected_before_numerical_import(self):
        from scripts.paper.shared import build_legend_review as command
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = ['build_legend_review.py', '--state', str(root/'figure-02.json'), '--output', str(root/'out.html')]
            numerical = 'classical_conditioning.figures.aligned_example_pair'
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {numerical: None}), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    command.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root/'out.html').exists())


if __name__ == '__main__':
    unittest.main()
