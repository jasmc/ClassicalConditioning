"""Scientific payload protection and physical-gap evidence regressions."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publish_axis_geometry import verify_protection,verify_metrics
from classical_conditioning.figures.layout_frames import embed_spec
from classical_conditioning.figure_workspace import sha


class AxisGeometryTests(unittest.TestCase):
    def fixture(self):
        raw=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 183 60"><g id="owner"><path id="tick" d="M1 2L1 4"/><path id="data" d="M1 2L3 4"/></g></svg>'
        return embed_spec(raw,{'scope':'panel','final_size_mm':[183,60],'inset_mm':1.,'allocations':[{'id':'A','kind':'panel','box_mm':[0,0,183,60],'group_ids':['owner']}],'background_ids':[]})

    def test_registered_axis_style_does_not_authorize_data_or_tick_coordinate_edits(self):
        raw=self.fixture();manifest={'changes':[{'xml_index_path':[0,0],'roles':['axis.tick']}]}
        # Layout metadata is inserted after existing groups.
        candidate=raw.replace(b'id="tick"',b'id="tick" style="stroke-width:1" transform="scale(1 2)"')
        verify_protection(raw,candidate,manifest)
        with self.assertRaises(ValueError):verify_protection(raw,candidate.replace(b'M1 2L3 4',b'M1 2L3 5'),manifest)
        with self.assertRaises(ValueError):verify_protection(raw,candidate.replace(b'M1 2L1 4',b'M1 2L1 5'),manifest)

    def test_physical_gap_failure_and_stale_svg_are_rejected(self):
        report={'artifact_sha256':sha(b'svg'),'axes':[{'id':'x','ticks':[],'tick_labels':[],'tick_label_gaps_mm':[3*25.4/72],'title_gap_mm':3*25.4/72}]}
        verify_metrics(report,b'svg')
        with self.assertRaises(ValueError):verify_metrics(report,b'new-svg')
        report['axes'][0]['tick_label_gaps_mm']=[.4]
        with self.assertRaises(ValueError):verify_metrics(report,b'svg')

if __name__=='__main__':unittest.main()
