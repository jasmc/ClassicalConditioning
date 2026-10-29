"""Phase guides preserve marks and use the observed trial coordinates."""
import copy
import unittest
import xml.etree.ElementTree as ET
from tests.test_figure2_statistics_registration import artwork
from scripts.paper.figure_02.assembly.stats_phase_guides import restore_guides,protect,guide_check,find,tag,STATS,LABELS

class PhaseGuideTests(unittest.TestCase):
    def source(self):
        r=artwork();g=find(r,STATS['G'])
        for idx,key in enumerate(LABELS):find(r,key).set('y',str(10+idx*4))
        for idx,trial in enumerate((14.5,64.5)):
            owner=ET.SubElement(g,tag('g'),{'id':f'edit-G-fig2__g__stats__boundary-{trial}'})
            ET.SubElement(owner,tag('path'),{'id':'edit-G-node-'+str(22+idx),'d':f'M {trial} 12 L {trial} 20','clip-path':'url(#old)','style':'fill:none;stroke:black;stroke-dasharray:1,2;vector-effect:non-scaling-stroke'})
        for panel in 'HI':find(r,STATS[panel]).set('transform','translate(3 4) scale(.7 .8)')
        return ET.tostring(r)
    def test_all_guides_follow_trial_map_and_cover_five_rows(self):
        source=self.source();raw,edits,geom=restore_guides(source);checks=guide_check(raw,edits,geom)
        self.assertEqual(len(checks),6);self.assertEqual(geom['y_extent_mm'],[8,27])
        self.assertTrue(protect(source,raw,edits)['all_other_artwork_exactly_preserved'])
    def test_statistical_mark_edits_are_rejected(self):
        source=self.source();raw,edits,_=restore_guides(source);r=ET.fromstring(raw);find(r,'result-H').text='changed'
        with self.assertRaisesRegex(ValueError,'outside declared'):protect(source,ET.tostring(r),edits)
    def test_displaced_boundary_is_rejected(self):
        raw,edits,geom=restore_guides(self.source());r=ET.fromstring(raw);find(r,'fig2-H-stats-phase-boundary-14.5').set('transform','translate(1 0)')
        with self.assertRaisesRegex(ValueError,'registration'):guide_check(ET.tostring(r),edits,geom)

if __name__=='__main__':unittest.main()
