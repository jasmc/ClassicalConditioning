"""Trial registration must survive label compaction, never whole-table scaling."""
import copy
import unittest
import xml.etree.ElementTree as ET
from scripts.paper.figure_02.assembly.align_statistics import (
    STATS, OBSERVED, PHASES, LABELS, find, registration, check_payload, tag,
)


def artwork():
    root=ET.Element(tag('svg'),{'width':'600mm','height':'100mm','viewBox':'0 0 600 100'})
    for column,panel in enumerate(STATS):
        owner=ET.SubElement(root,tag('g'),{'id':'panel-'+panel,'transform':f'translate({column*200} 0)'})
        observed=ET.SubElement(owner,tag('g'),{'id':OBSERVED[panel]})
        ET.SubElement(observed,tag('path'),{'id':'data-'+panel,'d':'M 5 20 L 94 30'})
        for value in (5,94):ET.SubElement(observed,tag('text'),{'x':str(value),'y':'40'}).text=str(value)
        stats=ET.SubElement(owner,tag('g'),{'id':STATS[panel]})
        for i,phase in enumerate(PHASES):
            if panel=='G':
                node=ET.SubElement(stats,tag('text'),{'id':'edit-G-node-'+str(53+2*i),'x':str(9.5+10*i),'y':'5'})
            else:
                prefix='matched-H__H-'if panel=='H'else'trace10-I__I-'
                group=ET.SubElement(stats,tag('g'),{'id':prefix+'phase-header-'+phase})
                node=ET.SubElement(group,tag('text'),{'x':str(9.5+10*i),'y':'5'})
            node.text=phase
        ET.SubElement(stats,tag('text'),{'id':'result-'+panel,'x':'25','y':'10'}).text='***'
        if panel=='G':
            for identity,(before,_)in LABELS.items():
                ET.SubElement(stats,tag('text'),{'id':identity,'x':'0','y':'15'}).text=before
    ET.SubElement(root,tag('metadata'),{'id':'layout-frames'}).text='{}'
    return root


class RegistrationTests(unittest.TestCase):
    def test_phase_headers_share_the_observed_trial_mapping(self):
        self.assertLess(max(r['max_error_mm']for r in registration(artwork()).values()),1e-10)

    def test_whole_table_shrink_is_detected(self):
        root=artwork();find(root,STATS['G']).set('transform','scale(0.8333333333)')
        self.assertGreater(registration(root)['G']['max_error_mm'],10)

    def test_glyph_resize_about_its_anchor_preserves_trial_registration(self):
        root=artwork()
        for node in find(root,STATS['G']).iter(tag('text')):
            x=node.get('x');node.set('transform',f'translate({x} 0) scale(.8 1) translate(-{float(x)} 0)')
        self.assertLess(registration(root)['G']['max_error_mm'],1e-10)

    def prepared_aliases(self):
        original=artwork();candidate=copy.deepcopy(original);edits={}
        for identity,(_,after)in LABELS.items():find(candidate,identity).text=after
        for panel,identity in STATS.items():
            node=find(candidate,identity)
            edits[panel]={'after_transform':node.get('transform'),
                          'text_transforms':[{'after':n.get('transform')}for n in node.iter(tag('text'))]}
        return ET.tostring(original),candidate,edits

    def test_only_declared_row_label_aliases_are_permitted(self):
        original,candidate,edits=self.prepared_aliases()
        result=check_payload(original,ET.tostring(candidate),original,edits)
        self.assertTrue(result['original_statistical_values_and_raw_mark_coordinates_preserved'])

    def test_significance_result_changes_are_rejected(self):
        original,candidate,edits=self.prepared_aliases();find(candidate,'result-G').text='**'
        with self.assertRaisesRegex(ValueError,'statistical values'):
            check_payload(original,ET.tostring(candidate),original,edits)

    def test_outcome_curve_changes_are_rejected(self):
        original,candidate,edits=self.prepared_aliases();find(candidate,'data-G').set('d','M 5 20 L 94 31')
        with self.assertRaisesRegex(ValueError,'outside statistical'):
            check_payload(original,ET.tostring(candidate),original,edits)


if __name__=='__main__':unittest.main()
