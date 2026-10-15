import unittest
from classical_conditioning.analysis.inference.panel_g_technical import protocol_review_flag


class TechnicalReviewTests(unittest.TestCase):
    def test_boundary_review_alone_does_not_exclude_a_fish(self):
        a={'alignment':{'protocol_events_outside_acquisition':False},
           'protocol':{'statistics':{'invalid_duration_count':0}},
           'status':'REVIEW','warnings':['camera/tracking frame ranges differ at their boundaries']}
        self.assertFalse(protocol_review_flag(a))
        a['protocol']['statistics']['invalid_duration_count']=1
        self.assertTrue(protocol_review_flag(a))
        a['protocol']['statistics']['invalid_duration_count']=0
        a['alignment']['protocol_events_outside_acquisition']=True
        self.assertTrue(protocol_review_flag(a))


if __name__=='__main__':unittest.main()
