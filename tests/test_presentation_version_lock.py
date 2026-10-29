"""Provisional locks preserve bytes and never publish past a changed source."""
import json,tempfile,unittest
from pathlib import Path
from scripts.paper.shared.lock_versions import prepare,publish,sha,packed
class VersionLockTests(unittest.TestCase):
 def fixture(self,base):
  source=base/'source';source.mkdir();versions=base/'versions';versions.mkdir();other=versions/'exploratory';other.mkdir();(other/'keep.svg').write_bytes(b'other exploratory bytes')
  state=source/'current-state.json';state.write_bytes(packed({'figure_id':'fig1','scientific_selection':{'records':[]},'draft':{'semantic_registry':{'whole_figure_complete':False}}}));files={'source-state.json':str(state)}
  for name,data in [('figure-01.pdf',b'unchanged pdf'),('figure-01.html',b'unchanged html'),('figure-01.svg',b'unchanged svg')]:p=source/name;p.write_bytes(data);files[name]=str(p)
  ticket=base/'ticket.json';ticket.write_bytes(packed({'kind':'provisional-presentation-lock','date':'2026-10-29','authorization':'Test exact version lock','starting_hashes':{p:sha(Path(p).read_bytes())for p in files.values()},'figures':[{'figure':1,'state_path':str(state),'state_exists':True,'files':files,'target_dir':str(versions/'locked'),'scope':'Test','retention':'Keep all'}]}));return ticket,state,other
 def staged(self,base):
  ticket,state,other=self.fixture(base);plan=prepare(ticket,base/'stage');plan['lock']=str(base/'publication');p=base/'plan.json';p.write_bytes(packed(plan));return p,state,other
 def test_preserves_artwork_science_and_exploratory_versions(self):
  with tempfile.TemporaryDirectory()as temp:
   base=Path(temp);plan,state,other=self.staged(base);before=json.loads(state.read_bytes());r=publish(plan);after=json.loads(state.read_bytes());self.assertEqual(r['status'],'locked');self.assertEqual(after['scientific_selection'],before['scientific_selection']);self.assertFalse(r['publication_freeze_completed']);self.assertEqual((base/'versions/locked/figure-01.pdf').read_bytes(),b'unchanged pdf');self.assertEqual((other/'keep.svg').read_bytes(),b'other exploratory bytes')
 def test_source_change_stops_publication(self):
  with tempfile.TemporaryDirectory()as temp:
   base=Path(temp);plan,state,_=self.staged(base);(base/'source/figure-01.svg').write_bytes(b'concurrent manual edit')
   with self.assertRaisesRegex(ValueError,'advanced'):publish(plan)
   self.assertNotIn('presentation_lock',json.loads(state.read_bytes()));self.assertFalse((base/'versions/locked/figure-01.pdf').exists())
 def test_refuses_to_replace_locked_artifacts(self):
  with tempfile.TemporaryDirectory()as temp:
   base=Path(temp);ticket,_,_=self.fixture(base);dest=base/'versions/locked';dest.mkdir();(dest/'keep').write_bytes(b'prior lock')
   with self.assertRaises(FileExistsError):prepare(ticket,base/'stage')
   self.assertEqual((dest/'keep').read_bytes(),b'prior lock')
if __name__=='__main__':unittest.main()
