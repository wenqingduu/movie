"""Regression checks for the actual identifier and duplicate-submission failures."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import qwen140_technical_resume_20261103 as resume

TOOL={'function':{'description':'lookup','parameters':{'properties':{
    'names':{'type':'array','items':{'type':'string'}}}}}}


class TechnicalResumeTests(unittest.TestCase):
    def test_character_tool_rejects_object_and_anonymous_ids(self):
        tool=resume.bounded_lookup_tool(TOOL,['Julian'])
        self.assertEqual(tool['function']['parameters']['properties']['names']['items']['enum'],['Julian'])
        self.assertNotIn('enum',TOOL['function']['parameters']['properties']['names']['items'])
        empty=resume.bounded_lookup_tool(TOOL,[])
        self.assertEqual(empty['function']['parameters']['properties']['names']['maxItems'],0)

    def test_empty_catalog_keeps_anonymous_action_and_allows_inherited_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'shot_2_7/agent';folder.mkdir(parents=True)
            parent=root/'shot_2_6/result.json';parent.parent.mkdir()
            parent.write_text(json.dumps({'route':{'frame_characters':['Liam']}}))
            ep={'shots':[{'shot_key':'2:6','entity_schedule':{'characters':['Liam']}},{'shot_key':'2:7'}]}
            observed={}
            def original(context,tail,catalog,path):
                observed.update(context=context,tool=copy.deepcopy(agent.LOOKUP_TOOL))
                return {'decision':{'frame_characters':[]}}
            agent=SimpleNamespace(LOOKUP_TOOL=copy.deepcopy(TOOL),plan_shot=original)
            core=SimpleNamespace(TAGS={'run4':'i80_j98'},descriptions=lambda ep:{'Liam':'Man in white shirt.'})
            episode=SimpleNamespace(load=lambda *args:(ep,None,{}),core=core)
            resume.install_planner_adapter(episode,agent)
            original_context={'shot_key':'2:7','current_shot':'An unnamed man writhes on the wet pavement.'}
            episode.plan_shot(original_context,root/'tail.png',{},folder)
            self.assertEqual(observed['context']['valid_lookup_character_names'],[])
            self.assertEqual(observed['context']['valid_frame_character_ids'],['Liam'])
            self.assertEqual(observed['context']['current_shot'],original_context['current_shot'])
            self.assertNotIn('valid_frame_character_ids',original_context)
            self.assertEqual(agent.LOOKUP_TOOL,TOOL)

    def test_get_tls_resume_reuses_same_task_url(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts=[]
            def original(url,key,folder,payload=None):
                attempts.append((url,payload))
                if len(attempts)==1:raise urllib.error.URLError('SSL EOF')
                return {'output':{'task_id':'saved-task','task_status':'SUCCEEDED'}}
            wan=SimpleNamespace(api_request=original)
            resume.install_get_poll_resume(wan)
            with patch.object(resume.time,'sleep'):
                result=wan.api_request('https://provider.invalid/tasks/saved-task','test',Path(directory))
            self.assertEqual(result['output']['task_id'],'saved-task')
            self.assertEqual(attempts,[('https://provider.invalid/tasks/saved-task',None)]*2)

    def test_post_task_creation_is_never_retried(self):
        attempts=[]
        def original(*args):
            attempts.append(args)
            raise urllib.error.URLError('SSL EOF')
        wan=SimpleNamespace(api_request=original)
        resume.install_get_poll_resume(wan)
        with self.assertRaises(urllib.error.URLError):
            wan.api_request('https://provider.invalid/create','test',Path('/tmp'),{'model':'wan3.0-video'})
        self.assertEqual(len(attempts),1)


if __name__=='__main__':unittest.main()
