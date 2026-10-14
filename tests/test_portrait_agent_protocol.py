"""Semantic contracts for the text Agent before any new GPU inference."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretest'))
from portrait_agent_interactive import validate, episode_input
from prepare_entitybench_flux_portraitagent_20ep import assert_new_root, PROJECT

class PortraitAgentContracts(unittest.TestCase):
    def setUp(self):
        self.inputs=dict(episode_id='one',characters=[dict(name='Leo',description='Young boy wearing a yellow jacket over a red polo shirt.')])
        self.output=dict(episode_id='one',style='photorealistic',characters=[dict(name='Leo',description=self.inputs['characters'][0]['description'],
            prompt='Photorealistic live-action portrait. '+self.inputs['characters'][0]['description']+' Frontal face, plain gray background.')])

    def test_appearance_and_child_preserved(self):
        validate(self.output,self.inputs)
        changed=copy.deepcopy(self.output);changed['characters'][0]['description']='Adult man.'
        with self.assertRaises(AssertionError):validate(changed,self.inputs)

    def test_prompt_cannot_omit_original_description_or_style(self):
        for prompt in ('Photorealistic adult man, gray background.',self.inputs['characters'][0]['description']+' Gray background portrait.'):
            changed=copy.deepcopy(self.output);changed['characters'][0]['prompt']=prompt
            with self.assertRaises(AssertionError):validate(changed,self.inputs)

    def test_no_missing_duplicate_or_foreign_characters(self):
        for characters in ([],self.output['characters']*2,[dict(self.output['characters'][0],name='foreign')]):
            changed=dict(self.output,characters=characters)
            with self.assertRaises(AssertionError):validate(changed,self.inputs)

    def test_rejected_illustration_run_cannot_be_overwritten(self):
        old=PROJECT/'outputs/entitybench_flux_native_i2v_a14b_4step_continuation_20ep'
        for root in (old,old/'child'):
            with self.assertRaises(AssertionError):assert_new_root(root)

    def test_inputs_preserve_explicit_animation_and_child_description(self):
        animation=episode_input('00007176-a5da-371b-baef-7a8f690f5a38__run53__i50_j59__T120')
        self.assertTrue(any('animated medium shot' in s['prompt'] for s in animation['shots']))
        child=episode_input('0003c349-4d2c-3a1b-a0a0-cf3596bb3dfc__run484__i15_j47__T420')
        self.assertTrue(next(c for c in child['characters'] if c['name']=='Leo')['description'].startswith('Young boy'))

if __name__=='__main__':unittest.main()
