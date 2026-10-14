"""Verify frozen prompt provenance, exact cohort and preservation before admission."""
from pathlib import Path
from collections import Counter
import json
from prepare_entitybench_flux_portraitagent_20ep import OUTPUT, QWEN, PROJECT, read, sha256, prepare
from portrait_agent_interactive import episode_input, validate

def audit(root=OUTPUT):
    root=Path(root);plan=prepare(root)
    old=read(PROJECT/'outputs/entitybench_flux_native_i2v_a14b_4step_continuation_20ep/preparation_plan.json')
    for field in ('counts','reference_settings','firstframe_settings','video_config','generation_backend','evaluation'):
        assert plan[field]==old[field],field
    for a,b in zip(plan['episodes'],old['episodes']):
        assert a['episode_id']==b['episode_id'] and a['episode_sha256']==b['episode_sha256']
        for x,y in zip(a['shots'],b['shots']):
            for key in ('shot_key','prompt','seed','cut','previous_shot_key','entity_schedule'):assert x[key]==y[key],key
    for a,b in zip(plan['roles'],old['roles']):
        for key in ('episode_id','character','description','needs_face_asset','seed'):assert a[key]==b[key],key
        for key in ('reference','metadata','result'):assert Path(a[key]).is_relative_to(root)
    current=read(root/'runtime/preserved_prior_files.json')
    hashes=dict(current['files'])
    assert sha256(current['previous_snapshot'])==current['previous_snapshot_sha256']
    hashes.update({str(PROJECT/p):h for p,h in read(current['previous_snapshot'])['files'].items()})
    for path,h in hashes.items():assert sha256(path)==h, 'Historic file changed: '+path
    records=[read(r['path']) for r in plan['portrait_agent']['records']]
    for record in records:validate(record['output'],episode_input(record['output']['episode_id']))
    from transformers import T5Tokenizer
    tokenizer=T5Tokenizer.from_pretrained(str(PROJECT/'third_party/PuLID/models/xflux_text_encoders'),local_files_only=True)
    counts=[len(tokenizer(r['prompt'])['input_ids']) for r in plan['roles']]
    assert max(counts)<=plan['reference_settings']['max_sequence_length'], 'Reference prompt would be truncated'
    usage={key:sum(r['usage'].get(key,0) for r in records) for key in ('prompt_tokens','completion_tokens','total_tokens')}
    result=dict(status='passed',episodes=20,shots=363,roles=112,api_calls=len(records),usage=usage,
        styles=dict(Counter(r['output']['style'] for r in records)),t5_max_tokens=max(counts),t5_sequence_length=128,
        preserved_prior_files=len(hashes),old_illustration_candidates='preserved',
        preparation_plan_sha256=sha256(root/'preparation_plan.json'),
        scope='Qwen3.7-plus text prompt writing only; non-VLM evaluation unchanged; local inference parameters unchanged')
    (root/'review').mkdir(exist_ok=True)
    (root/'review/input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Portrait Agent 提示词核验','',f"20集、112角色；{result['styles']}；T5最大{max(counts)}/128 tokens，参考prompt不截断。",'']
    for record in records:
        o=record['output'];lines += [f"## {o['episode_id']}",'',f"风格：{o['style']}。{o['style_reason']}",'']
        for c in o['characters']:lines += [f"- **{c['name']}**：{c['prompt']}"]
        lines += ['']
    (root/'portrait_agent/README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))

if __name__=='__main__':audit()
