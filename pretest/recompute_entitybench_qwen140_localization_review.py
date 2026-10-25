from pathlib import Path
import importlib.util,json,math,hashlib,collections,statistics,sys
import numpy as np
ROOT=Path('/root/autodl-tmp/movie');RUN=ROOT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_full140';BASE=RUN/'evaluation/character_region_visible_face_gate_20261025';D=BASE/'remaining_low_score_review';OUT=D/'recomputed';OUT.mkdir(exist_ok=True)
def read(p):return json.load(open(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('region_gate_math',ROOT/'pretest/evaluate_entitybench_qwen140_region_face_gate.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);mod=module.load_official_pure()
review=read(D/'review_decisions.json');review_hash=sha(D/'review_decisions.json');assert review['review_manifest_sha256']==sha(D/'review_manifest.json');assert len(review['decisions'])==80
reject={(r['episode_id'],r['character'],r['shot']):r for r in review['decisions'] if r['decision']=='exclude'};assert len(reject)==3
manifest=read(RUN/'manifest.json');gate=read(BASE/'gate_decisions.json');bykey={(r['episode_id'],r['character'],r['shot']):r for r in gate['rows']};base_report=read(BASE/'report.json')
assert sha(BASE/'report.json')==read(D/'review_plan.json')['base_report_sha256'];assert sha(BASE/'gate_decisions.json')==read(D/'review_plan.json')['base_gate_sha256']
for k,r in reject.items():assert bykey[k]['kept'] and sha(bykey[k]['original_crop'])==r['original_crop_sha256']
source_hashes={str(BASE/'report.json'):sha(BASE/'report.json'),str(BASE/'gate_decisions.json'):sha(BASE/'gate_decisions.json'),str(D/'review_decisions.json'):review_hash,str(D/'review_manifest.json'):sha(D/'review_manifest.json')}
for src in read(BASE/'plan.json')['inputs']:assert sha(src['path'])==src['sha256']
new_results=[];before_results=[];fixed_results=[];diagnostics=[];max_error=0.;removed_scored=[]
for ep in manifest['episodes']:
 eid=ep['episode_id'];p=RUN/'evaluation/official_parallel_v2'/f'{eid}.json';raw=read(p);order=[s['shot_key'] for s in ep['shots']];pos={s:i for i,s in enumerate(order)};full_pools={};subset_pools={}
 for name,data in raw['per_entity_scores'].items():
  if not name.startswith('face_'):continue
  name=name[5:];apps=sorted(data['per_shot_sim_to_centroid'],key=lambda a:pos[a['shot']]);pairs=[p for p in raw['gap_decay_data'] if p['entity_type']=='character' and p['entity_name']==name];n=len(apps);assert len(pairs)==n*(n-1)//2;g=np.eye(n);k=0
  for i in range(n):
   for j in range(i+1,n):
    assert pairs[k]['gap']==abs(pos[apps[j]['shot']]-pos[apps[i]['shot']]);g[i,j]=g[j,i]=pairs[k]['similarity'];k+=1
  features=module.gram_to_features(g);before=[];after=[]
  for app,v in zip(apps,features):
   key=(eid,name,app['shot']);r=bykey[key]
   if not r['kept']:continue
   c=mod.EntityCrop(name,'character',name,app['shot'],r['sample_idx'],tuple(r['region']),r['original_crop'],app['clip_text_sim'],v);before.append(c)
   if key not in reject:after.append(c)
   else:removed_scored.append(key)
  full_pools[name]=before;subset_pools[name]=after
 prev,_,pn=mod.compute_cross_shot_consistency(full_pools,'character',shot_order=order);stored=read(BASE/'results'/f'{eid}.json')['metrics']['cs_face'];assert (prev is None)==(stored['value'] is None)
 if prev is not None:max_error=max(max_error,abs(prev-stored['value']));assert abs(prev-stored['value'])<1e-10
 assert pn==stored['n_evaluated']
 val,entities,n=mod.compute_cross_shot_consistency(subset_pools,'character',shot_order=order)
 independent=[];old_sims=[]
 for name,crops in subset_pools.items():
  if len(crops)<2:continue
  gram=np.array([[np.dot(a.dino_embedding,b.dino_embedding) for b in crops] for a in crops]);independent.extend((gram.sum(1)/np.sqrt(gram.sum())).tolist())
  old_map={a['shot']:a['sim_to_centroid'] for a in read(BASE/'results'/f'{eid}.json')['per_entity_scores'][name]['per_shot_sim_to_centroid']}
  old_sims.extend(old_map[c.shot_key] for c in crops)
 assert len(independent)==n
 if independent:assert abs(val-statistics.mean(independent))<1e-10
 out=mod.EpisodeResult(eid,ep['difficulty'],len(order));out.metrics['cs_face']=mod.MetricValue(value=val,n_evaluated=n);new_results.append(out)
 old=mod.EpisodeResult(eid,ep['difficulty'],len(order));old.metrics['cs_face']=mod.MetricValue(value=prev,n_evaluated=pn);before_results.append(old)
 fixed=mod.EpisodeResult(eid,ep['difficulty'],len(order));fixed.metrics['cs_face']=mod.MetricValue(value=statistics.mean(old_sims) if old_sims else None,n_evaluated=len(old_sims));fixed_results.append(fixed)
 write(OUT/'results'/f'{eid}.json',dict(episode_id=eid,tier=ep['difficulty'],num_shots=len(order),metrics={'cs_face':out.metrics['cs_face'].to_json()},per_entity_scores=entities))
 diagnostics.append(dict(episode_id=eid,before=prev,after=val,before_appearances=pn,after_appearances=n))
split={tier:[e['episode_id'] for e in manifest['episodes'] if e['difficulty']==tier] for tier in ['easy','medium','hard']};before=mod.aggregate_results(before_results,split);after=mod.aggregate_results(new_results,split);fixed=mod.aggregate_results(fixed_results,split)
assert before['all']['cs_face']==base_report['visible_face_gate']['all']['cs_face'];assert len(removed_scored)==3
nnew=sum(e.metrics['cs_face'].n_evaluated for e in new_results);coverage={'scheduled_role_shots':4991,'before_gate_crops':1700,'after_gate_crops':1697,'after_gate_crop_coverage':1697/4991,'before_scored_appearances':1383,'after_scored_appearances':nnew,'before_scored_episodes':131,'after_scored_episodes':after['all']['cs_face']['n_episodes_evaluated']}
report={'status':'complete','scope':'post-hoc targeted localization review diagnostic; not official scores','reviewed_items':80,'selection':review['criteria']['selection'],'excluded_confirmed_localization_errors':list(reject.values()),'before':before,'after':after,'survivors_against_previous_gate_centroids':fixed,'coverage':coverage,'per_episode':diagnostics,'inputs_sha256':source_hashes,'review_decisions_sha256':review_hash,'max_before_reproduction_error':max_error,'vlm_calls':0,'new_model_inference_calls':0,'limitations':'Only the lowest60 plus20 random other crops were audited, with10 full-frame checks; remaining population is not certified. Differences from references alone were retained as generated failure or unresolved association. No score-based exclusion threshold.'}
write(OUT/'report.json',report)
for p,h in source_hashes.items():assert sha(Path(p))==h
print(json.dumps({'before':before['all']['cs_face'],'after':after['all']['cs_face'],'coverage':coverage,'previous_centroid_same_survivors':fixed['all']['cs_face'],'baseline_reproduction_error':max_error},ensure_ascii=False,indent=2))
(D/'README.md').write_text(f'''# 上一版保留样本的低分复查

本轮由助手逐图复查原可见单脸门控版的最低60项，加其余样本中固定随机种子抽取的20项。混排的四页图不显示分数；其中10项结合精确采样视频帧和镜头prompt核对。此抽样用于排查，不代表全量错误率。

## 确认仍有三项定位错误

| ID | 角色／镜头 | 裁剪问题 |
|---|---|---|
| R009 | run227 Walter 3:6 | 应是左侧灰胡子西装男性，选中右侧年轻男性 |
| R028 | run4 Ji-hoon 2:1 | 应是后景穿黑色hoodie的年轻人，选中前景戴眼镜的Seo-joon |
| R040 | run227 Walter 3:1 | 左侧目标在场，却选中右侧年轻男性 |

目标人物都可以在同一采样画面找到，所以按定位错误排除这三项。决策先写入[review_decisions.json](review_decisions.json)并冻结，再重算；没有按DINO分数设阈值或调整规则。

## 重算结果

| 独立诊断版本 | 分数 | 可评分episode | 参与评分的角色—镜头数 |
|---|---:|---:|---:|
| 原可见单脸门控版 | {before['all']['cs_face']['mean']:.4f} | 131 | 1383 |
| 排除三项确认定位错误 | {after['all']['cs_face']['mean']:.4f} | {coverage['after_scored_episodes']} | {nnew} |

门控保留裁剪从1700变为1697，计划集合覆盖率为{1697/4991:.2%}。若某角色只剩一个镜头，则该角色也不再参与跨镜头质心评分，因此评分appearance数量变化不一定等于排除条数。

依旧使用原人物区域像素、保存的DINO两两相似度和冻结官方质心／聚合函数。独立Gram行和公式复核；上版分数重现最大误差{max_error:.3g}。原官方0.7155、自定义脸部0.8237及上一版0.7860均未覆盖或修改。

## 保留的问题样本

- Min-jun单角色镜头生成红发女性、Isabel单角色镜头生成年长男性：不能仅因不像参考就归为定位错误，这属于生成／身份失败或尚未解决的歧义，继续保留。
- Frank、Kenji等样本存在卡通／写实风格或外貌改变，仍保留。
- 侧脸、全身远景、暗光、坐姿／躺姿、遮挡以及服装变化，仍保留。
- 多人案例不能确定是角色缺失、生成漂移还是找错人时，不做排除。

本轮80项中只排除3项，保留77项。这并非全量通用定位规则的验证，不能作为新的官方benchmark结果，也不意味着其余样本均正确。下一步应优先修复人物归属和候选框，而非持续删除低分。

## 证据

- [四页混排复查图1](review_page_1.jpg)／[2](review_page_2.jpg)／[3](review_page_3.jpg)／[4](review_page_4.jpg)
- [R009精确采样帧与原框](R009_context_marked.jpg)
- [完整画面复查1](context_page_1.jpg)／[2（包含R028、R040）](context_page_2.jpg)／[3](context_page_3.jpg)
- [逐项决策](review_decisions.json)、[协议](review_plan.json)、[重算报告](recomputed/report.json)
''')
