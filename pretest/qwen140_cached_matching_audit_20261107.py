"""Cache-only complete-cohort matching inventory and blinded crop review.

No model loading, paid APIs, media generation or automatic score repair. Original
official scores are immutable. Any extended localization diagnostic is separate
from the historical sampled-review result and applies to both editions.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import textwrap

from PIL import Image,ImageDraw,ImageFont,ImageOps
import qwen140_full140_latest_20261106 as f

OUT=f.ROOT/'outputs/entitybench_cached_matching_audit_20261107'
read,write,sha=f.read,f.write,f.sha
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'inventory.json').exists():return read(OUT/'inventory.json')
    snap=read(f.OUT/'snapshot.json');old,_=f.diag.historical();_,_,cur,_,_=f.paired.load_maps()
    old={f.diag.key(r):r for r in old}
    for item in snap['new_complete_episodes']:
        for p in (f.REGION/'extraction'/item['episode_id']).glob('*.json'):
            for r in read(p)['rows']:cur[f.diag.key(r)]=r
    groups={};counts=defaultdict(Counter);errors=[];sources=[]
    original_jobs={(r['episode_id'],r['shot_key']):r for r in read(f.BASE/'generation_report.json')['jobs']}
    for item in snap['episodes']:
        ep=item['episode'];eid=ep['episode_id'];descs={v['name']:v['description'] for v in ep['entity_descriptions'].values()}
        for edition,result,gate in [('original',item['original_result'],old),('current',item['result'],cur)]:
            raw=read(result);assert raw['num_shots']==len(ep['shots'])
            sources.append(dict(edition=edition,episode_id=eid,path=result,sha256=sha(result)))
            art=Path(result).parent/'.work/eval_artifacts'/eid/'intra_shot'
            for shot in ep['shots']:
                sk=shot['shot_key'];evfile=art/sk.replace(':','_')/'_eval.json';ev=read(evfile)
                source=(str(f.BASE/'videos'/eid/('shot_'+sk.replace(':','_'))/'video.mp4') if edition=='original'
                        else snap['media'][eid+'/'+sk]['source'])
                video_sha=original_jobs[eid,sk]['video_sha256'] if edition=='original' else snap['media'][eid+'/'+sk]['sha256']
                for kind,schedule in [('character','characters'),('object','objects'),('location','places')]:
                    value=ev['presence'][kind];assert {r['name'] for r in value['items']}==set(shot['entity_schedule'].get(schedule,[]))
                    for r in value['items']:
                        counts[edition+'/'+kind][r['status']]+=1
                        crop=r.get('canonical_path');digest=sha(crop) if crop and Path(crop).exists() else None
                        if crop and not digest:errors.append(dict(issue='missing_canonical_file',path=crop))
                        if crop and (r['clip_sim'] is None or not 0<=r['clip_sim']<=1):errors.append(dict(issue='nonfinite_or_invalid_clip',item=r))
                        expected='absent' if not crop else 'weak' if r['clip_sim']<.20 else 'present'
                        if r['status']!=expected:errors.append(dict(issue='status_threshold_mismatch',item=r))
                        k=(eid,sk,kind,r['name'],digest,r.get('frame_idx'),video_sha)
                        if k not in groups:
                            groups[k]=dict(episode_id=eid,run=ep['run'],window=eid.split('__')[2],shot=sk,kind=kind,
                                name=r['name'],description=descs.get(r['name'],r['name']),prompt=shot['prompt'],crop=crop,crop_sha256=digest,
                                sample_idx=r.get('frame_idx'),video_sha256=video_sha,references=[],gate_reason=gate.get((eid,r['name'],sk),{}).get('reason') if kind=='character' else None)
                        groups[k]['references'].append(dict(edition=edition,status=r['status'],clip_sim=r['clip_sim'],
                            video=source,video_sha256=video_sha,evaluation=str(evfile),evaluation_sha256=sha(evfile)))
    rows=sorted(groups.values(),key=lambda r:(r['episode_id'],r['shot'],r['kind'],r['name'],r['crop_sha256'] or ''))
    for i,r in enumerate(rows,1):r['record_id']=f'E{i:05d}'
    data=dict(snapshot_sha256=sha(f.OUT/'snapshot.json'),episodes=140,shots=2491,counts={k:dict(v) for k,v in counts.items()},
        unique_records=len(rows),errors=errors,rows=rows,source_scores=sources,models_loaded=False,vlm_calls=0,
        scope='All scheduled character/object/location items, original and current frozen57+141 snapshot; dedupe exact role/shot/crop/sample shared inputs. Canonical hashes verified.')
    write(OUT/'inventory.json',data)
    for category in ['unique_face','weak','face_collision']:
        chosen=[r for r in rows if r['kind']=='character' and r['crop'] and
                (r['gate_reason']=='unique_visible_face' if category=='unique_face' else
                 any(x['status']=='weak' for x in r['references']) if category=='weak' else
                 r['gate_reason']=='same_face_track_claimed_by_multiple_roles')]
        random.Random(20261107).shuffle(chosen)
        for i,r in enumerate(chosen,1):r=dict(r);r['review_id']=f'{category[:1].upper()}{i:04d}';chosen[i-1]=r
        write(OUT/category/'manifest.json',dict(category=category,selection='Entire category, no DINO-score ranking; original/current labels and scores hidden on sheets.',rows=chosen))
    write(OUT/'status.json',dict(stage='inventory_complete_review_pending',unique_records=len(rows),technical_errors=len(errors),vlm_calls=0))
    return data


def sheets(category,start=1,end=None):
    prepare();manifest=read(OUT/category/'manifest.json');rows=manifest['rows'];n=(len(rows)+47)//48;end=min(end or n,n)
    for page in range(start,end+1):
        path=OUT/category/f'page_{page:03d}.jpg'
        if path.exists():continue
        canvas=Image.new('RGB',(1500,2240),'#efefef');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows[(page-1)*48:page*48]):
            x=(i%6)*250;y=(i//6)*280
            draw.text((x+4,y+2),f"{r['review_id']} {r['run']} {r['window']}",font=FONT,fill='black')
            draw.text((x+4,y+18),r['shot']+' '+r['name'],font=FONT,fill='black')
            im=Image.open(r['crop']).convert('RGB');im=ImageOps.contain(im,(242,155));canvas.paste(im,(x+4,y+38))
            for j,line in enumerate(textwrap.wrap(r['description'],width=32)[:5]):draw.text((x+4,y+198+15*j),line,font=FONT,fill='black')
        canvas.save(path,quality=93)
    print(category,len(rows),'rows',n,'pages','generated',start,end,flush=True)



def contexts():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    manifest=read(OUT/'unique_face/manifest.json');screen=read(OUT/'screening.json')
    ids=set(screen['categories']['unique_face']['candidate_review_ids'])
    chosen=[r for r in manifest['rows'] if r['review_id'] in ids]
    snap=read(f.OUT/'snapshot.json');episodes={r['episode_id']:r['episode'] for r in snap['episodes']}
    dest=OUT/'context';dest.mkdir(exist_ok=True)
    for n,r in enumerate(chosen):
        target=dest/(r['review_id']+'.jpg')
        if target.exists():continue
        frames,indexes=f.diag.decode(r['references'][0]['video']);frame=frames[r['sample_idx']]
        bbox=f.diag.faces.canonical_bbox(frame,r['crop']);draw=ImageDraw.Draw(im:=Image.fromarray(frame));draw.rectangle(tuple(bbox),outline='#ff3030',width=3)
        im.save(target,quality=94)
        write(dest/(r['review_id']+'.json'),dict(review_id=r['review_id'],video_sha256=r['video_sha256'],crop_sha256=r['crop_sha256'],frame_index=indexes[r['sample_idx']],bbox=bbox,actual_sample_indices=indexes))
        if n%20==0:print('context',n,'/',len(chosen),flush=True)
    write(dest/'manifest.json',dict(source_manifest_sha256=sha(OUT/'unique_face/manifest.json'),screening_sha256=sha(OUT/'screening.json'),rows=chosen))
    for page in range(1,(len(chosen)+5)//6+1):
        path=dest/f'page_{page:03d}.jpg';canvas=Image.new('RGB',(1800,2400),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(chosen[(page-1)*6:page*6]):
            x=i%2*900;y=i//2*800
            d.text((x+5,y+3),f"{r['review_id']} {r['run']} {r['window']} {r['shot']} TARGET: {r['name']}",font=FONT,fill='black')
            im=ImageOps.contain(Image.open(dest/(r['review_id']+'.jpg')),(700,400));canvas.paste(im,(x+5,y+25))
            im=ImageOps.contain(Image.open(r['crop']),(185,330));canvas.paste(im,(x+710,y+25))
            text='SHOT: '+r['prompt']+'\nTARGET: '+r['description']
            ep=episodes[r['episode_id']];sh=next(s for s in ep['shots'] if s['shot_key']==r['shot'])
            desc={e['name']:e['description'] for e in ep['entity_descriptions'].values()}
            text+='\nOTHER SCHEDULED: '+'; '.join(name+': '+desc.get(name,name) for name in sh['entity_schedule'].get('characters',[]) if name!=r['name'])
            lines=[]
            for line in text.splitlines():lines.extend(textwrap.wrap(line,112))
            for j,line in enumerate(lines[:23]):d.text((x+5,y+433+15*j),line,font=FONT,fill='black')
        canvas.save(path,quality=93)
    print('contexts complete',len(chosen),flush=True)



def context_sheets():
    manifest=read(OUT/'unique_face/manifest.json');screen=read(OUT/'screening.json')
    ids=set(screen['categories']['unique_face']['candidate_review_ids'])
    chosen=[r for r in manifest['rows'] if r['review_id'] in ids]
    snap=read(f.OUT/'snapshot.json');episodes={r['episode_id']:r['episode'] for r in snap['episodes']}
    dest=OUT/'context'
    for page in range(1,(len(chosen)+5)//6+1):
        if not all((dest/(r['review_id']+'.jpg')).exists() for r in chosen[(page-1)*6:page*6]):break
        path=dest/f'page_{page:03d}.jpg';canvas=Image.new('RGB',(1800,2400),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(chosen[(page-1)*6:page*6]):
            x=i%2*900;y=i//2*800
            d.text((x+5,y+3),f"{r['review_id']} {r['run']} {r['window']} {r['shot']} TARGET: {r['name']}",font=FONT,fill='black')
            im=ImageOps.contain(Image.open(dest/(r['review_id']+'.jpg')),(700,400));canvas.paste(im,(x+5,y+25))
            im=ImageOps.contain(Image.open(r['crop']),(185,330));canvas.paste(im,(x+710,y+25))
            text='SHOT: '+r['prompt']+'\nTARGET: '+r['description']
            ep=episodes[r['episode_id']];sh=next(s for s in ep['shots'] if s['shot_key']==r['shot'])
            desc={e['name']:e['description'] for e in ep['entity_descriptions'].values()}
            text+='\nOTHER SCHEDULED: '+'; '.join(name+': '+desc.get(name,name) for name in sh['entity_schedule'].get('characters',[]) if name!=r['name'])
            lines=[]
            for line in text.splitlines():lines.extend(textwrap.wrap(line,112))
            for j,line in enumerate(lines[:23]):d.text((x+5,y+433+15*j),line,font=FONT,fill='black')
        canvas.save(path,quality=93)
    print('partial context sheets ready',flush=True)


def freeze_review():
    manifest=read(OUT/'unique_face/manifest.json');notes=read(OUT/'context_review_notes.json');screen=read(OUT/'screening.json')
    assert screen['categories']['unique_face']['viewed_pages']==list(range(1,57))
    assert set(notes)==set(screen['categories']['unique_face']['candidate_review_ids'])
    decisions=[]
    for r in manifest['rows']:
        decision,evidence=notes.get(r['review_id'],['keep','No confirmed semantic localization error during score-hidden canonical-region screening. Not a full-frame confirmation for this noncandidate.'])
        if r['review_id'] in notes:
            c=read(OUT/'context'/(r['review_id']+'.json'))
            assert c['crop_sha256']==r['crop_sha256'] and c['video_sha256']==r['video_sha256']
        assert decision in ['keep','exclude']
        decisions.append(dict(review_id=r['review_id'],episode_id=r['episode_id'],character=r['name'],shot=r['shot'],
            original_crop_sha256=r['crop_sha256'],video_sha256=r['video_sha256'],decision=decision,
            reason='confirmed_wrong_localization_with_target_visible' if decision=='exclude' else 'no_confirmed_localization_error',
            evidence=evidence,review_stage='full_frame_context' if r['review_id'] in notes else 'canonical_screen',
            context=str(OUT/'context'/(r['review_id']+'.jpg')) if r['review_id'] in notes else None,
            editions=[v['edition'] for v in r['references']]))
    f.immutable(OUT/'decisions.json',dict(manifest_sha256=sha(OUT/'unique_face/manifest.json'),screening_sha256=sha(OUT/'screening.json'),
        notes_sha256=sha(OUT/'context_review_notes.json'),review_design='All 2673 unique eligible canonical regions, shuffled seed20261107, scores/edition hidden. 206 script/full-frame confirmations. Extended audit differs in review coverage from historical sampled protocol.',
        decisions=decisions))
    print('frozen',len(decisions),'new screen exclusions',sum(r['decision']=='exclude' for r in decisions),flush=True)


def rescore():
    import numpy as np
    p=read(f.OUT/'snapshot.json');old,oldreject=f.diag.historical();oldmap={f.diag.key(r):r for r in old}
    _,_,cur,rejected,_=f.paired.load_maps()
    for r in read(f.J.OUT/'visible_region/review/decisions.json')['decisions']:
        k=f.diag.key(r)
        if r['decision']=='exclude' and cur[k]['original_crop_sha256']==r['original_crop_sha256']:rejected[k]=r
    for item in p['new_complete_episodes']:
        for path in (f.REGION/'extraction'/item['episode_id']).glob('*.json'):
            for r in read(path)['rows']:cur[f.diag.key(r)]=r
    for r in read(f.REGION/'review/decisions.json')['decisions']:
        k=f.diag.key(r)
        if r['decision']=='exclude' and cur[k]['original_crop_sha256']==r['original_crop_sha256']:rejected[k]=r
    rejected={k:r for k,r in rejected.items() if cur[k]['original_crop_sha256']==r['original_crop_sha256']}
    review=read(OUT/'decisions.json');assert review['manifest_sha256']==sha(OUT/'unique_face/manifest.json')
    files=[f.diag.gate.OUT/'remaining_low_score_review/review_decisions.json']+[d/'review/decisions.json' for d in f.paired.review.OLD_REGIONS]+[
        f.J.PRIOR/'visible_region/review/decisions.json',f.J.OUT/'visible_region/review/decisions.json',f.REGION/'review/decisions.json']
    known=[]
    for path in files:
        for r in read(path)['decisions']:
            if r['decision']=='exclude':known.append({**r,'decision_source':str(path)})
    extra=[{**r,'decision_source':str(OUT/'decisions.json')} for r in review['decisions'] if r['decision']=='exclude']
    extended={}
    for edition,rows in [('original',oldmap),('current',cur)]:
        extended[edition]={}
        for r in known+extra:
            k=f.diag.key(r)
            if k in rows and rows[k]['original_crop_sha256']==r['original_crop_sha256']:
                extended[edition][k]=r
    # Equality of exact shared input gates is required for symmetric transfer.
    inv=read(OUT/'inventory.json')
    for r in inv['rows']:
        if r['kind']=='character' and len(r['references'])==2:
            k=(r['episode_id'],r['name'],r['shot'])
            assert oldmap[k]['kept']==cur[k]['kept'] and oldmap[k]['reason']==cur[k]['reason']
            assert (k in extended['original'])==(k in extended['current'])
    mod=f.diag.gate.load_official_pure();historical=f.read(f.REGION/'report.json');check={r['episode_id']:r for r in historical['per_episode']}
    groups={ed:[] for ed in ['original','current']};basegroups={ed:[] for ed in groups};details=[];changes=[];sources_verified=0
    for i,item in enumerate(p['episodes']):
        ep=item['episode'];eid=ep['episode_id'];per={}
        for edition,rows,oldrej,path in [('original',oldmap,oldreject,item['original_result']),('current',cur,rejected,item['result'])]:
            assert sha(path)==item['original_result_sha256' if edition=='original' else 'result_sha256']
            raw=read(path);a,_=f.diag.compute(ep,raw,rows,oldrej,mod)
            expected=check[eid][edition];actual=a.metrics['cs_face'].to_json()
            assert actual['n_evaluated']==expected['n_evaluated']
            assert (actual['value'] is None and expected['value'] is None) or abs(actual['value']-expected['value'])<1e-10
            b,entities=f.diag.compute(ep,raw,rows,extended[edition],mod)
            groups[edition].append(b);basegroups[edition].append(a);per[edition]=dict(historical=actual,extended=b.metrics['cs_face'].to_json())
            write(OUT/'recomputed'/edition/'results'/(eid+'.json'),dict(episode_id=eid,tier=ep['difficulty'],num_shots=len(ep['shots']),
                metrics={'cs_face':b.metrics['cs_face'].to_json()},per_entity_scores=entities))
            sources_verified+=1
        details.append(dict(episode_id=eid,**per))
        if i%25==0:print('rescore',i,'/140',flush=True)
    split=read(f.BASE/'split_full140.json');aggregate={k:mod.aggregate_results(v,split) for k,v in groups.items()}
    baseaggregate={k:mod.aggregate_results(v,split) for k,v in basegroups.items()}
    for edition in groups:
        for t in ['all','easy','medium','hard']:assert baseaggregate[edition][t]['cs_face']==historical[edition][t]['cs_face']
    coverage={}
    for edition,rows,rej in [('original',oldmap,oldreject),('current',cur,rejected)]:
        coverage[edition]=dict(scheduled=4991,present=sum(r['original_status']=='present' for r in rows.values()),
            auto_eligible=sum(r['kept'] for r in rows.values()),historical_exclusions=sum(r['kept'] and k in rej for k,r in rows.items()),
            extended_exclusions=sum(r['kept'] and k in extended[edition] for k,r in rows.items()),
            retained=sum(r['kept'] and k not in extended[edition] for k,r in rows.items()),
            scored_appearances=sum(v.metrics['cs_face'].n_evaluated for v in groups[edition]),
            valid_episodes=sum(v.metrics['cs_face'].value is not None for v in groups[edition]),
            rejection_reasons=dict(Counter(r['reason'] for r in rows.values() if not r['kept'])))
    report=dict(status='complete_cache_only_extended_unique_face_review',snapshot_sha256=sha(f.OUT/'snapshot.json'),
        decision_sha256=sha(OUT/'decisions.json'),original=aggregate['original'],current=aggregate['current'],
        historical=baseaggregate,coverage=coverage,per_episode=details,
        applied_exclusions={ed:list(rej.values()) for ed,rej in extended.items()},known_review_sources={str(path):sha(path) for path in files},
        source_score_files_verified=sources_verified,historical_per_episode_reproduced=280,
        models_loaded=False,new_embedding_calls=0,new_detector_calls=0,vlm_calls=0,
        protocol='Extended canonical localization review; automatic gates, original pixels/features, centroid arithmetic and episode macro unchanged. Original and current reviewed symmetrically; official raw metrics immutable.',
        scope_limits='All eligible canonical face regions screened, candidate206 full-frame reviewed. Weak406 screened. Role-conflict988 automatically ineligible, only192 previewed. Object/location semantic localization not exhaustively visually reviewed. No re-grounding/re-embedding available; raw presence not hand-repaired.')
    write(OUT/'report.json',report)
    print({ed:aggregate[ed]['all']['cs_face'] for ed in groups},flush=True)
    write(OUT/'status.json',dict(stage='extended_dino_recomputed_report_writing',coverage=coverage,vlm_calls=0))



def finish_report():
    import csv,math
    r=read(OUT/'report.json');inv=read(OUT/'inventory.json');dec=read(OUT/'decisions.json');manifest=read(OUT/'unique_face/manifest.json')
    lookup={x['review_id']:x for x in manifest['rows']};assert dec['manifest_sha256']==sha(OUT/'unique_face/manifest.json')
    assert dec['notes_sha256']==sha(OUT/'context_review_notes.json') and dec['screening_sha256']==sha(OUT/'screening.json')
    assert not inv['errors'] and len(r['per_episode'])==140 and r['source_score_files_verified']==280
    assert r['historical_per_episode_reproduced']==280
    for row in inv['source_scores']:assert sha(row['path'])==row['sha256']
    for row in inv['rows']:
        if row['crop']:assert sha(row['crop'])==row['crop_sha256']
    exclusions=[x for x in dec['decisions'] if x['decision']=='exclude']
    fields=['review_id','episode_id','run','window','shot','character','editions','evidence','context']
    with (OUT/'confirmed_localization_errors.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
        for x in exclusions:
            m=lookup[x['review_id']];w.writerow({k:(','.join(x[k]) if k=='editions' else x.get(k,m.get(k))) for k in fields})
    fields=['episode_id','edition','before','after','delta','before_scored','after_scored']
    with (OUT/'episode_changes.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
        for x in r['per_episode']:
            for ed in ['original','current']:
                a,b=x[ed]['historical'],x[ed]['extended'];w.writerow(dict(episode_id=x['episode_id'],edition=ed,before=a['value'],after=b['value'],
                    delta=b['value']-a['value'] if a['value'] is not None and b['value'] is not None else None,before_scored=a['n_evaluated'],after_scored=b['n_evaluated']))
    official=read(f.OUT/'official_comparison.json')['metrics'];table=[dict(metric='cs_face_visible_single_face_extended_reviewed',
        label='人物一致性DINO（可见单脸＋本轮扩展定位复核）',original=r['original']['all']['cs_face']['mean'],current=r['current']['all']['cs_face']['mean'])]+official
    write(OUT/'comparison.json',dict(scope='Same frozen140/2491 snapshot, extended cache-only character-localization diagnostic added; official12 unchanged.',metrics=table))
    fixed={};change_counts={}
    for ed in ['original','current']:
        valid=[x for x in r['per_episode'] if x[ed]['extended']['value'] is not None];fixed[ed]=dict(episodes=len(valid),
            before=math.fsum(x[ed]['historical']['value'] for x in valid)/len(valid),after=math.fsum(x[ed]['extended']['value'] for x in valid)/len(valid))
        changes=[x[ed]['extended']['value']-x[ed]['historical']['value'] for x in valid]
        change_counts[ed]=dict(increased=sum(d>1e-10 for d in changes),decreased=sum(d< -1e-10 for d in changes),unchanged=sum(abs(d)<=1e-10 for d in changes),
            lost_episodes=[x['episode_id'] for x in r['per_episode'] if x[ed]['historical']['value'] is not None and x[ed]['extended']['value'] is None])
    write(OUT/'fixed_valid_episode_comparison.json',dict(fixed=fixed,changes=change_counts,interpretation='Fixed valid-episode set before/after expanded review; appearance populations/centroids still change.'))
    md=['# 全140集缓存匹配审计（2026-11-07）','',
        '**缓存范围已核对，扩展人物定位复核及DINO重算已完成。当前人物DINO由0.7970变为0.7986；原版也按相同扩展规则由0.7863重算为0.7887。** 这是评测定位纠错，视频没有变化。官方12项原始分数未手工修改。','',
        '范围冻结为2026-11-06 12:37北京时间的140集／2491shot：57集完整重跑＋83集保留原版，并纳入141个已确认v2独立shot替换；不含后来完成的第58集或未完成前缀。生成保持停止，未加载模型、未请求API、未重新生成媒体。','',
        '## 完整benchmark表','',
        '| 指标↑ | 原始完整benchmark（同口径） | 当前冻结快照（同口径） |','|---|---:|---:|']
    for v in table:md.append(f"| {v['label']} | {v['original']:.4f} | {v['current']:.4f} |")
    md+=['','VLM继续关闭（strict/skip_vlm，调用0）；外观保真、动作对齐等依赖VLM的指标仍缺失，raw cs_face/cs_object仍为partial_without_vlm。表内是官方非VLM12项＋项目条件DINO诊断，不等同完整论文全指标。','',
        '## 覆盖与处理','',
        '- 原/新各140集、2491shot，人物/物体/场景合计22,874条逻辑调度记录；按相同角色/镜头/视频/代表裁剪/采样去重16,650条。裁剪文件无缺失，status与原0.20门槛无不一致；完全相同输入的原/新评分和状态也无差异。','',
        '- 对全部2,673份去重可见单脸合格裁剪，按随机顺序隐藏分数及原/新版标记筛查；206项再核对精确代表完整帧、原文、同场角色和裁剪边界。确认63份错定位，143项保留。其余2,467项仅完成裁剪初筛，不能称逐项完整视频人工审查。','',
        '- 63不是新增剔除数：包含已有复核决定和两版共享输入。本轮与有效历史精确SHA决定取并集，并对原/新对称应用；原版实际增加34项剔除（3→37），当前增加22项（35→57）。不按角色名字迁移旧决定。','',
        '- 406份去重weak人物裁剪全部初筛；988份人脸track冲突仍按原门控全部拒绝，其中192份做了可视预览。冲突项不能凭肉眼把正确角色重新纳入DINO：原门控和正确区域embedding都需要完整一致处理。','',
        '- 物体/场景完成全量文件、调度、门槛、重复像素引用检查；未完成全部物体/场景的人工语义复核。227组相同视频/采样/像素被多个实体引用仅作为候选，可能是合法重叠，未据此自动改分。','',
        '## 为什么出现率没有直接提高','',
        '错框会使人物DINO比较了另一个人；当目标本人也在该帧时，本轮可删除已确认错误的代表项，再用缓存Gram矩阵重建原特征等价表示，重算完整episode人物中心。不是删掉低分后平均旧分数。原/新280份逐集历史诊断和raw中心均已复现。','',
        '角色出现率的官方规则仍是原grounding代表区域的CLIP文本分数≥0.20。缓存没有全部候选框及正确新区域的CLIP/DINO embedding；不能把人工“看见了”直接改成官方present，也不能把错误框原来碰巧过线当成正确定位。因此本轮官方角色出现率仍为0.9479；物体、场景、动态、美学、成像等也保持不变。','',
        '## 保留的真实生成问题','',
        '人物换服装/年龄/性别/画风，单人图生成成另一个人，重复人物，目标缺失，或同画面也无法确认正确目标：均不据此剔除。部分region主体正确却把人头裁掉、仅包含背景另一张脸，保留并记录为混合区域/门控污染，不能冒充已经修好了。','',
        '例如run44/i40_j76多处Marcus生成得年轻、肤色变化，但找不到独立正确Marcus，保留；run4医院蓝衫/蓝手术服与黑马甲的身份/服装交换仍有歧义，保留。','',
        '## 口径与覆盖变化','',
        '| 项目 | 原版 | 当前版 |','|---|---:|---:|']
    for label,key in [('计划角色次','scheduled'),('官方present','present'),('自动合格','auto_eligible'),('扩展复核后保留','retained'),('形成跨镜头计分的角色次','scored_appearances'),('有效episode','valid_episodes')]:md.append(f"| {label} | {r['coverage']['original'][key]} | {r['coverage']['current'][key]} |")
    md+=['','原历史抽样复核有效集131；本轮两侧均为130，run893/i5_j15剔除Nadia错误记者框后不足两次有效出现，不填0。固定相同130个有效episode，扩展复核前后原版'+f"{fixed['original']['before']:.6f}→{fixed['original']['after']:.6f}"+'、当前'+f"{fixed['current']['before']:.6f}→{fixed['current']['after']:.6f}"+'；提升不只是少计了一集，但计分人物集合/中心仍变化，不能解释为生成身份本身提高。','',
        '**本轮扩大了复核覆盖范围，需使用本轮0.7887／0.7986成对比较。** 历史抽样复核0.7863／0.7970及原始JSON保留，不静默覆盖或混用两行。','',
        '## 可直接查看','',
        '- `confirmed_localization_errors.csv`：63项完整episode/window/shot/角色、证据与原图位置。','- `context/page_001.jpg`～`page_035.jpg`：206项隐藏分数的原始完整帧＋红框＋原文。','- `decisions.json`：全部2,673项冻结决定与精确SHA。','- `report.json`、`comparison.json`、`episode_changes.csv`：新全量表、逐集变化及原/新排除项。','- `inventory.json`、`shared_canonical_claims.json`：全量记录与技术检查。','',
        '下一阶段若恢复评分硬件，可对已确认错框重新定位并补算正确区域CLIP/DINO；人物出现率与其他官方指标需在完整受影响episode上遵循原算法重算。本轮不启动该阶段，不恢复生成。','']
    (OUT/'report.md').write_text('\n'.join(md))
    (OUT/'README.md').write_text('# 缓存匹配审计\n\n见 [完整报告](report.md)、[错定位列表](confirmed_localization_errors.csv)、[原帧证据](context/)。\n')
    write(OUT/'final_validation.json',dict(episodes=140,shots=2491,logical_items=22874,unique_items=len(inv['rows']),
        canonical_screened=2673,context_reviewed=206,confirmed_wrong_localizations=len(exclusions),weak_screened=406,
        historical_episode_scores_reproduced=280,source_score_files_reverified=280,canonical_hashes_reverified=True,
        inventory_sha256=sha(OUT/'inventory.json'),snapshot_sha256=r['snapshot_sha256'],decisions_sha256=sha(OUT/'decisions.json'),
        script_sha256=sha(__file__),official_comparison_sha256=sha(f.OUT/'official_comparison.json'),
        no_generation=True,new_embedding_calls=0,new_detector_calls=0,vlm_calls=0,official_raw_metrics_modified=False))
    write(OUT/'status.json',dict(stage='complete_cache_only_audit',report=str(OUT/'report.md'),coverage=r['coverage'],
        scope_limits=r['scope_limits'],generation_stopped=True,vlm_calls=0))
    print('report finalized',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','sheets','contexts','context_sheets','freeze_review','rescore','finish_report']);p.add_argument('--category',default='unique_face');p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int);a=p.parse_args()
    if a.action=='prepare':d=prepare();print({k:v for k,v in d.items() if k not in ['rows','source_scores']})
    elif a.action=='contexts':contexts()
    elif a.action=='context_sheets':context_sheets()
    elif a.action=='freeze_review':freeze_review()
    elif a.action=='rescore':rescore()
    elif a.action=='finish_report':finish_report()
    else:sheets(a.category,a.start,a.end)
