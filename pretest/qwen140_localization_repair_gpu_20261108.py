"""Input-bound GPU localization repair diagnostic, preserving all official outputs.

Same frozen140 videos, original representative sample, FP32 CLIP/DINO and
thresholds. Human role review is score-hidden. Person proposals are supplementary
localization candidates, never silently substituted into the official benchmark.
"""
from __future__ import annotations
import argparse,importlib.util,json,os,random,re,sys,textwrap,time
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np
import qwen140_full140_latest_20261106 as f

ROOT=f.ROOT
AUDIT=ROOT/'outputs/entitybench_cached_matching_audit_20261107'
OUT=ROOT/'outputs/entitybench_localization_repair_gpu_20261108'
OFFICIAL=f.BASE/'runtime/evaluate_benchmark_official.py'
read,write,sha,immutable=f.read,f.write,f.sha,f.immutable
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)


def prepare():
    if (OUT/'plan.json').exists():return read(OUT/'plan.json')
    audit=read(AUDIT/'decisions.json');inv=read(AUDIT/'inventory.json')
    keys={(r['episode_id'],r['character'],r['shot']) for r in audit['decisions'] if r['decision']=='exclude'}
    changes=read(ROOT/'outputs/entitybench_v2_presence_change_audit_20261107/changed_roles.json')['changes']
    declines={(r['episode_id'],r['character'],r['shot']) for r in changes if r['before']['status']=='present' and r['after']['status']=='weak'}
    keys|=declines
    rows=[r for r in inv['rows'] if r['kind']=='character' and (r['episode_id'],r['name'],r['shot']) in keys]
    random.Random(20261108).shuffle(rows)
    for i,r in enumerate(rows,1):r['repair_id']=f'P{i:03d}'
    p=dict(snapshot=str(f.OUT/'snapshot.json'),snapshot_sha256=sha(f.OUT/'snapshot.json'),
        inventory_sha256=sha(AUDIT/'inventory.json'),decisions_sha256=sha(AUDIT/'decisions.json'),
        original_evaluator=str(OFFICIAL),original_evaluator_sha256=sha(OFFICIAL),rows=rows,
        target_keys=len(keys),decline_keys=[list(k) for k in sorted(declines)],
        protocol='Supplementary score-hidden human localization repair. All confirmed63 error inputs plus counterpart original/current and five presence-decline keys. Fixed original representative sample. Original description proposals plus generic person proposals; original FP32 CLIP, DINO and .20 threshold. Official metrics and original reviewed DINO unchanged.',
        review_rule='Select only independently identifiable intended role. Preserve identity/style/costume drift, duplicates and unresolved target absence. Do not select by CLIP/DINO or accept only positive score changes.',
        official_outputs_mutable=False,paid_api_calls=0,media_generation=False)
    immutable(OUT/'plan.json',p)
    write(OUT/'status.json',dict(stage='prepared',records=len(rows),target_keys=len(keys),gpu_required=True,generation_stopped=True))
    print('prepared',len(rows),'records',len(keys),'role-shot keys',flush=True)
    return p


def load_official():
    m=read(f.BASE/'evaluation/official_parallel_v2/run_manifest.json')
    for k in ['GDINO_CONFIG','GDINO_WEIGHTS','DINO_MODEL_PATH','CLIP_MODEL_PATH']:
        value=m['model_checkpoints'][k]['path'];assert Path(value).exists(),value;os.environ[k]=value
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    spec=importlib.util.spec_from_file_location('localization_repair_frozen_official',OFFICIAL)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    assert mod.BENCHMARK_CONFIG==m['config']
    import torch,cv2
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    mod.get_grounding_dino('cuda',strict=True);mod.get_clip_model('cuda');mod.get_dino_model('cuda')
    return mod


def padded_bbox(frame,bbox):
    h,w=frame.shape[:2];a,b,c,d=map(int,bbox);dx=int((c-a)*.1);dy=int((d-b)*.1)
    return [max(0,a-dx),max(0,b-dy),min(w,c+dx),min(h,d+dy)]


def extract(limit=None):
    p=prepare();assert sha(OFFICIAL)==p['original_evaluator_sha256'];mod=load_official();start=time.monotonic()
    import torch
    for i,r in enumerate(p['rows'][:limit]):
        out=OUT/'proposals'/r['repair_id'];receipt=out/'record.json'
        if receipt.exists():
            d=read(receipt);assert d['video_sha256']==r['video_sha256'] and d['old_crop_sha256']==r['crop_sha256'];continue
        ref=r['references'][0];assert sha(ref['video'])==r['video_sha256'];assert sha(r['crop'])==r['crop_sha256']
        frames,indexes=f.diag.decode(ref['video']);frame=frames[r['sample_idx']];out.mkdir(parents=True,exist_ok=True)
        oldbox=f.diag.faces.canonical_bbox(frame,r['crop']);Image.fromarray(frame).save(out/'frame.png')
        candidates=[];hashes=set()
        def save_crop(crop,bbox,rawbox,source,confidence=None):
            path=out/f'candidate_{len(candidates):02d}.png';Image.fromarray(crop).save(path);digest=sha(path)
            if digest in hashes:path.unlink();return
            hashes.add(digest);cid=len(candidates)
            clip=mod.compute_clip_text_sim(crop,r['description'],'cuda');comp=mod.compute_selection_score(clip,crop,frame.shape)
            emb=mod.compute_dino_embedding(crop,'cuda');np.save(out/f'candidate_{cid:02d}.npy',emb)
            candidates.append(dict(candidate_id=cid,source=source,crop=str(path),crop_sha256=digest,padded_bbox=bbox,bbox=rawbox,
                confidence=confidence,clip=clip,selection=comp,embedding=str(out/f'candidate_{cid:02d}.npy')))
        save_crop(np.array(Image.open(r['crop']).convert('RGB')),oldbox,None,'original_canonical')
        for desc,source in [(r['description'],'original_description'),('person','supplementary_person_proposal')]:
            ds=mod.ground_one_entity_in_frame(frame,r['name'],desc,'character','cuda',.25,.20,True)
            for det in ds:
                bb=det['bbox'];crop=mod.crop_entity(frame,bb,.1)
                if not crop.size:continue
                save_crop(crop,padded_bbox(frame,bb),list(bb),source,det['confidence'])
        assert abs(candidates[0]['clip']-ref['clip_sim'])<1e-5,(r['repair_id'],'CLIP replay mismatch',candidates[0]['clip'],ref['clip_sim'])
        write(receipt,dict(repair_id=r['repair_id'],video_sha256=r['video_sha256'],old_crop_sha256=r['crop_sha256'],
            description=r['description'],frame_sample=r['sample_idx'],frame_index=indexes[r['sample_idx']],actual_sample_indices=indexes,
            frame_shape=list(frame.shape),candidates=candidates,plan_sha256=sha(OUT/'plan.json')))
        write(OUT/'status.json',dict(stage='extracting_gpu_candidates',completed=i+1,expected=len(p['rows']),elapsed_seconds=time.monotonic()-start,
            gpu_max_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,generation_stopped=True,api_calls=0))
        print(r['repair_id'],i+1,'/',len(p['rows']),'candidates',len(candidates),'elapsed',round(time.monotonic()-start,1),flush=True)
    print('extraction pass finished',flush=True)


def sheets():
    p=prepare();eps={x['episode_id']:x['episode'] for x in read(f.OUT/'snapshot.json')['episodes']};rows=[]
    for r in p['rows']:
        path=OUT/'proposals'/r['repair_id']/'record.json'
        if path.exists():rows.append(r)
    # Six records per sheet; no CLIP/DINO/edition labels displayed.
    for page in range(1,(len(rows)+5)//6+1):
        canvas=Image.new('RGB',(1920,2400),'white');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[(page-1)*6:page*6]):
            x=j%2*960;y=j//2*800;folder=OUT/'proposals'/r['repair_id'];data=read(folder/'record.json')
            d.text((x+4,y+3),f"{r['repair_id']} {r['run']} {r['window']} {r['shot']} TARGET {r['name']}",font=FONT,fill='black')
            im=Image.open(folder/'frame.png').convert('RGB');dr=ImageDraw.Draw(im);scale=700/im.width
            colors=['#f00000','#0044ff','#00aa00','#aa00aa','#f08000','#00aaaa']
            for c in data['candidates']:
                bb=c['padded_bbox'];dr.rectangle(bb,outline=colors[c['candidate_id']%len(colors)],width=3)
                dr.text((bb[0]+3,bb[1]+3+18*(c['candidate_id']%3)),str(c['candidate_id']),font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22),fill='white',stroke_width=2,stroke_fill='black')
            im=ImageOps.contain(im,(700,395));canvas.paste(im,(x+4,y+25))
            for k,c in enumerate(data['candidates'][:10]):
                xx=x+710+k%2*123;yy=y+25+k//2*77
                thumb=ImageOps.contain(Image.open(c['crop']),(118,57));canvas.paste(thumb,(xx,yy));d.text((xx,yy+58),'ID '+str(c['candidate_id']),font=FONT,fill='black')
            ep=eps[r['episode_id']];sh=next(s for s in ep['shots'] if s['shot_key']==r['shot']);de={v['name']:v['description'] for v in ep['entity_descriptions'].values()}
            tx='SHOT: '+r['prompt']+'\nTARGET: '+r['description']+'\nOTHER: '+'; '.join(n+': '+de[n] for n in sh['entity_schedule'].get('characters',[]) if n!=r['name'])
            lines=[]
            for line in tx.splitlines():lines.extend(textwrap.wrap(line,115))
            for k,line in enumerate(lines[:24]):d.text((x+4,y+430+15*k),line,font=FONT,fill='black')
        path=OUT/'review'/f'page_{page:03d}.jpg';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=94)
    print('sheets',len(rows),flush=True)


def freeze_choices():
    p=prepare();notes=read(OUT/'review/choices_notes.json')
    assert set(notes['decisions'])=={r['repair_id'] for r in p['rows']}
    rows=[]
    for r in p['rows']:
        path=OUT/'proposals'/r['repair_id']/'record.json';data=read(path)
        ids=notes['decisions'][r['repair_id']]['accepted_candidate_ids']
        assert set(ids)<=set(range(len(data['candidates'])))
        rows.append(dict(repair_id=r['repair_id'],accepted_candidate_ids=ids,
            reason=notes['decisions'][r['repair_id']]['reason'],proposal_sha256=sha(path),
            accepted_crop_sha256=[data['candidates'][i]['crop_sha256'] for i in ids]))
    immutable(OUT/'review/frozen_choices.json',dict(plan_sha256=sha(OUT/'plan.json'),notes_sha256=sha(OUT/'review/choices_notes.json'),
        rule=notes['review_rule'],score_hidden=True,decisions=rows))
    chosen=[]
    # Only after the immutable, score-hidden role decisions have been saved.
    for r,decision in zip(p['rows'],rows):
        data=read(OUT/'proposals'/r['repair_id']/'record.json');ids=decision['accepted_candidate_ids']
        i=0 if not ids or 0 in ids else max(ids,key=lambda j:data['candidates'][j]['selection']['selection'])
        c=data['candidates'][i]
        chosen.append(dict(**r,chosen=c,review_resolution='unresolved' if not ids else 'original_correct' if i==0 else 'relocalized',
            original_clip=data['candidates'][0]['clip'],new_status='present' if c['clip']>=.20 else 'weak'))
    immutable(OUT/'selected.json',dict(frozen_choices_sha256=sha(OUT/'review/frozen_choices.json'),rows=chosen))
    print('frozen',len(chosen),'relocalized',sum(r['review_resolution']=='relocalized' for r in chosen),flush=True)


def gate_maps():
    old,_=f.diag.historical();old={f.diag.key(r):r for r in old}
    _,_,cur,_,_=f.paired.load_maps()
    for item in read(f.OUT/'snapshot.json')['new_complete_episodes']:
        for path in (f.REGION/'extraction'/item['episode_id']).glob('*.json'):
            for r in read(path)['rows']:cur[f.diag.key(r)]=r
    return {'original':old,'current':cur}


def repaired_gates(selected,maps):
    """Re-gate every role claim in each changed shot, including cross-time collisions."""
    import copy,torch,cv2
    from facexlib.detection import init_detection_model
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=init_detection_model('retinaface_resnet50',half=False,device='cuda',model_rootpath=str(f.diag.faces.WEIGHT.parent))
    groups=defaultdict(dict)
    for r in selected:
        if r['review_resolution']!='relocalized':continue
        for ref in r['references']:groups[ref['edition'],r['episode_id'],r['shot']][r['name']]=r
    inv=read(AUDIT/'inventory.json');anchors=defaultdict(list)
    for r in inv['rows']:
        if r['kind']!='character':continue
        for ref in r['references']:
            k=(ref['edition'],r['episode_id'],r['shot'])
            if k in groups:anchors[k].append((r,ref))
    result=copy.deepcopy(maps);changes=[];start=time.monotonic()
    for n,(key,repairs) in enumerate(groups.items(),1):
        ed,eid,sk=key;dest=OUT/'face_gates'/ed/eid/(sk.replace(':','_')+'.json')
        if dest.exists():
            d=read(dest);assert d['selected_sha256']==sha(OUT/'selected.json')
        else:
            items=anchors[key];video=items[0][1]['video'];vsha=items[0][0]['video_sha256'];assert sha(video)==vsha
            frames,indexes=f.diag.decode(video);rows=[]
            for r,ref in items:
                rr=copy.deepcopy(maps[ed][eid,r['name'],sk]);rr['relocalized']=r['name'] in repairs
                if rr['relocalized']:
                    ch=repairs[r['name']];c=ch['chosen'];rr.update(original_crop=c['crop'],original_crop_sha256=c['crop_sha256'],
                        original_status=ch['new_status'],region=c['padded_bbox'])
                rows.append(rr)
            claims=[r['sample_idx'] for r in rows if r['original_status']=='present']
            required=range(min(claims),max(claims)+1) if claims else []
            ds=[]
            for sample,frame in enumerate(frames):
                det=[]
                if sample in required:
                    with torch.inference_mode():raw=model.detect_faces(frame[:,:,::-1].copy(),conf_threshold=.9,nms_threshold=.4,use_origin_size=True)
                    det=[dict(bbox=[float(v) for v in z[:4]],confidence=float(z[4]),landmarks=[float(v) for v in z[5:]])
                         for z in raw if min(z[2]-z[0],z[3]-z[1])>=24]
                ds.append(det)
            tracks=f.diag.faces.build_tracks(ds)
            for r in rows:
                r.update(kept=False,reason='original_not_present',candidates=[],face_track_id=None)
                if r['original_status']!='present':continue
                for tr in tracks:
                    for det in tr['items']:
                        if det['sample_idx']!=r['sample_idx']:continue
                        overlap=f.diag.gate.containment(det['bbox'],r['region'])
                        if overlap>=.8:r['candidates'].append(dict(track_id=tr['track_id'],bbox=det['bbox'],face_coverage=overlap,confidence=det['confidence']))
                count=len(r['candidates']);r['kept']=count==1
                r['reason']='unique_visible_face' if count==1 else 'multiple_faces_inside_region' if count>1 else 'no_complete_face_inside_region'
                if count==1:r['face_track_id']=r['candidates'][0]['track_id']
            collisions=defaultdict(list)
            for r in rows:
                if r['kept']:collisions[r['face_track_id']].append(r)
            for rs in collisions.values():
                if len(rs)>1:
                    for r in rs:r.update(kept=False,reason='same_face_track_claimed_by_multiple_roles')
            d=dict(rows=rows,tracks=tracks,actual_sample_indices=indexes,video_sha256=vsha,selected_sha256=sha(OUT/'selected.json'),
                detector_sha256=sha(f.diag.faces.WEIGHT),required_span=list(required))
            write(dest,d)
        for r in d['rows']:
            k=f.diag.key(r);before=maps[ed][k];result[ed][k]=r
            if (before['kept'],before['reason'])!=(r['kept'],r['reason']):changes.append(dict(edition=ed,episode_id=eid,shot=sk,character=r['character'],
                relocalized=r.get('relocalized'),before=before['reason'],after=r['reason']))
        if n%10==0:print('gates',n,'/',len(groups),'elapsed',round(time.monotonic()-start,1),flush=True)
    write(OUT/'gate_changes.json',dict(changed_shot_editions=len(groups),changes=changes,elapsed_seconds=time.monotonic()-start,
        rule='All original/new present claims in contiguous required sample span; reject every shared spatial-track claim. No identity embedding assignment.'))
    del model;torch.cuda.empty_cache()
    return result


def score():
    import copy
    from collections import Counter
    p=prepare();frozen=read(OUT/'review/frozen_choices.json');assert frozen['notes_sha256']==sha(OUT/'review/choices_notes.json')
    selected=read(OUT/'selected.json');assert selected['frozen_choices_sha256']==sha(OUT/'review/frozen_choices.json')
    selected=selected['rows'];maps=gate_maps();newmaps=repaired_gates(selected,maps)
    mod=load_official();snap=read(f.OUT/'snapshot.json');audit=read(AUDIT/'report.json');inv=read(AUDIT/'inventory.json')
    affected={r['episode_id'] for r in selected};byed={ed:defaultdict(list) for ed in maps};new={};rejected={ed:{} for ed in maps}
    for ed in maps:
        for r in audit['applied_exclusions'][ed]:rejected[ed][f.diag.key(r)]=r
    for r in selected:
        for ref in r['references']:new[ref['edition'],r['episode_id'],r['name'],r['shot']]=r
    embed_dir=OUT/'embeddings';embed_dir.mkdir(exist_ok=True)
    calls=0;started=time.monotonic()
    for r in inv['rows']:
        if r['kind']!='character' or r['episode_id'] not in affected:continue
        assert not r['crop'] or sha(r['crop'])==r['crop_sha256']
        for ref in r['references']:byed[ref['edition']][r['episode_id']].append((r,ref))
        if not any(ref['status']=='present' for ref in r['references']):continue
        path=embed_dir/(r['crop_sha256']+'.npy')
        if not path.exists():np.save(path,mod.compute_dino_embedding(np.array(Image.open(r['crop']).convert('RGB')),'cuda'));calls+=1
        if calls and calls%200==0:print('embeddings',calls,'elapsed',round(time.monotonic()-started,1),flush=True)
    baseline_raw={ed:[] for ed in maps};baseline_review={ed:[] for ed in maps};repaired_raw={ed:[] for ed in maps};repaired_review={ed:[] for ed in maps}
    validation=[];episode_changes=[];role_changes=[];coverage={ed:Counter() for ed in maps}
    for item in snap['episodes']:
        ep=item['episode'];eid=ep['episode_id'];order=[s['shot_key'] for s in ep['shots']]
        for ed in maps:
            source=item['original_result' if ed=='original' else 'result'];raw=read(source)
            assert sha(source)==item['original_result_sha256' if ed=='original' else 'result_sha256']
            reviewraw=read(AUDIT/'recomputed'/ed/'results'/(eid+'.json'))
            br=mod._episode_result_from_json(eid,ep['difficulty'],raw);bd=mod._episode_result_from_json(eid,ep['difficulty'],reviewraw)
            baseline_raw[ed].append(br);baseline_review[ed].append(bd)
            if eid not in affected:repaired_raw[ed].append(br);repaired_review[ed].append(bd);continue
            pools={name:[] for name in [v['name'] for v in ep['entity_descriptions'].values()]};newp=copy.deepcopy(pools);oldfiltered=copy.deepcopy(pools);nf=copy.deepcopy(pools)
            oldpresence=defaultdict(list);newpresence=defaultdict(list)
            for r,ref in byed[ed][eid]:
                k=(eid,r['name'],r['shot']);oldpresence[r['shot']].append(int(ref['status']=='present'))
                replace=new.get((ed,*k));changed=replace is not None and replace['review_resolution']=='relocalized'
                c=replace['chosen'] if changed else None;status=replace['new_status'] if changed else ref['status'];newpresence[r['shot']].append(int(status=='present'))
                if changed:role_changes.append(dict(edition=ed,episode_id=eid,shot=r['shot'],character=r['name'],repair_id=replace['repair_id'],
                    old_clip=ref['clip_sim'],new_clip=c['clip'],old_status=ref['status'],new_status=status,old_crop=r['crop'],new_crop=c['crop'],
                    face_gate=newmaps[ed][k]['reason']))
                if ref['status']=='present':
                    emb=np.load(embed_dir/(r['crop_sha256']+'.npy'))
                    crop=mod.EntityCrop(r['name'],'character',r['description'],r['shot'],r['sample_idx'],tuple(maps[ed][k]['region']),r['crop'],ref['clip_sim'],emb)
                    pools[r['name']].append(crop)
                    if maps[ed][k]['kept'] and k not in rejected[ed]:oldfiltered[r['name']].append(crop)
                if status=='present':
                    emb=np.load(c['embedding']) if changed else np.load(embed_dir/(r['crop_sha256']+'.npy'))
                    crop=mod.EntityCrop(r['name'],'character',r['description'],r['shot'],r['sample_idx'],tuple(c['padded_bbox'] if changed else maps[ed][k]['region']),
                        c['crop'] if changed else r['crop'],c['clip'] if changed else ref['clip_sim'],emb)
                    newp[r['name']].append(crop);rr=newmaps[ed][k]
                    excluded=k in rejected[ed] and rr['original_crop_sha256']==rejected[ed][k]['original_crop_sha256']
                    if rr['kept'] and not excluded:nf[r['name']].append(crop)
                    coverage[ed]['retained_affected']+=int(rr['kept'] and not excluded)
            oldvalue,_,oldn=mod.compute_cross_shot_consistency(pools,'character',shot_order=order)
            oldd,_,olddn=mod.compute_cross_shot_consistency(oldfiltered,'character',shot_order=order)
            val=lambda a,b:0. if a is None and b is None else abs(a-b) if a is not None and b is not None else float('inf')
            err=val(oldvalue,raw['metrics']['cs_face']['value']);derr=val(oldd,reviewraw['metrics']['cs_face']['value'])
            op=float(np.mean([np.mean(oldpresence[s]) for s in order if oldpresence[s]]));npres=float(np.mean([np.mean(newpresence[s]) for s in order if newpresence[s]]))
            assert err<1e-5 and derr<1e-5 and abs(op-raw['metrics']['intra_character_presence']['value'])<1e-10,(eid,ed,err,derr)
            assert oldn==raw['metrics']['cs_face']['n_evaluated'] and olddn==reviewraw['metrics']['cs_face']['n_evaluated']
            validation.append(dict(edition=ed,episode_id=eid,raw_dino_absolute_error=err,reviewed_dino_absolute_error=derr,presence_reproduced=True))
            value,entities,n=mod.compute_cross_shot_consistency(newp,'character',shot_order=order)
            dvalue,dentities,dn=mod.compute_cross_shot_consistency(nf,'character',shot_order=order)
            nr=mod._episode_result_from_json(eid,ep['difficulty'],copy.deepcopy(raw));nr.metrics['cs_face']=mod.MetricValue(value=value,n_evaluated=n)
            nr.metrics['intra_character_presence'].value=npres
            nd=mod.EpisodeResult(eid,ep['difficulty'],len(order));nd.metrics['cs_face']=mod.MetricValue(value=dvalue,n_evaluated=dn)
            repaired_raw[ed].append(nr);repaired_review[ed].append(nd)
            detail=dict(episode_id=eid,edition=ed,presence_before=op,presence_after=npres,raw_dino_before=oldvalue,raw_dino_after=value,
                reviewed_dino_before=oldd,reviewed_dino_after=dvalue,old_scored=olddn,new_scored=dn)
            episode_changes.append(detail)
            write(OUT/'recomputed'/ed/'results'/(eid+'.json'),dict(**detail,metrics={k:v.to_json() for k,v in nr.metrics.items()},
                corrected_visible_single_face_dino=nd.metrics['cs_face'].to_json(),per_entity_scores=entities,reviewed_per_entity_scores=dentities))
    split=read(f.BASE/'split_full140.json');agg=lambda groups:{ed:mod.aggregate_results(v,split) for ed,v in groups.items()}
    report=dict(status='complete_supplementary_localization_repair',episodes=140,shots=2491,affected_episodes=len(affected),affected_context_shots=sum(len(x['episode']['shots']) for x in snap['episodes'] if x['episode_id'] in affected),
        original_official=agg(baseline_raw),original_reviewed=agg(baseline_review),localization_repaired=agg(repaired_raw),localization_repaired_visible_single_face=agg(repaired_review),
        per_episode=episode_changes,role_changes=role_changes,validation=validation,coverage={ed:dict(c) for ed,c in coverage.items()},
        review_counts=dict(Counter(r['review_resolution'] for r in selected)),snapshot_sha256=sha(f.OUT/'snapshot.json'),
        selected_sha256=sha(OUT/'selected.json'),frozen_review_sha256=sha(OUT/'review/frozen_choices.json'),new_canonical_embedding_calls=calls,
        scoring_elapsed_seconds=time.monotonic()-started,official_files_modified=False,generation_stopped=True,vlm_calls=0,api_calls=0,
        scope='Supplementary, human-localization-repaired original/current fixed-sample person-region diagnostic. Official raw metrics and canonical-pixel reviewed default remain immutable. Proposals from original descriptions + generic person, score-hidden role approval; highest original selection score among accepted adequate target regions. Other 11 official metrics unchanged; complete affected episodes re-centred.')
    write(OUT/'report.json',report);write(OUT/'status.json',dict(stage='scores_complete_report_pending',report=str(OUT/'report.json'),generation_stopped=True,api_calls=0))
    print('scores complete',len(validation),'episode validations',flush=True)


def validate_gate_replay():
    """Verify GPU/native RetinaFace preserves old decisions before changing regions."""
    import copy,torch
    from facexlib.detection import init_detection_model
    torch.set_num_threads(4)
    maps=gate_maps();inv=read(AUDIT/'inventory.json');videos={}
    for r in inv['rows']:
        if r['kind']=='character':
            for ref in r['references']:videos[ref['edition'],r['episode_id'],r['shot']]=ref['video']
    model=None;differences=[];checked=0;extra=0
    for path in sorted((OUT/'face_gates').glob('*/*/*.json')):
        d=read(path);ed=path.parts[-3];rows=[copy.deepcopy(maps[ed][f.diag.key(r)]) for r in d['rows']]
        samples=[r['sample_idx'] for r in rows if r['original_status']=='present'];span=list(range(min(samples),max(samples)+1)) if samples else []
        detections=[[] for _ in range(5)]
        for tr in d['tracks']:
            for face in tr['items']:
                if face['sample_idx'] in span:detections[face['sample_idx']].append({k:v for k,v in face.items() if k!='sample_idx'})
        missing=set(span)-set(d['required_span'])
        if missing:
            if model is None:model=init_detection_model('retinaface_resnet50',half=False,device='cuda',model_rootpath=str(f.diag.faces.WEIGHT.parent))
            video=videos[ed,rows[0]['episode_id'],rows[0]['shot']];assert sha(video)==d['video_sha256'];frames,_=f.diag.decode(video)
            for sample in sorted(missing):
                with torch.inference_mode():det=model.detect_faces(frames[sample][:,:,::-1].copy(),conf_threshold=.9,nms_threshold=.4,use_origin_size=True)
                detections[sample]=[dict(bbox=[float(v) for v in z[:4]],confidence=float(z[4]),landmarks=[float(v) for v in z[5:]]) for z in det if min(z[2]-z[0],z[3]-z[1])>=24];extra+=1
        # build_tracks is order-sensitive for exact ties; retain detector order.
        # Tracks preserve assignment, not detector ordering. For unambiguous
        # assignments geometric permutation cannot affect accepted claims.
        tracks=f.diag.faces.build_tracks(detections);claims=defaultdict(list)
        for r in rows:
            found=[]
            if r['original_status']=='present':
                for tr in tracks:
                    for face in tr['items']:
                        if face['sample_idx']==r['sample_idx'] and f.diag.gate.containment(face['bbox'],r['region'])>=.8:found.append(tr['track_id'])
            r['replay_kept']=len(found)==1;r['replay_reason']='original_not_present' if r['original_status']!='present' else 'unique_visible_face' if len(found)==1 else 'multiple_faces_inside_region' if found else 'no_complete_face_inside_region'
            if r['replay_kept']:claims[found[0]].append(r)
        for rs in claims.values():
            if len(rs)>1:
                for r in rs:r.update(replay_kept=False,replay_reason='same_face_track_claimed_by_multiple_roles')
        for r in rows:
            checked+=1
            if (r['kept'],r['reason'])!=(r['replay_kept'],r['replay_reason']):differences.append(dict(edition=ed,episode_id=r['episode_id'],shot=r['shot'],character=r['character'],before=r['reason'],replay=r['replay_reason']))
    write(OUT/'gate_replay_validation.json',dict(shot_editions=82,canonical_claims_checked=checked,differences=differences,extra_original_span_frames_detected=extra,
        detector_sha256=sha(f.diag.faces.WEIGHT),description='Before changing regions: replay all original claims with FP32 native GPU detector, original required spans/size/containment/tracking/conflict gates.'))
    assert not differences,differences
    print('gate replay verified',checked,'original claims',flush=True)


def finish_report():
    import csv,math
    from collections import Counter
    p=prepare();r=read(OUT/'report.json');selection=read(OUT/'selected.json');chosen=selection['rows'];review=read(OUT/'review/frozen_choices.json')
    assert r['selected_sha256']==sha(OUT/'selected.json') and r['frozen_review_sha256']==sha(OUT/'review/frozen_choices.json')
    assert review['notes_sha256']==sha(OUT/'review/choices_notes.json') and review['plan_sha256']==sha(OUT/'plan.json')
    assert sha(f.OUT/'snapshot.json')==p['snapshot_sha256']==r['snapshot_sha256']
    assert sha(OFFICIAL)==p['original_evaluator_sha256'] and sha(AUDIT/'inventory.json')==p['inventory_sha256']
    assert len(chosen)==118 and len(r['validation'])==66
    gatecheck=read(OUT/'gate_replay_validation.json');assert not gatecheck['differences'] and gatecheck['canonical_claims_checked']==203
    clips=[]
    for x,d in zip(chosen,review['decisions']):
        assert d['repair_id']==x['repair_id'];path=OUT/'proposals'/x['repair_id']/'record.json';assert sha(path)==d['proposal_sha256']
        rec=read(path);clips.append(abs(rec['candidates'][0]['clip']-x['references'][0]['clip_sim']))
        for c in rec['candidates']:assert sha(c['crop'])==c['crop_sha256']
        assert sha(x['references'][0]['video'])==x['video_sha256']
    assert max(clips)<1e-5
    official=read(f.OUT/'official_aggregates.json')
    for ed in ['original','current']:
        assert r['original_official'][ed]==official[ed]
        for tier in ['all','easy','medium','hard']:
            for metric,value in official[ed][tier].items():
                if metric not in ['intra_character_presence','cs_face']:assert r['localization_repaired'][ed][tier][metric]==value,(ed,tier,metric)
    maps=gate_maps();audit=read(AUDIT/'report.json');snap=read(f.OUT/'snapshot.json');coverage={};exact={};oldreview={};newreview={}
    for ed in maps:
        for path in (OUT/'face_gates'/ed).glob('*/*.json'):
            for row in read(path)['rows']:maps[ed][f.diag.key(row)]=row
        rejections={f.diag.key(x):x for x in audit['applied_exclusions'][ed]}
        excluded={k for k,x in rejections.items() if maps[ed][k]['original_crop_sha256']==x['original_crop_sha256']}
        nr=[];nd=[];br=[];bd=[];lost=[];gained=[]
        for item in snap['episodes']:
            eid=item['episode_id'];before=read(item['original_result' if ed=='original' else 'result'])
            assert sha(item['original_result' if ed=='original' else 'result'])==item['original_result_sha256' if ed=='original' else 'result_sha256']
            path=OUT/'recomputed'/ed/'results'/(eid+'.json');after=read(path) if path.exists() else before
            oldd=read(AUDIT/'recomputed'/ed/'results'/(eid+'.json'))['metrics']['cs_face']
            newd=after['corrected_visible_single_face_dino'] if path.exists() else oldd
            br.append(before['metrics']);nr.append(after['metrics']);bd.append(oldd);nd.append(newd)
            if oldd['value'] is not None and newd['value'] is None:lost.append(eid)
            if oldd['value'] is None and newd['value'] is not None:gained.append(eid)
        mean=lambda seq:math.fsum(x for x in seq if x is not None)/sum(x is not None for x in seq)
        exact[ed]={m:dict(before=mean([a[m]['value'] for a in br]),after=mean([a[m]['value'] for a in nr])) for m in ['intra_character_presence','cs_face']}
        exact[ed]['visible_single_face_dino']=dict(before=mean([a['value'] for a in bd]),after=mean([a['value'] for a in nd]))
        coverage[ed]=dict(scheduled=len(maps[ed]),present=sum(x['original_status']=='present' for x in maps[ed].values()),
            auto_eligible=sum(x['kept'] for x in maps[ed].values()),retained=sum(x['kept'] and k not in excluded for k,x in maps[ed].items()),
            remaining_original_exclusions=sum(x['kept'] and k in excluded for k,x in maps[ed].items()),scored_appearances=sum(a['n_evaluated'] for a in nd),
            valid_episodes=sum(a['value'] is not None for a in nd),lost_valid_episodes=lost,gained_valid_episodes=gained,
            rejection_reasons=dict(Counter(x['reason'] for x in maps[ed].values() if not x['kept'])))
    report=dict(episodes=140,shots=2491,affected_episodes=33,complete_context_shots=r['affected_context_shots'],deduplicated_inputs=118,
        old_canonical_clip_max_error=max(clips),raw_dino_max_error=max(x['raw_dino_absolute_error'] for x in r['validation']),
        reviewed_dino_max_error=max(x['reviewed_dino_absolute_error'] for x in r['validation']),all66_episode_baselines_reproduced=True,
        original_gpu_face_gate_claims_reproduced=203,gate_differences=[],official_other_metrics_exactly_unchanged=True,
        official_files_unchanged=True,review= r['review_counts'],coverage=coverage,exact_episode_macro=exact,
        frozen_choices_sha256=sha(OUT/'review/frozen_choices.json'),selected_sha256=sha(OUT/'selected.json'),
        report_sha256=sha(OUT/'report.json'),script_sha256=sha(__file__),snapshot_sha256=sha(f.OUT/'snapshot.json'),
        canonical_embedding_files=len(list((OUT/'embeddings').glob('*.npy'))),
        proposal_embedding_files=len(list((OUT/'proposals').glob('*/*.npy'))),vlm_calls=0,paid_api_calls=0,media_generation=False,
        limits='Only118 predeclared records re-localized.16 unresolved retained, not forced present. Nonhuman/back-view/small/occluded faces still follow original gates. Human-approved supplementary person proposals change pixel support; not interchangeable with canonical-pixel default DINO or official EntityBench. Objects/locations were technically audited previously, not exhaustively semantically repaired.')
    write(OUT/'coverage_and_validation.json',report)
    for name,rows in [('role_changes',r['role_changes']),('episode_changes',r['per_episode'])]:
        with (OUT/(name+'.csv')).open('w',newline='') as file:
            w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=read(f.OUT/'official_comparison.json')['metrics'];baseline_default=dict(metric='canonical_visible_single_face_extended_reviewed',label='人物DINO（原像素、可见单脸＋扩展定位复核）',
        original=audit['original']['all']['cs_face']['mean'],current=audit['current']['all']['cs_face']['mean'])
    supplement=[dict(metric=m,**{ed:exact[ed][m] for ed in exact}) for m in ['intra_character_presence','cs_face','visible_single_face_dino']]
    write(OUT/'comparison.json',dict(scope=r['scope'],official_and_canonical_default=[baseline_default]+table,supplementary_localization_repair=supplement,
        coverage=coverage,episodes=140,shots=2491,vlm_calls=0,official_scores_immutable=True))
    md=['# GPU人物定位修复与完整benchmark补充评估（2026-11-08）','',
        '**已完成118份输入的GPU候选框生成、隐藏分数角色复核、正确区域CLIP/DINO补算与33个受影响完整episode的重新聚合。** 视频生成继续停止；API/VLM调用均为0。','',
        '输入仍冻结在11月6日的140集／2491shot：57集完整重跑＋83集原版，并含141个已确认v2独立shot替换。不混入第58集或未完成前缀。原始官方评分及历史报告保留。','',
        '## 先看定位修复的实际变化','',
        '下表是**补充诊断**：原版和当前版对称处理，固定原代表采样时刻；新的正确人物区域重新编码。它不覆盖原始官方指标，也不与原像素的默认DINO混称同一结果。','',
        '| 补充评估指标↑ | 原版：修复前 | 原版：修复后 | 当前：修复前 | 当前：修复后 |','|---|---:|---:|---:|---:|']
    for label,key in [('角色出现率','intra_character_presence'),('人物区域DINO（未加单脸条件）','cs_face'),('人物区域DINO（可见单脸条件）','visible_single_face_dino')]:
        a,b=exact['original'][key],exact['current'][key];md.append(f"| {label} | {a['before']:.6f} | {a['after']:.6f} | {b['before']:.6f} | {b['after']:.6f} |")
    md+=['','修复没有带来大幅“涨分”。当前未加单脸条件的区域DINO小幅上升，单脸条件DINO略降；不能只保留涨分项。纠错会同时找回原来漏掉的人、移除碰巧过线的错误框，并改变条件计分覆盖和整集人物中心。这不是视频生成结果发生变化。','',
        '## 当前完整benchmark：官方评分与原像素默认诊断','',
        '| 指标↑ | 原始完整benchmark | 当前冻结140集 |','|---|---:|---:|']
    for x in [baseline_default]+table:md.append(f"| {x['label']} | {x['original']:.4f} | {x['current']:.4f} |")
    md+=['','上表官方12项不变；默认DINO沿用11月7日的原像素扩展复核值.7887／.7986。更早历史抽样复核.7863／.7970继续保留。新增正确区域补充诊断另列上表，未静默替代。VLM依赖的外观保真、动作对齐等仍为空；raw cs_face/cs_object为partial_without_vlm。','',
        '## 处理范围与规则','',
        '- 来源：之前确认的63份错框去重为61个角色/shot键，加上5个出现率下降案例，共66键；补齐原版/当前版对应输入后去重118份，覆盖33集、887个完整上下文shot。','- 118份全部按打乱顺序隐藏CLIP/DINO及版本标签复核：74份有可用正确新区域，28份对应版本的原框本来正确，16份目标缺失、混淆或无可用候选区域，保持原框和既有排除状态。74份映射到82条版本角色记录。','- 使用原描述GroundingDINO候选＋补充通用person候选；先冻结人审接受集合，再以原CLIP×清晰度×面积选择规则在集合内取代表区域。新区域不按DINO分数选择；保持原0.20出现阈值，不用肉眼直接写present。','- 只改空间定位，不换代表时间，不补生成缺失人物，不修饰真实身份/服装/画风漂移或重复生成。背身区域即使正确仍可能没有合格人脸；混框含多张脸仍拒绝。','- 改框的82个版本shot，对同shot所有原/新present角色重做可见单脸及跨采样时刻的空间track冲突门控。无身份余弦匹配；同一track被多角色认领仍全部拒绝。','- 整集重新编码原canonical区域并重算全部人物中心：66份逐集原始/当前基线全部复现。其余107集复用冻结兼容结果，再由官方聚合函数在完整140集split上集间等权聚合。','',
        '## 出现率变化的具体项','',
        '| 输入 | 版本 | 人物 | 原CLIP→正确区域CLIP | 原状态→新状态 |','|---|---|---|---:|---|']
    lookup={x['repair_id']:x for x in chosen}
    for c in r['role_changes']:
        if c['old_status']==c['new_status']:continue
        x=lookup[c['repair_id']];md.append(f"| {x['run']}/{x['window']}/{x['shot']} | {c['edition']} | {c['character']} | {c['old_clip']:.4f}→{c['new_clip']:.4f} | {c['old_status']}→{c['new_status']} |")
    md+=['','Meilin（run654/5:3）原框主要是Arthur，修复后红袍女子CLIP过线；Min-jun（run1/5:1）原框是蓝衣Sung-min，改到桌边黑衣Min-jun后过线。原版中部分旧错框本来碰巧过线，正确目标背身、远景或服装漂移后分数反而不过线，照实计入。','',
        '此前5降案例不能全部归结为评测：Anya的原代表时刻已处于变装/遮挡，无法无歧义识别；Marcus缺少足够正确区域；run1/1:1 Min-jun的原框正确但CLIP仍不足.20。没有换时间或阈值来消掉这些问题。','',
        '## 单脸条件计分覆盖','',
        '| 项目 | 原版修复前 | 原版修复后 | 当前修复前 | 当前修复后 |','|---|---:|---:|---:|---:|']
    for label,key in [('官方/补充present','present'),('自动单脸合格','auto_eligible'),('复核后保留','retained'),('跨镜头计分角色次','scored_appearances'),('有效episode','valid_episodes')]:
        md.append(f"| {label} | {audit['coverage']['original'][key]} | {coverage['original'][key]} | {audit['coverage']['current'][key]} | {coverage['current'][key]} |")
    md+=['','新旧有效episode数相同仍不表示计分角色集合相同。新增人物区域可能包含更难的姿态/身份漂移；均值下降不能据此认定视频变差，也不构成撤回正确定位的理由。','',
        '## 数值与执行核验','',
        f"- 118份旧canonical CLIP最大复现误差：{max(clips):.3g}；66份逐集raw DINO最大误差{report['raw_dino_max_error']:.3g}、默认复核DINO最大误差{report['reviewed_dino_max_error']:.3g}。",
        '- 203条原canonical角色门控用GPU原生FP32检测重放，结果差异0；包含原/新所需采样跨度不同的情况。',
        '- 视频/原图/原评分/评测源/决策输入SHA核验通过。官方其余指标的各难度聚合逐项完全不变。',
        f"- 新提取原canonical embedding {report['canonical_embedding_files']}份，候选embedding {report['proposal_embedding_files']}份；第二遍验证复用相同输入缓存，无重复推理。候选框GPU阶段约106秒，82个shot版本单脸检测约28秒，原区域embedding约40秒；另有模型加载、完整帧人工复核和汇总时间。单卡48GiB4090运行，未与生成共享GPU。",'',
        '## 文件','',
        '- `comparison.json`：完整benchmark原始13行＋独立定位修复对比。','- `coverage_and_validation.json`：精确未四舍五入集均值、覆盖、输入绑定和验证。','- `role_changes.csv`／`episode_changes.csv`：每个改框角色及全部33集两版变化，含负变化。','- `selected.json`：原/新框、选择、CLIP、原采样时刻及源视频。','- `review/frozen_choices.json`、`review/page_001.jpg`～`page_020.jpg`：计分前冻结的人审集合及隐藏分数证据。','- `proposals/P###/`：完整帧、原区域、候选区域及embedding；`face_gates/`保留重算门控。','',
        '范围限制：不是所有人物框/角色关联均已证明正确；本轮只修复预先登记118份输入。此前988项track冲突大多仍按原规则拒绝；物体/场景未做全量语义纠错。无法识别的真实生成问题保留，不能宣称整套benchmark已完全人工标注或已达到最优。','']
    (OUT/'report.md').write_text('\n'.join(md));(OUT/'README.md').write_text('# GPU人物定位修复补充评估\n\n见[完整报告](report.md)、[逐角色变化](role_changes.csv)、[逐集变化](episode_changes.csv)、[覆盖与验证](coverage_and_validation.json)。\n\n原官方分数与默认原像素DINO保留；新区域诊断单独列出。生成保持停止，API/VLM调用0。\n')
    write(OUT/'status.json',dict(stage='complete_reviewed_supplementary_localization_repair',report=str(OUT/'report.md'),coverage=coverage,
        canonical_default_unchanged=True,official_raw_unchanged=True,generation_stopped=True,api_calls=0,vlm_calls=0))
    print('report complete',exact,flush=True)


if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('action',choices=['prepare','extract','sheets','freeze','score','validate','finish']);pa.add_argument('--limit',type=int);a=pa.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='extract':extract(a.limit)
    elif a.action=='sheets':sheets()
    elif a.action=='freeze':freeze_choices()
    elif a.action=='validate':validate_gate_replay()
    elif a.action=='finish':finish_report()
    else:score()
