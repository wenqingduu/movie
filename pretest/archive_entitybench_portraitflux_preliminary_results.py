"""Archive paper, old SDXL->FLUX, Qwen and new FLUX->FLUX official results."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
from prepare_entitybench_flux_portraitagent_20ep import PROJECT,BASELINE,QWEN,OUTPUT,METHOD,read,sha256,write_json

DEST=PROJECT/'outputs/entitybench_paper_flux_qwen_portraitflux20_comparison'
METRICS=['subject_consistency','temporal_flickering','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality',
    'intra_character_presence','intra_object_presence','intra_location_presence','cs_face','cs_object','cs_transition_boundary']


def preserve(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha256(target)==sha256(source), f'Completed report changed: {source}'
    else:shutil.copy2(source,target)
    return dict(source=str(source),archive=str(target),sha256=sha256(target))


def metric(report,name,tier='all'):
    if report is None:return None
    return report[tier]['vbench'].get(name,report[tier].get(name,{})).get('mean')


def render(root=OUTPUT,dest=DEST):
    root,dest=Path(root),Path(dest);dest.mkdir(parents=True,exist_ok=True)
    paper=PROJECT/'outputs/entitybench_paper_flux_qwen20_comparison/paper_reference.json'
    preserve(paper,dest/'paper_reference.json');paper=read(paper)
    paths=dict(flux_sdxl20=BASELINE/'evaluation/official/report_pulid_a14b_4step_continuation_20ep_no_vlm.json',
        qwen20=QWEN/'evaluation/official_parallel_v2/report_qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm.json',
        flux_native20=root/'evaluation/official_parallel_v2'/f'report_{METHOD}.json')
    reports={};sources={};complete=False
    for route,path in paths.items():
        if route=='flux_native20' and not path.exists():reports[route]=None;sources[route]=None;continue
        report=read(path);assert report['all']['num_episodes']==20 and report['all']['num_shots']==363
        assert report['_manifest']['skip_vlm'] and report['_manifest']['vlm_call_stats']['total_calls']==0
        assert report['_manifest']['evaluation_scope']=='partial_without_vlm'
        if route=='flux_native20':
            status=read(root/'evaluation/official_parallel_v2/parallel_status.json')
            assert status['complete_episodes']==20 and status['complete_shots']==363
            assert read(root/'split_20ep.json')==read(BASELINE/'split_20ep.json');complete=True
            preserve(root/'evaluation/official_parallel_v2/parallel_status.json',dest/'flux_native20_parallel_status.json')
        reports[route]=report;sources[route]=preserve(path,dest/(route+'_official_report.json'))
    preserve(BASELINE/'split_20ep.json',dest/'split_20ep.json')
    rows={name:dict(paper={m:v[name] for m,v in paper['methods'].items()},
        **{route:metric(report,name) for route,report in reports.items()}) for name in METRICS}
    write_json(dest/'comparison.json',dict(status='complete_preliminary_comparison' if complete else 'waiting_for_flux_native20',
        updated_at=datetime.now(timezone.utc).isoformat(),paper=paper,reports=sources,metrics=rows,
        scope=dict(episodes=20,shots=363,vlm_enabled=False,fidelity_gate_applied=False,
            aggregation='pinned official episode macro mean',cross_shot_scores='provisional_without_vlm',
            systems='old SDXL references->FLUX; Qwen references->Qwen; new FLUX references->FLUX; no plugin causal claim',
            flux_multi_reference=False,paper_full_cohort=140,
            portrait_agent='Qwen3.7-plus text-only adapted Portrait Agent prompts; not full paper Agent/SAM2 pipeline')))
    lines=['# 论文／SDXL→FLUX／Qwen／FLUX→FLUX 初步比较','',
        '项目三路线使用相同冻结20episode/363shot和4步A14B，各自尾帧接续。新FLUX素材对应全新3D/标定及全新视频。',
        '论文140episode/2491shot；项目非VLM、无fidelity gate，官方episode宏平均；跨shot分数暂定，仅供初步观察。',
        'FLUX沿用现有单人PuLID/v7、多人物文生图路径；Qwen为原生多参考。FLUX首帧640×352/50步、Qwen1024×576/20步，模型与生成协议不同。',
        '没有新增无插件视频组，也没有通过正式指标筛选参考素材。','',
        '| 指标 | EntityMem论文 | StoryMem论文 | HoloCine论文 | CineTrans论文 | SDXL→FLUX20 | Qwen20 | FLUX→FLUX20 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in rows.items():
        values=[f'{v:.4f}' for v in row['paper'].values()]
        values += [f'{row[key]:.4f}' if row[key] is not None else '待完成' for key in reports]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','原始官方报告、split和SHA保留在comparison.json；未完成新路线字段保持null。',
        'FLUX全140保持停止；三路线已有产物保持。官方非VLM完成与全量人工审片分别记录。','']
    (dest/'README.md').write_text('\n'.join(lines))
    if complete:
        table=['# 相同20episode三条完整路线官方结果','',
            '无VLM/fidelity gate，cs_face/cs_object暂定。差值为系统路线差异，不是插件消融。','',
            '| 范围 | 指标 | 旧SDXL→FLUX | Qwen | 新FLUX→FLUX | 新−旧FLUX | 新−Qwen | 有效episode（旧/Qwen/新） |',
            '|---|---|---:|---:|---:|---:|---:|---|']
        for tier in ['all','easy','medium','hard']:
            for name in METRICS:
                values=[metric(reports[key],name,tier) for key in reports]
                if any(v is None for v in values):continue
                a,b,c=values;coverage=[]
                for report in reports.values():
                    item=report[tier]['vbench'].get(name,report[tier].get(name,{}));coverage.append(str(item['n_episodes_evaluated']))
                table.append(f'| {tier} | {name} | {a:.4f} | {b:.4f} | {c:.4f} | {c-a:+.4f} | {c-b:+.4f} | '+ '/'.join(coverage)+' |')
        (root/'official_comparison.md').write_text('\n'.join(table)+'\n')
        write_json(root/'official_comparison_sources.json',dict(sources=sources,scope='same full episodes; different reference and firstframe systems; no plugin causality',reports=reports))
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT);p.add_argument('--archive-root',type=Path,default=DEST)
    a=p.parse_args();print('complete' if render(a.output_root,a.archive_root) else 'waiting_for_flux_native20')
