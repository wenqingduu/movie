"""Preserve paper values and raw official reports; wait for Qwen without touching active inference."""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
DEST=PROJECT/'outputs/entitybench_paper_flux_qwen20_comparison'
FLUX=PROJECT/'outputs/entitybench_pulid_i2v_a14b_4step_continuation_20ep'
QWEN=PROJECT/'outputs/entitybench_qwen21_i2v_a14b_4step_continuation_20ep'
QWEN_METHOD='qwen21_native_reference_a14b_4step_continuation_20ep_no_vlm'
METRICS=['subject_consistency','temporal_flickering','motion_smoothness','dynamic_degree',
    'aesthetic_quality','imaging_quality','intra_character_presence','intra_object_presence',
    'intra_location_presence','cs_face','cs_object','cs_transition_boundary']
# Numerical facts transcribed from arXiv v1 Table 4 and Appendix F.1 Tables 12/13.
PAPER_VALUES={
    'EntityMem':[.881,.976,.988,.657,.593,66.00,.967,.888,.687,.737,.798,.738],
    'StoryMem':[.759,.838,.849,.562,.475,56.41,.849,.893,.681,.792,.839,.663],
    'HoloCine':[.860,.957,.964,.721,.518,49.97,.882,.723,.624,.751,.803,.498],
    'CineTrans':[.968,.979,.990,.688,.596,68.57,.796,.776,.651,.772,.794,.508],
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def preserve(source,destination):
    if destination.exists():assert sha(source)==sha(destination),f'Changed completed report: {source}'
    else:shutil.copy2(source,destination)
    return dict(source=str(source),archive=str(destination),sha256=sha(destination))

def metric(report,name):
    if report is None:return None
    row=report['all']['vbench'].get(name,report['all'].get(name,{}))
    return row.get('mean')

def render(root=DEST):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    paper=dict(title='EntityBench: Towards Entity-Consistent Long-Range Multi-Shot Video Generation',
        source='https://arxiv.org/html/2605.15199v1',version='v1',tables=['4','12','13'],
        episode_count=140,shot_count=2491,scope='paper-reported metrics; full benchmark; VLM/fidelity gate available',
        methods={method:dict(zip(METRICS,values)) for method,values in PAPER_VALUES.items()},
        metric_selection='12 metrics shared with current non-VLM scope; other paper metrics are not transcribed here')
    path=root/'paper_reference.json'
    if path.exists():assert read(path)==paper
    else:write(path,paper)
    flux_source=FLUX/'evaluation/official/report_pulid_a14b_4step_continuation_20ep_no_vlm.json'
    flux=read(flux_source);assert flux['all']['num_episodes']==20 and flux['all']['num_shots']==363
    records=dict(flux=preserve(flux_source,root/'flux20_official_report.json'),qwen=None)
    preserve(FLUX/'split_20ep.json',root/'split_20ep.json')
    qwen_source=QWEN/'evaluation/official_parallel_v2'/f'report_{QWEN_METHOD}.json'
    qwen=None;status=QWEN/'evaluation/official_parallel_v2/parallel_status.json'
    if qwen_source.exists() and status.exists():
        complete=read(status)
        if complete.get('complete_episodes')==20 and complete.get('complete_shots')==363:
            qwen=read(qwen_source)
            assert qwen['all']['num_episodes']==20 and qwen['all']['num_shots']==363
            assert read(QWEN/'split_20ep.json')==read(FLUX/'split_20ep.json')
            records['qwen']=preserve(qwen_source,root/'qwen20_official_report.json')
            preserve(status,root/'qwen20_parallel_status.json')
    rows={name:dict(paper={m:paper['methods'][m][name] for m in PAPER_VALUES},
        flux20=metric(flux,name),qwen20=metric(qwen,name)) for name in METRICS}
    write(root/'comparison.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),
        status='complete_preliminary_comparison' if qwen else 'waiting_for_qwen',paper=paper,
        project_scope=dict(episodes=20,shots=363,vlm_enabled=False,fidelity_gate_applied=False,
            aggregation='pinned official code episode macro mean',cross_shot_scores='provisional_without_vlm',
            interpretation='descriptive comparison of different systems/cohorts, no plugin causality or fair paper ranking'),
        reports=records,metrics=rows))
    lines=['# EntityBench初步结果存档','',
        '保存论文四个方法的12项共同指标、已有FLUX原始官方报告及本轮Qwen原始官方报告。',
        '论文140episode/2,491shot；项目20episode/363shot，非VLM、gate缺失、公开代码episode宏平均。',
        '数据集范围、身份素材、视频条件和聚合口径不同，表格只供初步观察。','',
        '[论文原文（Table 4、附录F.1）](https://arxiv.org/html/2605.15199v1)。','',
        '| 指标 | EntityMem论文 | StoryMem论文 | HoloCine论文 | CineTrans论文 | FLUX20 | Qwen20 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in METRICS:
        row=rows[name];cells=[f'{row["paper"][m]:.4f}' for m in PAPER_VALUES]
        cells += [f'{row["flux20"]:.4f}' if row['flux20'] is not None else '缺失',
                  f'{row["qwen20"]:.4f}' if row['qwen20'] is not None else '待完成']
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '机器可读表：comparison.json；论文数值/出处：paper_reference.json。',
        '原始报告：flux20_official_report.json；Qwen评分覆盖完整后保存qwen20_official_report.json。',
        '未完成的Qwen值保留null，不拿旧身份指标填充；原始报告和split包含SHA保存证据。',
        '已有FLUX/Qwen详细all/easy/medium/hard比较仍保存在Qwen输出official_comparison.md。',
        'Qwen非VLM分数归档完成不代表人工视频效果审查已完成。','']
    (root/'README.md').write_text('\n'.join(lines))
    print('Archive complete' if qwen else 'Paper and FLUX archived; waiting for Qwen',flush=True)
    return qwen is not None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=DEST);p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    while True:
        done=render(args.output_root)
        if done or not args.wait:break
        time.sleep(30)
