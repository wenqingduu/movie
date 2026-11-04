"""Recover exact per-role noised-reference operands from saved v7 states.

VAE-only reconstruction, validated against every saved post-injection latent.
No transformer denoising, new noise draw, detection, 3D rendering or video/API.
Original frozen trajectory sources and earlier image packages are untouched.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import MethodType, SimpleNamespace
import zipfile

ROOT = Path('/root/autodl-tmp/movie')
OLD = ROOT/'outputs/run594_new_injected_denoising_replay_20261101'
OUT = ROOT/'outputs/run594_noised_reference_reconstruction_20261101'
DELIVERY = ROOT/'outputs/run594_new_injected_denoising_with_noised_refs_20261101'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    import torch
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
    from diffusers.image_processor import VaeImageProcessor
    import run_qwen_image21_shared_prefix_adaptive as v7
    import export_run594_injected_trajectory_v2 as previous
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    plan=read(OLD/'plan.json');previous.validate(plan)
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=True)
    frozen=[OLD/'plan.json',Path(__file__).resolve()]
    for shot in plan['shots']:
        frozen += [Path(shot['task']),Path(shot['prepared']),OLD/'runtime'/shot['tag']/'trajectory_states.pt',
                   OLD/'runtime'/shot['tag']/'result.json',OLD/shot['tag']/'trajectory_metadata.json']
    frozen += [Path(x['path']) for x in plan['code']]
    vae_root=Path(plan['model'])/'vae'
    frozen += [p for p in vae_root.iterdir() if p.is_file()]
    snapshot={'sources':[{'path':str(p.resolve()),'sha256':sha(p)} for p in frozen],
              'source_mode':'Reconstruct noised-reference operands with original VAE methods and saved noise/sigmas; require exact next-state equality at all injected steps.'}
    if (OUT/'inputs.json').exists():assert read(OUT/'inputs.json')==snapshot
    else:write(OUT/'inputs.json',snapshot)
    write(OUT/'status.json',{'stage':'vae_loading'})
    vae=AutoencoderKLQwenImage21.from_pretrained(str(vae_root),torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    # Bind the pristine pipeline's VAE-only methods; avoid loading unused VL/DiT.
    pipe=SimpleNamespace(vae=vae,vae_scale_factor=16,latent_channels=vae.config.z_dim,
        image_processor=VaeImageProcessor(vae_scale_factor=16,vae_latent_channels=vae.config.z_dim),
        _execution_device=torch.device('cuda'),_pack_latents=QwenImage21Pipeline._pack_latents,
        _unpack_latents=QwenImage21Pipeline._unpack_latents)
    pipe._encode_vae_image=MethodType(QwenImage21Pipeline._encode_vae_image,pipe)
    validation=[];files=[]

    def record(path,kind,**kw):
        files.append({'file':str(path.relative_to(OUT)),'source':str(path),'sha256':sha(path),'kind':kind,**kw})

    def decode(tensor,path,kind,**kw):
        path.parent.mkdir(parents=True,exist_ok=True)
        v7._decode(pipe,tensor.to('cuda'),1024,576).save(path)
        record(path,kind,**kw)

    for shot in plan['shots']:
        tag=shot['tag'];dest=OUT/tag;dest.mkdir(exist_ok=True)
        data=torch.load(OLD/'runtime'/tag/'trajectory_states.pt',map_location='cpu',weights_only=True)
        initial=data['rebuilt_noise'].to('cuda');sigmas=data['sigmas'].to('cuda')
        state={'device':torch.device('cuda'),'initial':initial,'sigmas':sigmas,'latent_width':64,'latent_height':36}
        args=SimpleNamespace(**shot['settings']);stored=read(shot['prepared'])
        targets,failures=v7._target_records(pipe,stored,state,args,read(shot['task'])['seed'])
        assert not failures
        expected={t['name']:t['active_injection_steps'] for t in shot['original_result']['targets']}
        assert {t['name']:t['active_injection_steps'] for t in targets}==expected
        tensors={'initial_noise':data['rebuilt_noise'],'sigmas':data['sigmas'],'targets':{},'noised_references':{},'injected_residuals':{}}
        decode(initial,dest/'reference_latents/shared_initial_noise.png','shared original initial noise')
        assert np.array_equal(np.asarray(Image.open(dest/'reference_latents/shared_initial_noise.png')),
                              np.asarray(Image.open(OLD/tag/'treatment/z_00.png')))
        for target in targets:
            role=target['name'];folder=dest/'reference_latents'/role;folder.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target['harmonized_reference'],folder/'reference_R.png');record(folder/'reference_R.png','Original harmonized aligned reference R')
            decode(target['reference_x0'],folder/'reference_clean_latent_decode.png','VAE decode of encoded reference E(R), not original R',role=role)
            tensors['targets'][role]={k:target[k].detach().cpu().clone() for k in ['reference_x0','alpha']}
            tensors['targets'][role]['effective_strength']=target['effective_strength']
            bbox=tuple(map(int,target['bbox']))
            for name in ['reference_R','reference_clean_latent_decode']:
                im=Image.open(folder/(name+'.png'));im.crop(bbox).save(folder/(name+'_face.png'))
                record(folder/(name+'_face.png'),'Native ROI display crop, no rescale or retouch',role=role,bbox=bbox)
        rows=[]
        log=read(OLD/'runtime'/tag/'result.json')['step_log']
        for step in log:
            if not step['injected']:continue
            i=step['step'];before=data['states']['treatment_euler_before'][i].to('cuda')
            sigma_next=sigmas[i+1].to(device='cuda',dtype=before.dtype)
            active=[t for t in targets if i<args.inject_start+t['active_injection_steps']]
            alphas=[];refs=[];tensors['noised_references'][i]={}
            for target in active:
                role=target['name'];folder=dest/'reference_latents'/role
                ref=sigma_next*initial+(1.0-sigma_next)*target['reference_x0']
                refs.append(ref);alphas.append(target['alpha']*target['effective_strength'])
                tensors['noised_references'][i][role]=ref.detach().cpu().clone()
                path=folder/f'step_{i:02d}_sigma_next_reference.png'
                decode(ref,path,'Actual noised reference operand reconstructed from saved initial noise and original VAE encoding',role=role,step=i,sigma_index=i+1,sigma_used=float(sigma_next))
                face=folder/f'step_{i:02d}_sigma_next_reference_face.png'
                Image.open(path).crop(tuple(map(int,target['bbox']))).save(face)
                record(face,'Native ROI display crop of noised reference; full canvas supplied separately',role=role,step=i,bbox=target['bbox'])
            stack=torch.stack(alphas,dim=0);total=stack.sum(dim=0)
            normalization=torch.where(total>1.0,total.reciprocal(),torch.ones_like(total))
            normalized=stack*normalization.unsqueeze(0);residual=torch.zeros_like(before)
            for alpha,ref in zip(normalized,refs):residual=residual+alpha*(ref-before)
            rebuilt=before+residual;expected_after=data['states']['treatment_z'][i+1].to('cuda')
            exact=torch.equal(rebuilt,expected_after)
            row={'step_zero_based':i,'next_state_index':i+1,'active_roles':[t['name'] for t in active],
                'sigma_saved_float32':float(sigmas[i+1]),'sigma_used_bfloat16':float(sigma_next),
                'next_latent_bit_exact':exact,'max_abs_latent_difference':float((rebuilt.float()-expected_after.float()).abs().max()),
                'residual_norm':float(residual.float().norm()),'recorded_residual_norm':step['residual_norm']}
            rows.append(row);write(dest/'verification_in_progress.json',rows)
            assert exact, f'Reconstructed operands do not reproduce original update: {tag}, {row}'
            tensors['injected_residuals'][i]=residual.detach().cpu().clone()
        tensor_path=dest/'reference_operand_states.pt';torch.save(tensors,tensor_path)
        for target in targets:
            folder=dest/'reference_latents'/target['name']
            steps=range(12,12+target['active_injection_steps'])
            panels=[('reference_R_face.png','Aligned R'),('reference_clean_latent_decode_face.png','Decoded E(R)')]
            panels += [(f'step_{i:02d}_sigma_next_reference_face.png',f'Noised ref: step{i}, sigma{i+1}') for i in steps]
            sheet=Image.new('RGB',(4*260,((len(panels)+3)//4)*280),'#eeeeee');draw=ImageDraw.Draw(sheet)
            for j,(file,label) in enumerate(panels):
                x=(j%4)*260;y=(j//4)*280;draw.text((x+5,y+5),label,fill='black')
                im=ImageOps.contain(Image.open(folder/file).convert('RGB'),(250,245));sheet.paste(im,(x+(260-im.width)//2,y+27+(245-im.height)//2))
            sheet.save(folder/'reference_noise_overview.jpg',quality=95);record(folder/'reference_noise_overview.jpg','Display overview of actual reference operands')
        verification={'shot':shot['key'],'per_step':rows,'all_next_latents_bit_exact':all(r['next_latent_bit_exact'] for r in rows),
            'source_states_sha256':sha(OLD/'runtime'/tag/'trajectory_states.pt'),'operand_states_sha256':sha(tensor_path),
            'source_final_sha256':sha(OLD/tag/'treatment.png'),'schedule':expected,
            'vae_only_reconstruction_not_original_capture':True,'new_transformer_steps':0,'new_noise_draws':0}
        write(dest/'complete.json',verification);validation.append(verification)
        print('Exact reference operands recovered:',tag,expected,flush=True)
    spec=importlib.util.spec_from_file_location('pack_skill','/root/.codex/skills/movie-firstframe-figure-assets/scripts/pack_trajectory.py')
    pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
    manifest=pack.pack(OLD,DELIVERY)
    for row in files:
        source=Path(row['source']);target=DELIVERY/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);assert sha(target)==row['sha256']
        manifest['files'].append({**row,'source':str(source)})
    for shot in plan['shots']:
        tag=shot['tag']
        for source,name in [(OLD/tag/'trajectory_metadata.json','trajectory_metadata.json'),(OUT/tag/'complete.json','reference_operand_verification.json')]:
            shutil.copy2(source,DELIVERY/tag/name)
    manifest['total_images']=len(manifest['files']);manifest['reference_operand_extension_images']=len(files)
    manifest['source_reference_operand_root']=str(OUT);write(DELIVERY/'image_manifest.json',manifest)
    write(DELIVERY/'reference_operand_verification.json',validation)
    text='''# 新版3:4、3:5完整去噪图片＋参考latent加噪过程

包含此前全部148张注入流程图片，并补齐3:4 Leo、3:5 Leo与Marcus的参考编码/加噪图；没有Control图片/视频。

每个shot的reference_latents/角色名/：

- reference_R.png：原始对齐、颜色协调的3D参考图R。
- reference_clean_latent_decode.png：原VAE编码E(R)再解码的图，区别于原参考R。
- step_12_sigma_next_reference.png：step12实际使用的参考latent r_13解码图；其他步同理。此图是参考分支，不是主生成轨迹的z_13。
- *_face.png：相同原始角色bbox的局部显示图；完整画布同时保留，不重绘、不修脸、不调色。
- reference_noise_overview.jpg：原R、编码重建及各步带噪参考的预览拼图。
- reference_latents/shared_initial_noise.png：本shot原始共享噪声的解码图。

原有treatment/仍保存实际z_00..z_20与pred_x0_00..pred_x0_19；injection/为主轨迹Euler-before与actual-after；guidance/为原3D渲染、对齐、颜色协调和mask。

公式：r_(i+1)=sigma_(i+1)*epsilon_0+(1-sigma_(i+1))*E(R)。epsilon_0取自刚才重放保存的初始噪声；无新噪声采样。系数按原代码转BF16后计算。再按原mask/强度/多人重叠归一化计算残差，与保存的Euler-before相加。

此次是从保存噪声/时间步和原参考通过同一VAE重建参考分支，不冒充原生成时直接截取。已逐步核验：3:4六次、3:5六次注入更新（其中首次包含Leo与Marcus），重算下一latent全部逐元素、逐比特等于此前保存的actual-after。共13份实际带噪参考（Leo6+Leo6+Marcus1）。最终图仍使用之前已精确复现的原结果；未重跑DiT、视频、API或评测。

原始E(R)、mask/alpha、每步r、残差、sigmas及共享噪声张量保存在源目录run594_noised_reference_reconstruction_20261101/shot_*/reference_operand_states.pt，不放入图片ZIP。核验见reference_operand_verification.json；原件/派生图片SHA见image_manifest.json。

实际含噪latent的VAE解码可能呈现强烈色噪，这是原数值的解码表现，不等于清晰预测pred_x0。绘图请保持名称区分。

使用movie-firstframe-figure-assets skill（/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md）。
'''
    (DELIVERY/'README.md').write_text(text)
    archive=DELIVERY.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for file in sorted(DELIVERY.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(DELIVERY.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    for row in snapshot['sources']:assert sha(row['path'])==row['sha256']
    write(OUT/'status.json',{'stage':'complete','total_images':manifest['total_images'],
        'added_reference_images':len(files),'delivery':str(DELIVERY),'zip':str(archive),
        'all12_injection_updates_latent_bit_exact':True,'actual_noised_role_operands':13})
    print(read(OUT/'status.json'),flush=True)


if __name__=='__main__': main()
