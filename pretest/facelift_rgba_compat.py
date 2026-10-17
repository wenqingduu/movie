"""Normalize only FaceLift's RGBA fallback to its documented white RGB background.

No detector, crop, reference, model, seed or sampler is changed. RGB inputs are
passed through by identity. The original FaceLift sources remain untouched.
"""
import hashlib
import json
import os
from pathlib import Path
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def verify_contract():
    root=Path(os.environ['ENTITYBENCH_RGBA_RECOVERY_ROOT'])
    contract=json.loads((root/'runtime/rgba_recovery_implementation.json').read_text())
    for path,expected in contract['sources'].items():
        assert sha(path)==expected,f'Changed recovery implementation: {path}'

def rgb_fallback(image,audit_dir=None):
    if not isinstance(image,Image.Image) or image.mode!='RGBA':return image
    white=Image.new('RGBA',image.size,(255,255,255,255))
    result=Image.alpha_composite(white,image).convert('RGB')
    if audit_dir is not None:
        directory=Path(audit_dir);directory.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(image.tobytes()).hexdigest()
        before=directory/(key+'.rgba.png');after=directory/(key+'.rgb.png')
        for path,pixels in [(before,image),(after,result)]:
            if path.exists():
                saved=Image.open(path);assert saved.mode==pixels.mode and saved.size==pixels.size and saved.tobytes()==pixels.tobytes()
            else:pixels.save(path)
        record=dict(kind='facelift_rgba_fallback_format_compatibility',background=[255,255,255],
            before=str(before),before_sha256=sha(before),after=str(after),after_sha256=sha(after),
            source_reference=os.environ['ENTITYBENCH_RGBA_SOURCE_REFERENCE'],
            source_reference_sha256=os.environ['ENTITYBENCH_RGBA_SOURCE_SHA256'],
            policy='Original FaceLift fallback and dimensions; white alpha compositing only; no crop/detector/seed/model changes')
        assert sha(record['source_reference'])==record['source_reference_sha256']
        path=directory/(key+'.json')
        if path.exists():assert json.loads(path.read_text())==record
        else:path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return result

def install():
    verify_contract()
    from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
    cls=StableUnCLIPImg2ImgPipeline
    assert not getattr(cls,'_entitybench_rgba_compat',False)
    original=cls.__call__
    def call(self,image,*args,**kwargs):
        directory=os.environ['ENTITYBENCH_RGBA_AUDIT_DIR']
        if isinstance(image,(list,tuple)):
            converted=[rgb_fallback(item,directory) for item in image]
            image=converted if any(a is not b for a,b in zip(converted,image)) else image
        else:image=rgb_fallback(image,directory)
        return original(self,image,*args,**kwargs)
    cls.__call__=call;cls._entitybench_rgba_compat=True

