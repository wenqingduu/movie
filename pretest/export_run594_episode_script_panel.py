"""Export the selected run594 script text and editable paper-figure panel.

Uses dataset text verbatim. Styling annotates text; it does not rewrite prompts
or edit any generated portrait/first-frame pixel. No inference is performed.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import zipfile

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path('/root/autodl-tmp/movie')
EID = '00048e6c-5e69-3381-9079-c43db78ed8f0__run594__i75_j98__T300'
PACKAGE = ROOT/'outputs/run594_i75_j98_identity_examples_20261101'
OUT = PACKAGE/'episode_script'
DATASET = ROOT/'benchmarks/entitybench/data/scripts'/f'{EID}.json'
RUN = ROOT/'outputs/entitybench_qwen140_multiface_remaining126_20261030/experiments'/EID
NORMAL = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
COLORS = {'character':'#df433c','appearance':'#e58b06','camera':'#1671cc','location':'#079564'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Panel:
    def __init__(self):
        self.items = []; self.width = 760
    def rectangle(self,x,y,w,h,fill='white',stroke='#ebcba6',radius=17,dash=False):
        self.items.append(('rect',x,y,w,h,fill,stroke,radius,dash))
    def text(self,x,y,text,size=18,fill='#29313d',bold=False):
        self.items.append(('text',x,y,text,size,fill,bold))
    def paragraph(self,text,x,y,width,size=18,lineheight=25,fill='#29313d',bold=False,highlight=False):
        font = ImageFont.truetype(BOLD if bold else NORMAL,size)
        # Keep the original string, including spaces. Highlight phrase spans
        # before wrapping; line breaks are only presentation metadata.
        matches = []
        if highlight:
            terms = {
                'character':['Leo','Erik','Marcus'],
                'appearance':['long blonde hair','short dark hair','camo jacket','patterned scarf',
                    'light-colored collared shirt','dark hoodie','grey t-shirt','blue tartan scarf'],
                'camera':['Medium close-up','View from outside a parked car','in profile','passenger-side window','passenger side window'],
                'location':['Oakshott Cafe and Yard'],
            }
            patterns = sorted(((term,kind) for kind,terms_ in terms.items() for term in terms_),key=lambda p:-len(p[0]))
            for term,kind in patterns:
                for m in re.finditer(re.escape(term),text,flags=re.I):
                    if not any(m.start()<b and m.end()>a for a,b,_ in matches): matches.append((m.start(),m.end(),kind))
        line_x = x; offset = 0
        for token in re.findall(r'\s+|\S+',text):
            color = fill
            if highlight:
                color = next((COLORS[k] for a,b,k in matches if a<=offset<b),fill)
            length = font.getlength(token)
            if line_x + length > x+width and token.strip() and line_x>x:
                line_x=x; y+=lineheight
            if token.isspace() and line_x==x:
                offset+=len(token);continue
            if token.strip(): self.text(line_x,y,token,size,color,bold)
            line_x+=length;offset+=len(token)
        return y+lineheight
    def export(self,height):
        # Every format uses the same primitive/text layout. PNG is a rendering
        # of this panel, not a transformation of the experimental images.
        scale=2;im=Image.new('RGB',(self.width*scale,height*scale),'white');draw=ImageDraw.Draw(im)
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{height}" viewBox="0 0 {self.width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>']
        pdfmetrics.registerFont(TTFont('FigureNormal',NORMAL));pdfmetrics.registerFont(TTFont('FigureBold',BOLD))
        pdf=pdf_canvas.Canvas(str(OUT/'episode_script_panel.pdf'),pagesize=(self.width,height))
        for item in self.items:
            if item[0]=='rect':
                _,x,y,w,h,fill,stroke,radius,dash=item
                draw.rounded_rectangle((round(x*scale),round(y*scale),round((x+w)*scale),round((y+h)*scale)),radius=round(radius*scale),fill=fill,outline=stroke,width=3)
                dashattr=' stroke-dasharray="9 7"' if dash else ''
                svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dashattr}/>')
                pdf.setFillColor(fill);pdf.setStrokeColor(stroke);pdf.setLineWidth(1.5);pdf.setDash([9,7] if dash else [])
                pdf.roundRect(x,height-y-h,w,h,radius,stroke=1,fill=1);pdf.setDash([])
            else:
                _,x,y,text,size,fill,bold=item
                font=ImageFont.truetype(BOLD if bold else NORMAL,size*scale)
                draw.text((round(x*scale),round(y*scale)),text,font=font,fill=fill,anchor='lt')
                family='DejaVu Sans';weight='bold' if bold else 'normal'
                svg.append(f'<text x="{x:.3f}" y="{y+size:.3f}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>')
                pdf.setFont('FigureBold' if bold else 'FigureNormal',size);pdf.setFillColor(fill);pdf.drawString(x,height-y-size,text)
        svg.append('</svg>');(OUT/'episode_script_panel.svg').write_text('\n'.join(svg)+'\n')
        im.save(OUT/'episode_script_panel.png',dpi=(300,300));pdf.showPage();pdf.save()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads(DATASET.read_text());entities=data['entity_descriptions']
    characters=[v for k,v in entities.items() if k.startswith('Character ')]
    names={c['name']:c for c in characters};locations={v['name']:v for k,v in entities.items() if k.startswith('Location ')}
    keys=['2:2','3:4','3:5'];shots=[]
    for key in keys:
        scene,index=[int(i) for i in key.split(':')]
        record=next(s for s in data['scenes'] if s['scene_num']==scene)
        schedule=data['entity_schedule'][key]
        shots.append({'shot':key,'scene_num':scene,'shot_num':index,'cut':record['cut'][index-1],
            'locations':[locations[n] for n in schedule['places']], 'entity_schedule':schedule,
            'action_description':record['action_descriptions'][index-1],'dataset_video_prompt':record['video_prompts'][index-1]})
    references={}
    for name in ['Leo','Erik']:
        path=RUN/'references/run594'/f'{name.lower()}.json';references[name]=json.loads(path.read_text())
        shutil.copy2(path,OUT/f'{name.lower()}_portrait_generation.json')
    record={'episode_id':EID,'characters':characters,'story_overview':data['story_overview'],'selected_shots':shots,
            'portrait_generation_records':references,'dataset_path':str(DATASET),'dataset_sha256':sha(DATASET),
            'verbatim_fields':'character descriptions, story_overview, action_description, dataset_video_prompt, location descriptions and portrait prompts',
            'display_note':'Panel displays original dataset character one-liners. Full actual portrait-generation prompts are exported separately.'}
    (OUT/'episode_script.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    md=['# (a) Episode Script','',f'Episode: `{EID}`','', '## Characters','']
    for name in ['Leo','Erik']+[c['name'] for c in characters if c['name'] not in ('Leo','Erik')]:md.append(f"**{name}:** {names[name]['description']}  ")
    md+=['','## Story Overview','',data['story_overview']]
    for shot in shots:
        md+=['',f"## Scene {shot['scene_num']} — Shot {shot['shot_num']}",'']
        for loc in shot['locations']:md+=[f"**{loc['name']}:** {loc['description']}",'']
        md += [shot['action_description']]
    md+=['','## Actual portrait-generation prompts','',
         '以下为本次人物参考图实际生成提示词，保留完整原文。上方Characters使用数据集一句话描述。']
    for name in ['Leo','Erik']:md+=['',f'### {name}','',references[name]['inputs']['prompt']]
    md+=['','## Provenance','',
         '角色描述、故事概述、地点描述和shot动作均逐字取自本episode数据，未另写剧情；自动换行和颜色只属于绘图排版。',
         '原始dataset_video_prompt完整保存在episode_script.json；原实际首帧生成提示词仍在上级各shot的prompt.txt，两者可能不同，未互相替换。',
         '角色红色、外观橙色、机位蓝色、地点绿色；这些颜色为展示标注。PNG/PDF/SVG只包含脚本文字，未修改实验图片。']
    (OUT/'episode_script.md').write_text('\n'.join(md)+'\n')
    plain=['(a) Episode Script','','Characters']+[f"{c['name']}: {c['description']}" for c in [names['Leo'],names['Erik']]+[c for c in characters if c['name'] not in ('Leo','Erik')]]
    plain+=['','Story Overview',data['story_overview']]
    for shot in shots:
        plain+=['',f"Scene {shot['scene_num']} — Shot {shot['shot_num']}"]
        plain += [f"{loc['name']}: {loc['description']}" for loc in shot['locations']]+[shot['action_description']]
    (OUT/'episode_script.txt').write_text('\n'.join(plain)+'\n')
    # Build content first to obtain card heights; add backgrounds underneath.
    panel=Panel(); y=76
    panel.text(230,26,'(a) Episode Script',25,bold=True)
    left=40; width=680; px=left+18; content_width=width-36
    def card(title,body):
        nonlocal y
        top=y;start=len(panel.items);panel.text(px,y+13,title,21,'#d77a00',True);y+=48
        body();y+=15
        panel.items.insert(start,('rect',left,top,width,y-top,'white','#ebcba6',17,False));y+=16
    def character_body():
        nonlocal y
        for name in ['Leo','Erik']:
            start=len(panel.items);top=y-6
            y=panel.paragraph(f"{name}: {names[name]['description']}",px+8,y,content_width-16,19,26,highlight=True)+9
            panel.items.insert(start,('rect',px,top,content_width,y-top-3,'#fff5eb','#ffb78f',13,False))
        for c in characters:
            if c['name'] not in ('Leo','Erik'):
                y=panel.paragraph(f"{c['name']}: {c['description']}",px,y,content_width,16,22,'#929bab')+2
    card('Characters',character_body)
    def story_body():
        nonlocal y
        y=panel.paragraph(data['story_overview'],px,y,content_width,18,25)
    card('Story Overview',story_body)
    for shot in shots:
        def shot_body(shot=shot):
            nonlocal y
            for loc in shot['locations']:
                y=panel.paragraph(f"{loc['name']}: {loc['description']}",px,y,content_width,18,25,COLORS['location'])+10
            y=panel.paragraph(shot['action_description'],px,y,content_width,18,25,highlight=True)+16
            panel.text(left+width-205,y,'Shot description p'+shot['shot'].replace(':','_'),14,'#929bab');y+=20
        card(f"Scene {shot['scene_num']} — Shot {shot['shot_num']}",shot_body)
    legend_y=y+1
    for x,(kind,color) in zip([73,230,405,550],COLORS.items()):
        panel.rectangle(x,legend_y+5,11,11,color,color,0);panel.text(x+17,legend_y,kind,14,'#667085')
    height=int(y+50);panel.items.insert(0,('rect',18,12,724,height-24,'#fffbf6','#929bab',20,True));panel.export(height)
    readme='''# run594脚本文字与绘图面板

按提供的(a) Episode Script格式组织：Characters、Story Overview、三个Scene–Shot卡片。

- episode_script.md：可复制的完整内容，附实际Leo/Erik人物参考图生成提示词。
- episode_script.txt：纯文本，仅主面板内容。
- episode_script.json：逐字原文、完整dataset_video_prompt、调度、来源路径与SHA。
- episode_script_panel.svg：可编辑矢量版，文本保持为text元素。
- episode_script_panel.pdf：矢量PDF，嵌入字体。
- episode_script_panel.png：2倍尺寸预览。

episode为run594/i75_j98；Leo对应3:4和3:5，Erik对应2:2。角色列表保留原集全部8人；主面板突出Leo/Erik。Story Overview与shot内容均为原数据原文，未编写新剧情。图内Characters采用原数据一句话描述，实际人物素材生成完整提示词另行保留。

采用movie-firstframe-figure-assets skill：/root/.codex/skills/movie-firstframe-figure-assets/SKILL.md。无需GPU或API，没有重生成/修改人物图或首帧图。
'''
    (OUT/'README.md').write_text(readme)
    copied=OUT/'source_episode.json';shutil.copy2(DATASET,copied);assert sha(copied)==sha(DATASET)
    inventory=[{'file':str(p.relative_to(OUT)),'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='files_manifest.json']
    (OUT/'files_manifest.json').write_text(json.dumps({'episode_id':EID,'source_sha256':sha(DATASET),'files':inventory},indent=2)+'\n')
    for shot in shots:
        scene=next(s for s in data['scenes'] if s['scene_num']==shot['scene_num'])
        assert shot['action_description']==scene['action_descriptions'][shot['shot_num']-1]
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.name))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'folder':str(OUT),'panel_size':[panel.width,height],'shots':keys,'zip':str(archive)},ensure_ascii=False))


if __name__=='__main__':main()
