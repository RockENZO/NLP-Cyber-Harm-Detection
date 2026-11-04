"""Render a project overview from the checked-in evaluation report (Pillow)."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
report=json.loads((ROOT/'evaluation/reports/nine_class_test_20260930.json').read_text())
r=report['results']['selected']; m=r['nine_class']
W,H=1600,900
im=Image.new('RGB',(W,H),'#0b1424');d=ImageDraw.Draw(im)
FONT_DIRS=['/System/Library/Fonts/Supplemental','/System/Library/Fonts','/Library/Fonts',
           '/usr/share/fonts/truetype/dejavu','/usr/share/fonts/truetype/liberation','C:/Windows/Fonts']
def font(n,b=False):
    names=(['Arial Bold.ttf','Arial-Bold.ttf','DejaVuSans-Bold.ttf','LiberationSans-Bold.ttf'] if b
           else ['Arial.ttf','DejaVuSans.ttf','LiberationSans.ttf'])
    for directory in FONT_DIRS:
        for name in names:
            path=Path(directory)/name
            if path.is_file():return ImageFont.truetype(str(path),n)
    try:return ImageFont.load_default(size=n)
    except TypeError:return ImageFont.load_default()
def text(x,y,s,n=25,c='#eaf2ff',b=False):d.text((x,y),s,font=font(n,b),fill=c)
def box(x,y,w,h,fill='#142239',outline='#2c405a',r=20):d.rounded_rectangle((x,y,x+w,y+h),r,fill=fill,outline=outline,width=2)
text(70,40,'NLP / CYBER HARM DETECTION',23,'#6ce5cf',True)
text(70,86,'From raw messages to nine scam categories',51,b=True)
text(70,154,'A reproducible text-classification study • word + character features • frozen evaluation',27,'#acbed5')
steps=[('01 / CURATE','194,913 input records','Normalize, deduplicate, group templates'),('02 / REPRESENT','Word + character TF-IDF','Word 1–2 grams • character 3–5 grams'),('03 / CLASSIFY','Balanced LinearSVC','Validation-selected margin adjustment')]
for j,(k,title,desc) in enumerate(steps):
 x=70+j*500;box(x,225,460,182);text(x+24,245,k,20,'#6ce5cf',True);text(x+24,289,title,28,b=True);text(x+24,345,desc,20,'#b6c8de')
 if j<2:
  d.line((x+466,315,x+493,315),fill='#6ce5cf',width=4);d.polygon([(x+494,315),(x+484,308),(x+484,322)],fill='#6ce5cf')
text(70,450,'FROZEN INTERNAL TEST',22,'#6ce5cf',True)
text(70,487,'18,261 records • template-grouped split',28,'#d4e2f3')
vals=[(f'{m["macro_f1"]:.3f}','Macro F1'),(f'{m["accuracy"]*100:.2f}%','Accuracy'),(f'{r["binary"]["legitimate_false_positive_rate"]*100:.2f}%','Legitimate false-positive rate')]
for j,(v,l) in enumerate(vals):
 x=70+j*500;box(x,545,460,132);text(x+24,560,v,49,'#ffffff',True);text(x+24,627,l,23,'#b6c8de')
box(70,720,1460,110,fill='#1d2838',outline='#465467');text(94,737,'Scope matters',24,'#ffd08b',True)
text(94,774,'Internal benchmark, not unseen-source validation. Job-scam recall: 48.3% on 60 examples.',25,'#dce4ee')
text(70,853,'Source: evaluation/reports/nine_class_test_20260930.json  •  RockENZO / NLP-Cyber-Harm-Detection',20,'#8ca3be')
im.save(ROOT/'docs/preview/nlp-project-preview.png')
