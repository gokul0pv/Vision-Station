"""Accuracy of the geometry tools against exactly known sub-pixel ground truth.

    python bench_geometry.py

Edges are drawn at 8x scale and downsampled, so their true position is known to
1/8 pixel, then blurred and noised. Reports Find Circle and Find Line error, what
outlier rejection buys when a chip bites into an edge, and whether a distance
between two holes stays constant as the part moves and turns.
"""
import sys, math, cv2, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from engine import Context
from vstools import make_tool
S=8
def render_circle(cx,cy,r,W=320,H=240,blur=1.2,noise=4.0,seed=0):
    big=np.full((H*S,W*S),60,np.uint8)
    cv2.circle(big,(int(round(cx*S+S/2-0.5)),int(round(cy*S+S/2-0.5))),int(round(r*S)),190,-1,cv2.LINE_AA)
    im=cv2.resize(big,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
    im=cv2.GaussianBlur(im,(0,0),blur)+np.random.default_rng(seed).normal(0,noise,im.shape)
    return cv2.cvtColor(np.clip(im,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
def render_line(angle_deg,y_at_centre,W=320,H=240,blur=1.2,noise=4.0,seed=0):
    # half-plane below the line is bright
    big=np.full((H*S,W*S),60,np.uint8)
    a=math.radians(angle_deg); cxp,cyp=W/2.0,y_at_centre
    d=np.array([math.cos(a),math.sin(a)]); L=2000
    p0=np.array([cxp,cyp])-d*L; p1=np.array([cxp,cyp])+d*L
    poly=np.array([p0,p1,p1+np.array([0,3000]),p0+np.array([0,3000])])*S+S/2-0.5
    cv2.fillPoly(big,[np.round(poly).astype(np.int32)],190,cv2.LINE_AA)
    im=cv2.resize(big,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
    im=cv2.GaussianBlur(im,(0,0),blur)+np.random.default_rng(seed).normal(0,noise,im.shape)
    return cv2.cvtColor(np.clip(im,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
def run(t,params,img):
    ctx=Context(reference=img)
    r=make_tool({"id":"g","type":t,"params":dict({"enabled":True,"critical":True,"angle":0.0,"shape":"rect"},**params)}).run(img,ctx)
    return r,ctx
rng=np.random.default_rng(42)
ce,re=[],[]
for i in range(20):
    cx,cy,rad=160+rng.uniform(-6,6),120+rng.uniform(-6,6),rng.uniform(35,55)
    img=render_circle(cx,cy,rad,seed=i)
    R=rad*1.35
    r,_=run("find_circle",{"roi":[int(cx-R),int(cy-R),int(2*R),int(2*R)],"search_inner":0.5},img)
    v=r.values; ce.append(math.hypot(v["cx"]-cx,v["cy"]-cy)); re.append(abs(v["radius_px"]-rad))
print("FIND CIRCLE (20 random circles, blur 1.2, noise 4):")
print("   centre error  mean %.3f px  worst %.3f px"%(np.mean(ce),np.max(ce)))
print("   radius error  mean %.3f px  worst %.3f px"%(np.mean(re),np.max(re)))
ae,oe=[],[]
for i in range(20):
    ang,yc=rng.uniform(-8,8),120+rng.uniform(-10,10)
    img=render_line(ang,yc,seed=100+i)
    r,ctx=run("find_line",{"roi":[40,70,240,100]},img)
    f=ctx.share["g"]; ae.append(abs(r.values["angle_deg"]-ang))
    c=np.array(f["point"]); d=np.array(f["direction"])
    ycalc=c[1]+d[1]*((160-c[0])/d[0]); oe.append(abs(ycalc-yc))
print("FIND LINE (20 random lines, +-8 deg):")
print("   angle error   mean %.4f deg worst %.4f deg"%(np.mean(ae),np.max(ae)))
print("   offset error  mean %.3f px  worst %.3f px"%(np.mean(oe),np.max(oe)))

print("\nOUTLIER: a chip on the circle's edge (bright notch cut into the dark rim)")
cx,cy,rad=160.3,119.6,45.0
img=render_circle(cx,cy,rad,seed=7)
cv2.circle(img,(int(cx+rad),int(cy)),9,(60,60,60),-1)     # a bite out of the edge
R=rad*1.35; roi=[int(cx-R),int(cy-R),int(2*R),int(2*R)]
for ign in (0,3):
    r,_=run("find_circle",{"roi":roi,"search_inner":0.5,"num_to_ignore":ign},img)
    v=r.values
    print("   ignore %d: centre error %.3f px  radius error %.3f px"%(ign,math.hypot(v["cx"]-cx,v["cy"]-cy),abs(v["radius_px"]-rad)))

print("\nDISTANCE between two holes on a part that moves and turns (true spacing 150.00 px)")
def part(dx,dy,deg,seed):
    W,H=480,320; big=np.full((H*S,W*S),60,np.uint8)
    for hx in (165,315):
        cv2.circle(big,(hx*S,160*S),30*S,190,-1,cv2.LINE_AA)
    im=cv2.resize(big,(W,H),interpolation=cv2.INTER_AREA)
    M=cv2.getRotationMatrix2D((W/2,H/2),deg,1.0); M[0,2]+=dx; M[1,2]+=dy
    im=cv2.warpAffine(im,M,(W,H),borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    im=cv2.GaussianBlur(im,(0,0),1.0)+np.random.default_rng(seed).normal(0,3,im.shape)
    return cv2.cvtColor(np.clip(im,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
ref=part(0,0,0,0)
from schema import defaults
fx={"id":"fx","type":"align","params":dict(defaults("align"),roi=[100,100,280,120],method="template",
    min_score=0.4,angle_range=[-8,8],angle_step=1.0)}
errs=[]
for i,(dx,dy,deg) in enumerate([(0,0,0),(20,-12,0),(-25,15,4),(15,10,-6),(30,-20,7)]):
    img=part(dx,dy,deg,i+1); ctx=Context(reference=ref)
    make_tool(fx).run(img,ctx)
    for hid,hx in (("hole_a",165),("hole_b",315)):
        make_tool({"id":hid,"type":"find_circle","params":dict({"enabled":True,"critical":True,"angle":0.0,"shape":"ellipse"},
                   roi=[hx-42,118,84,84],search_inner=0.5)}).run(img,ctx)
    r=make_tool({"id":"d","type":"distance","params":{"enabled":True,"critical":True,"angle":0.0,"shape":"rect",
                 "first":"hole_a","second":"hole_b"}}).run(img,ctx)
    errs.append(abs(r.values["value_px"]-150.0))
    print("   moved %+3d,%+3d rotated %+d deg -> %.3f px"%(dx,dy,deg,r.values["value_px"]))
print("   worst error %.3f px"%max(errs))
