import json, os, shutil, glob, subprocess
SRC="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
DST="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep_video"
VIDMAP={"image":"observation.images.exterior","extra_view_image":"observation.images.wrist"}

info=json.load(open(SRC+"/meta/info.json"))
fps=float(info["fps"])
feats={}
for k,v in info["features"].items():
    if k in VIDMAP:
        nk=VIDMAP[k]; h,w,c=v["shape"]
        feats[nk]={"dtype":"video","shape":[h,w,c],"names":["height","width","channel"],
                   "info":{"video.fps":fps,"video.height":h,"video.width":w,"video.channels":c,
                           "video.codec":"h264","video.pix_fmt":"yuv420p",
                           "video.is_depth_map":False,"has_audio":False}}
    else:
        feats[k]=v
info["features"]=feats
info["total_videos"]=info["total_episodes"]*len(VIDMAP)
info["video_path"]="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
json.dump(info, open(DST+"/meta/info.json","w"), indent=4)

shutil.copy(SRC+"/meta/tasks.jsonl", DST+"/meta/tasks.jsonl")
shutil.copy(SRC+"/meta/episodes.jsonl", DST+"/meta/episodes.jsonl")
# stats 里的图像键改名
with open(DST+"/meta/episodes_stats.jsonl","w") as out:
    for l in open(SRC+"/meta/episodes_stats.jsonl"):
        d=json.loads(l)
        d["stats"]={VIDMAP.get(k,k):v for k,v in d["stats"].items()}
        out.write(json.dumps(d)+"\n")

print("codebase_version :", info["codebase_version"])
print("video_path       :", info["video_path"])
print("total_videos     :", info["total_videos"])
print("视频特征         :", [k for k,v in feats.items() if v["dtype"]=="video"])
print("parquet 列        :", [k for k,v in feats.items() if v["dtype"]!="video"])
nv=len(glob.glob(DST+"/videos/chunk-000/*/*.mp4")); npq=len(glob.glob(DST+"/data/chunk-000/*.parquet"))
print("mp4 数 :", nv, "| parquet 数 :", npq)
p=DST+"/videos/chunk-000/observation.images.exterior/episode_000000.mp4"
print("抽查:", subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
      "stream=codec_name,pix_fmt,width,height,nb_frames","-of","csv=p=0",p],
      capture_output=True,text=True).stdout.strip())
print("总大小:", subprocess.run(["du","-sh",DST],capture_output=True,text=True).stdout.split()[0])
