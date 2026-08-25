"""FR3 正运动学(modified DH),并用 5 月标定 anchor 里的真值验证。"""
import numpy as np, json, yaml

# Franka FR3 / Panda modified-DH: (a, d, alpha) per joint, flange = link8
MDH = [(0.0,0.333,0.0),(0.0,0.0,-np.pi/2),(0.0,0.316,np.pi/2),
       (0.0825,0.0,np.pi/2),(-0.0825,0.384,-np.pi/2),(0.0,0.0,np.pi/2),
       (0.088,0.0,np.pi/2)]
D_FLANGE = 0.107   # link7 -> flange(link8)

def fk(q):
    """q: (...,7) -> T_base_flange (...,4,4)"""
    q=np.atleast_2d(q); N=len(q)
    T=np.tile(np.eye(4),(N,1,1))
    for i,(a,d,al) in enumerate(MDH):
        ct,st=np.cos(q[:,i]),np.sin(q[:,i]); ca,sa=np.cos(al),np.sin(al)
        A=np.zeros((N,4,4))
        A[:,0,0]=ct;      A[:,0,1]=-st;     A[:,0,2]=0;   A[:,0,3]=a
        A[:,1,0]=st*ca;   A[:,1,1]=ct*ca;   A[:,1,2]=-sa; A[:,1,3]=-d*sa
        A[:,2,0]=st*sa;   A[:,2,1]=ct*sa;   A[:,2,2]=ca;  A[:,2,3]=d*ca
        A[:,3,3]=1
        T=T@A
    F=np.tile(np.eye(4),(N,1,1)); F[:,2,3]=D_FLANGE
    return T@F

if __name__=="__main__":
    a=yaml.safe_load(open("/home/franka_desktop/work/franka_r3/code/anchors/eth_right_cam.yaml"))
    q=np.array(a["joint_q"]); gt=np.array(a["ee_translation"]); gtT=np.array(a["ee_T_4x4_base"])
    T=fk(q)[0]
    print("=== FK 验证(对 5 月标定 anchor 的真值)===")
    print("  FK 算出的末端位置 :", T[:3,3].round(4))
    print("  anchor 记录的真值 :", gt.round(4))
    print("  位置误差          : %.2f mm" % (np.linalg.norm(T[:3,3]-gt)*1000))
    R=T[:3,:3].T@gtT[:3,:3]
    ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    print("  姿态误差          : %.2f 度" % ang)
