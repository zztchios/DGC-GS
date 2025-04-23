import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
import cv2
import torch
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from skimage.transform import resize
import os
import shutil
from argparse import ArgumentParser

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap depth")
parser.add_argument("--scan", "-s", required=True, type=str)
parser.add_argument("--input", "-i", required=True, type=str)
parser.add_argument("--output", "-o", required=True, type=str)
args = parser.parse_args()

# input_dir='/media/pc/D/zzt/DNGaussian/DNGaussian_Dataset/DTU/'
input_dir=args.input
output_dir=args.output

dir_now=None
file_now=None

def dbscan(t, eps=0.0005, min_samples=1):
    t = t.cpu().reshape(-1, 1)
    scan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = scan.fit(t)

    counts = Counter(labels.labels_)
    max_element = max(counts, key=counts.get)
    cnt = counts[max_element]

    x = t[np.where(labels.labels_==max_element)].reshape(-1)
    mean = torch.mean(x)
    # 计算标准差
    std_deviation = torch.std(x)
    # 计算方差
    variance = torch.var(x)
    return mean,std_deviation,x

def linear_fit(final_res):
    arra=np.array(final_res)
    X=arra[:,1].reshape(-1,1)
    y=arra[:,0]
    model = LinearRegression()
    model.fit(X, y)
    return model

def poly_fit(final_res):
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=True),LinearRegression())   
    arra=np.array(final_res)
    X=arra[:,1].reshape(-1,1)
    y=arra[:,0]
    model.fit(X, y)
    return model

def data_process(depth_gt,depth_estimate,mask_image):
    global dir_now, file_now, input_dir, output_dir
    mask_image = resize(mask_image, (298, 398))
    depth_estimate = resize(depth_estimate, (298, 398))
    #depth_estimate 中的异常点也计入mask_image
    print(f'depth_estimate:{depth_estimate.shape}, {mask_image.shape}')
    mask_image[depth_estimate<1e-5]=0
    depth_estimate0=depth_estimate.copy()
    depth_gt0=depth_gt.copy()
    #用mask对colmap和anything进行过滤
    depth_estimate=depth_estimate*mask_image/255
    depth_gt=depth_gt*mask_image/255

    #计算depth_estimate和depth_gt中非0数字集合的分位数，用于筛掉离群点
    depth_gt_nonzero=depth_gt[np.nonzero(depth_gt)]
    depth_estimate_nonzero=depth_estimate[np.nonzero(depth_estimate)]

    depth_estimate_max,depth_estimate_min=np.percentile(depth_estimate_nonzero,95),np.percentile(depth_estimate_nonzero,5)
    depth_gt_max,depth_gt_min=np.percentile(depth_gt_nonzero,95),np.percentile(depth_gt_nonzero,5)

    depth_estimate_range=np.linspace(depth_estimate_min-1e-3,depth_estimate_max+1e-3, 20)
    print('The range of depth before and after 99 filtering',[np.min(depth_estimate0),np.max(depth_estimate0)],[depth_estimate_min,depth_estimate_max])


    mse = np.mean((depth_estimate - depth_gt) ** 2)
    print('The mse before scaling', mse)  

    final_result=[]
    final_result1=[]
    for k in range(len(depth_estimate_range)-1):
        scale_list=[]
        for i in range(depth_estimate.shape[0]):
            for j in range(depth_estimate.shape[1]):
                if mask_image[i][j]!=0 and depth_gt_min<=depth_gt[i][j]<=depth_gt_max and depth_estimate_range[k]<depth_estimate[i][j]<depth_estimate_range[k+1]:
                    scale_list.append(depth_gt[i][j]/depth_estimate[i][j])
        if len(scale_list)!=0:
            a,b,c=dbscan(torch.tensor(scale_list), eps=0.0006)    
            # a,b,c=dbscan(torch.tensor(scale_list), eps=0.01)
            final_result1.append([a.item(),b.item(),c.shape[0],len(scale_list)])
            if c.shape[0]>100:
                final_result.append([a.item(),(depth_estimate_range[k]+depth_estimate_range[k+1])/2,k])

            ## final_result=[深度区域内的scale,区域下界，区域上界]  


    scale_arr=np.zeros(depth_estimate.shape)
    mask_scale_arr=np.zeros(depth_estimate.shape)
    model=poly_fit(final_result)
    depth_estimate_save=np.zeros_like(depth_estimate)
    depth_estimate_mask=np.zeros_like(depth_estimate)
    scale_max=np.max(np.array(final_result)[:,0])
    scale_min=np.min(np.array(final_result)[:,0])

    plot_list=np.array(final_result)
    scale_plot=np.zeros_like(plot_list[:,1])

    for i in range(plot_list[:,1].shape[0]):
        scale_plot[i]=model.predict(plot_list[:,1][i].reshape(-1,1))       
    
    fig, ax = plt.subplots()
    ax.plot(plot_list[:,1], plot_list[:,0], label='Before', linewidth=5, linestyle=':')   
    ax.plot(plot_list[:,1],scale_plot, label='after') 
    fig.legend()
    plt.savefig('/media/pc/D/zzt/CoR-GS/data/DTU_new/DTU_observe/'+dir_now+'+'+file_now.replace('npy','png'))

    for i in range(depth_estimate0.shape[0]):
        for j in range(depth_estimate0.shape[1]):
            if depth_estimate0[i][j]!=0:
                scale_arr[i][j]=model.predict(depth_estimate0[i][j].reshape(1,-1))
                scale_arr[i][j]=max(scale_arr[i][j], scale_min)
                scale_arr[i][j]=min(scale_arr[i][j], scale_max)

    for i in range(depth_estimate0.shape[0]):
        for j in range(depth_estimate0.shape[1]):
            if depth_estimate0[i][j]!=0:
                depth_estimate_save[i][j]=depth_estimate0[i][j]*scale_arr[i][j]
                # if depth_estimate_save[i][j]>1.1*depth_gt_max:
                #     depth_estimate_save[i][j]=1*depth_gt_max
                # if depth_estimate_save[i][j]<0.9*depth_gt_min:
                #     depth_estimate_save[i][j]=1*depth_gt_min

    mse = np.mean((depth_estimate_save - depth_gt0) ** 2)
    print('The mse after scaling', mse) 
    if mse>1:
        with open('zzh_abnormal.txt', 'a') as file:
                file.write(str(mse)+file_now+'\n')
                                
    np.save(output_dir+dir_now+'/depth_npy/'+file_now, depth_estimate_save)
    # depth_estimate_save=depth_estimate_save/np.max(depth_estimate_save)    
    # plt.imsave(output_dir+dir_now+'/depth/'+file_now.replace('npy','png'),depth_estimate_save, cmap='gray')
    
    
def read_file(file_path):
    try:
        tmp=cv2.imread(file_path)   
        print(tmp.shape)
        return np.mean(tmp,axis=2)
    except FileNotFoundError:
        print(f"MASK 不存在")
        return None

def main():
    global dir_now, file_now, input_dir, output_dir
    if os.path.exists(output_dir+'submission_data/idrmasks'):
        shutil.rmtree(output_dir+'submission_data/idrmasks')
    shutil.copytree(input_dir+'submission_data/idrmasks', output_dir+'submission_data/idrmasks')
    for dir_name in os.listdir(input_dir):
        if dir_name.startswith("scan") and dir_name == args.scan: # or dir_name=="scan103" and (dir_name=="scan40" )
            os.makedirs(output_dir+dir_name,exist_ok=True)
            os.makedirs(output_dir+dir_name+'/sparse',exist_ok=True)
            os.makedirs(output_dir+dir_name+'/sparse/0',exist_ok=True)
            if os.path.exists(output_dir+dir_name+'/images'):
                shutil.rmtree(output_dir+dir_name+'/images')
            shutil.copytree(input_dir+dir_name+'/dense/images', output_dir+dir_name+'/images')
            shutil.copy(input_dir+dir_name+'/dense/sparse/cameras.bin', output_dir+dir_name+'/sparse/0/cameras.bin')
            shutil.copy(input_dir+dir_name+'/dense/sparse/images.bin', output_dir+dir_name+'/sparse/0/images.bin')
            shutil.copy(input_dir+dir_name+'/dense/sparse/points3D.bin', output_dir+dir_name+'/sparse/0/points3D.bin')
            os.makedirs(output_dir+dir_name+'/depth_npy', exist_ok=True)  
            dir_now=dir_name
            for file_name in os.listdir(input_dir+dir_name+'/depth_pho_colmap_npy'):
                if file_name.endswith(".npy"):  
                    file_now=file_name 
                    if '023' in file_name or '026' in file_name or  '029' in file_name: 
                        depth_gt=np.load(input_dir+dir_name+'/depth_pho_colmap_npy/'+file_name)
                        depth_estimate=np.load(input_dir+dir_name+'/depth_npy_anything/'+file_name)
                        print(input_dir+'mask/'+dir_name+'/'+file_name[5:file_name.index('_3_r5000.npy')]+'.png')
                        j = int(file_name[5:file_name.index('_3_r5000.npy')]) - 1
                        mask_image=read_file(input_dir+'submission_data/resize_mask/'+dir_name+'/'+f'{j:03d}'+'.png')
                        data_process(depth_gt,depth_estimate,mask_image)

if __name__ == "__main__":
    main()