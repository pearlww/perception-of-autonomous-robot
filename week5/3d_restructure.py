import numpy as np
import cv2
import glob
import open3d as o3d
import re


rgb = glob.glob('car_challange/rgb/*.jpg')
depth = glob.glob('car_challange/depth/*.png')
rgb.sort(key=lambda f: int(re.sub('\D', '', f)))
depth.sort(key=lambda f: int(re.sub('\D', '', f)))
assert rgb
assert depth

def draw_registrations(source, target, transformation = None, recolor = False):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        if(recolor):
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
        if(transformation is not None):
            source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down , pcd_fpfh

def local_registration(source,target,threshold,trans_init):
  

    point_to_point =  o3d.registration.TransformationEstimationPointToPoint(False)
    point_to_plane = o3d.registration.TransformationEstimationPointToPlane()

    icp_result = o3d.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        #point_to_plane
        point_to_point
        )
    return icp_result.transformation

def global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    point_to_point =  o3d.registration.TransformationEstimationPointToPoint(False)
    point_to_plane =  o3d.registration.TransformationEstimationPointToPlane()

    corr_length = 0.9
    distance_threshold = voxel_size * 1.5

    c0 = o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(corr_length)
    c1 = o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    c2 = o3d.registration.CorrespondenceCheckerBasedOnNormal(0.095)

    checker_list = [c0,c1,c2]

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, 
        source_fpfh, target_fpfh,
        distance_threshold,
        point_to_point,
        #point_to_plane,
        checkers = checker_list)

    return result.transformation

def generate_pcd(color_raw,depth_raw,camera):

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)

    return pcd



camera = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

color_raw0 = o3d.io.read_image("car_challange/rgb/0000001.jpg")
depth_raw0 = o3d.io.read_image("car_challange/depth/0000001.png")

source = generate_pcd(color_raw0,depth_raw0,camera)

n=0
voxel_size=0.02 #越大越稀疏
threshold = 0.02

for rgb_file, depth_file in zip(rgb,depth):

    if n%10==0:

        print("procressing {}".format(rgb_file))
        color_raw1 = o3d.io.read_image(rgb_file)
        depth_raw1 = o3d.io.read_image(depth_file)
        target = generate_pcd(color_raw1,depth_raw1,camera)


        source_down,source_fpfh = preprocess_point_cloud(source,voxel_size)
        target_down,target_fpfh = preprocess_point_cloud(target,voxel_size)  

        glob_trans = global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size)

        #evaluation = o3d.registration.evaluate_registration(source, target, threshold, glob_trans)
        #print(evaluation)

        local_trans = local_registration(source_down,target_down,threshold,glob_trans)

        #source.transform(glob_trans) don't need this
        source.transform(local_trans)

        source=source+target
        source = source.voxel_down_sample(voxel_size=0.02)

        n+=1

    elif n>50:
        break
    else:
        n+=1
        continue
        #break

o3d.visualization.draw_geometries([source])