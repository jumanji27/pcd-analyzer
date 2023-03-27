from copy import deepcopy

import open3d as o3d


class Aligner:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def draw_registration_result(self, standart, sample, transformation):
        standart_temp = deepcopy(standart)
        sample_temp = deepcopy(sample)
        standart_temp.paint_uniform_color([1, 0, 0])
        sample_temp.paint_uniform_color([1, 1, 0])
        standart_temp.transform(transformation)
        o3d.visualization.draw_geometries([standart_temp, sample_temp])

    def preprocess(self, pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def execute_fast_global_registration(
        self, standart_down, sample_down, standart_fpfh, sample_fpfh, voxel_size
    ):
        distance_threshold = voxel_size * 0.5
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            standart_down, sample_down, standart_fpfh, sample_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
        )
        return result
