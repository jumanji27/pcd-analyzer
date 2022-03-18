import glob
import math

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import open3d as o3d
from scipy.spatial.transform import Rotation
import yaml

from aligner import Aligner


class PCDAnalyzer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._extract_counter = 0
        self._inliers = []

    def preprocess(self):
        path = self._config['preprocess']['input_path']
        path += '' if path[-1] == '/' else '/'
        txt_files = glob.glob(path + '*.txt')
        raws = []
        for data in txt_files:
            raws.append(pd.read_csv(
                data, names=['x', 'y', 'z']
            ))
        frame = pd.concat(raws, axis=0, ignore_index=True)
        cloud = PyntCloud(pd.DataFrame(
            data=frame, columns=['x', 'y', 'z']
        ))
        cloud.to_file(self._config['preprocess']['ouput_file'])
        pcd = self.read(self._config['preprocess']['ouput_file'])
        pcd = pcd.voxel_down_sample(voxel_size=self._config['preprocess']['downsampling_voxel_size'])
        o3d.io.write_point_cloud(self._config['preprocess']['ouput_file'], pcd)

    def read(self, file):
        return o3d.io.read_point_cloud(file)

    def filter_table(self, pcd):
        labels = np.array(pcd.cluster_dbscan(
            eps=20, min_points=10, print_progress=True
        ))
        all_filtered_indices = {}
        for index, label in enumerate(labels):
            if not all_filtered_indices.get(label):
                all_filtered_indices[label] = []
            all_filtered_indices[label].append(index)
        filtered_indices = None
        max_length = 0
        for index, indices in all_filtered_indices.items():
            if len(indices) > max_length:
                max_length = len(indices)
                filtered_indices = indices
        return pcd.select_by_index(filtered_indices), pcd

    def _extract_plane(self, pcd):
        model, inliers = pcd.segment_plane(
            distance_threshold=self._config['plane_segmentation_threshold'],
            ransac_n=3,
            num_iterations=1000
        )
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        if len(inliers) > self._config['min_plane_size']:
            self._inliers.append([inlier_cloud, len(inliers), model])
            self._extract_plane(outlier_cloud)
            self._extract_counter += 1
        return self._inliers

    def _calculate_angle(self, surface_model, compare_surface_type):
        compare_surface_types = {
            'xy': [0, 0, 1],
            'yz': [1, 0, 0],
            'xz': [0, 1, 0]
        }
        a1 = surface_model[0]
        b1 = surface_model[1]
        c1 = surface_model[2]
        a2 = compare_surface_types[compare_surface_type][0]
        b2 = compare_surface_types[compare_surface_type][1]
        c2 = compare_surface_types[compare_surface_type][2]
        d = (a1 * a2 + b1 * b2 + c1 * c2)
        e1 = math.sqrt(a1 * a1 + b1 * b1 + c1 * c1)
        e2 = math.sqrt(a2 * a2 + b2 * b2 + c2 * c2)
        d = d / (e1 * e2)
        return math.degrees(math.acos(d))

    def calculate_angles(self, standart, sample):
        standart = self.read(standart)
        sample = self.read(sample)
        trans_init = np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        standart.transform(trans_init)
        aligner = Aligner()
        if self._config['visualize']['initial_aligment']:
            aligner.draw_registration_result(standart, sample, np.identity(4))
        standart_down, standart_fpfh = aligner.preprocess(standart, 20)
        sample_down, sample_fpfh = aligner.preprocess(sample, 20)
        result_ransac = aligner.execute_fast_global_registration(
            standart_down, sample_down, standart_fpfh, sample_fpfh, self._config['alignment_voxel_size']
        )
        transformation_matrix = np.asarray([
            result_ransac.transformation[0][:3],
            result_ransac.transformation[1][:3],
            result_ransac.transformation[2][:3]
        ])
        if self._config['visualize']['final_aligment']:
            aligner.draw_registration_result(standart_down, sample_down, result_ransac.transformation)
        return Rotation.from_matrix(transformation_matrix).as_euler(
            self._config['angles_format'], degrees=True
        ).tolist()

    def process(self, pcds):
        # Horizontal plane finding
        inliers = self._extract_plane(pcds[0])
        if inliers[0][1] / inliers[1][1] <= self._config['horizontal_plane_coefficient']:
            print('Error: can\'t find main plane!')
            return

        horizontal_plane = inliers[0][0]
        np_horizontal_plane = np.asarray(horizontal_plane.points)
        horizontal_point = np.average(np_horizontal_plane, axis=0)
        horizontal_point_plane = o3d.geometry.PointCloud()
        horizontal_point_plane.points = o3d.utility.Vector3dVector([horizontal_point])
        print(f'Resultant point for horizontal plane: {horizontal_point}')
        print(f'Horizontal plane equation coefficients: {inliers[0][2]}')

        # Vertical planes finding
        side_planes = []
        first_side_plane_index = None
        for index, inlier in enumerate(inliers):
            if not first_side_plane_index and \
                    abs(self._calculate_angle(inliers[0][2], 'xy') - self._calculate_angle(inlier[2], 'xy')) \
                    > self._config['side_planes_coefficient'][1]:
                side_planes.append([inlier[0], inlier[2]])
                print(f'Vertical plane {len(side_planes)} equation coefficients: {inlier[2]}')
                first_side_plane_index = index
                continue
            if first_side_plane_index and \
                    abs(self._calculate_angle(inlier[2], 'xy')
                        - self._calculate_angle(inliers[first_side_plane_index][2], 'xy')) \
                    < self._config['side_planes_coefficient'][0]:
                side_planes.append([inlier[0], inlier[2]])
                print(f'Vertical plane {len(side_planes)} equation coefficients: {inlier[2]}')
                break

        if len(side_planes) < 2:
            print('Error: can\'t find enough side planes!')
            return

        # Resultant vertical plane finding
        np_pcd = np.asarray(pcds[0].points)
        plane = [
            (side_planes[0][1][0] + side_planes[1][1][0]) / 2,
            (side_planes[0][1][1] + side_planes[1][1][1]) / 2,
            (side_planes[0][1][2] + side_planes[1][1][2]) / 2
        ]
        averages = np.vstack([
            np.average(
                np.asarray(side_planes[0][0].points), axis=0
            ),
            np.average(
                np.asarray(side_planes[1][0].points), axis=0
            )
        ])
        vertical_point = np.average(averages, axis=0)
        vertical_point_plane = o3d.geometry.PointCloud()
        vertical_point_plane.points = o3d.utility.Vector3dVector([vertical_point])
        D = -vertical_point.dot(plane)
        np_vertical_plane_pcd = np_pcd[abs(np_pcd.dot(plane) + D) <= self._config['vertical_plane_width']]
        vertical_plane_pcd = o3d.geometry.PointCloud()
        vertical_plane_pcd.points = o3d.utility.Vector3dVector(np_vertical_plane_pcd)
        model, inliers = vertical_plane_pcd.segment_plane(
            distance_threshold=self._config['vertical_plane_width'],
            ransac_n=3,
            num_iterations=1000
        )
        vertical_plane = vertical_plane_pcd.select_by_index(inliers)
        print(f'Resultant vertical plane equation coefficients: {model}')

        # Intersection of horizontal and resultant vertical planes finding
        resultant_plane_points = []
        for vertical_point in np_vertical_plane_pcd:
            if vertical_point in np_horizontal_plane:
                resultant_plane_points.append(vertical_point)
        np_resultant_plane_points = np.asarray(resultant_plane_points)

        # Final resultant point finding
        resultant_point_plane = o3d.geometry.PointCloud()
        np_resultant_plane_point = np.argmax(np_resultant_plane_points, axis=0)
        resultant_plane_point = np_resultant_plane_points.tolist()[np_resultant_plane_point.tolist()[1]]
        resultant_point_plane.points = o3d.utility.Vector3dVector([resultant_plane_point])
        if self._config['visualize']['surfaces']:
            self._draw_surfaces([pcds[1], horizontal_plane, vertical_plane])
        if self._config['visualize']['resultant_point']:
            self._draw_point([pcds[1], resultant_point_plane])
        angles = self.calculate_angles(self._config['pcd_standart'], self._config['pcd_sample'])

        print('-----------')
        print(f'xyz: {resultant_plane_point}')
        print(f'abc: {angles}')

    def _draw_surfaces(self, pcds):
        pcds[0].paint_uniform_color([0.6, 0.6, 0.6])
        pcds[1].paint_uniform_color([1, 0, 0])
        pcds[2].paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries(pcds)

    def _draw_point(self, pcds):
        pcds[0].paint_uniform_color([0.6, 0.6, 0.6])
        pcds[1].paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(pcds)

    def __call__(self):
        self.process(self.filter_table(self.read(
            self._config['pcd_sample'],
        )))


with open('system/config.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

pcd_analyzer = PCDAnalyzer(_config=config)
# pcd_analyzer.preprocess()
pcd_analyzer()
