#!/usr/bin/env python2
"""Visualize a pose."""
# pylint: disable=invalid-name, wrong-import-order, no-member
import os.path as path
import numpy as np
import click
from scipy.misc import imread, imsave
from clustertools.visualization import visualize_pose
import clustertools.pose_models as cpm


@click.command()
@click.argument("image_fp", type=click.Path(exists=True,
                                            dir_okay=False,
                                            readable=True))
@click.option('--scale', type=click.FLOAT, default=1.)
@click.option('--line_thickness', type=click.INT, default=3)
@click.option('--dash_length', type=click.INT, default=15)
@click.option('--opacity', type=click.FLOAT, default=0.6)
@click.option('--noconn', type=click.BOOL, default=False, is_flag=True)
def cli(image_fp, scale=1., line_thickness=3, dash_length=15, opacity=0.6,  # pylint: disable=too-many-arguments
        noconn=False):
    """Visualize a pose."""
    # Load the image.
    image = imread(image_fp)[:, :, :3]
    # Load the pose.
    pose_fp = image_fp + '_pose.npz'
    if path.exists(pose_fp):
        pose = np.load(pose_fp)['pose']
    else:
        pose_fp = image_fp + '_pose.npy'
        if path.exists(pose_fp):
            pose = np.load(pose_fp)
    connections = None
    if pose.shape[1] == 14 and not noconn:
        connections = cpm.connections_lsp
        lm_region_mapping = None
    elif pose.shape[1] == 91 and not noconn:
        connections = cpm.connections_landmarks_91
        lm_region_mapping = cpm.lm_region_mapping
    vis_im = visualize_pose(image,
                            pose[:2],
                            line_thickness=line_thickness,
                            dash_length=dash_length,
                            opacity=opacity,
                            connections=connections,
                            lm_region_mapping=lm_region_mapping,
                            scale=scale)
    imsave(image_fp + '_pose_vis.png', vis_im)

if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
