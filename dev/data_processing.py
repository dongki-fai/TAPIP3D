import cv2
import argparse
import open3d as o3d
import numpy as np
from rich import print
from einops import rearrange
from pathlib import Path


def process_data(
    data: np.lib.npyio.NpzFile,
):
    # parse relevant data
    video = data["video"]
    depths = data["depths"]
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    print(f"Intrinsics:\n{intrinsics[0]}")

    assert video.shape[0] == depths.shape[0], "Temporal dimension of depths must match video"
    assert video.shape[0] == intrinsics.shape[0], "Temporal dimension of intrinsics must match video"
    assert video.shape[0] == extrinsics.shape[0], "Temporal dimension of extrinsics must match video"

    # process video and depths
    video = (rearrange(video, "T C H W -> T H W C") * 255).astype(np.uint8)
    depths = depths.astype(np.float32)
    print(f"Video shape: {video.shape} and depths shape: {depths.shape}")

    # compute inv_extrinsics
    # notation convention: T_{ab}: transformation from a to b
    first_frame_inv = np.linalg.inv(extrinsics[0])  # T_{w->c0} -> T_{c0->w}
    normalized_extrinsics = np.array([
        first_frame_inv @ ext for ext in extrinsics
    ])  # T_{c0->w} x T_{w->cx} -> T_{c0->cx}
    inv_extrinsics = np.linalg.inv(normalized_extrinsics)  # T_{cx->c0}
    return video, depths, intrinsics, inv_extrinsics

    
def filter_pcd(pcd, top_k=1):
    print(f"Original {pcd}")

    # apply statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    print(f"After statistical outlier: {pcd}")

    # # apply radius outlier removal (catch isolated points)
    # pcd, _ = pcd.remove_radius_outlier(
    #     nb_points=15,  # 3D radius around each point to search for neighbors
    #     radius=0.05  # min number of neighbors required in that radius for a point to be kept
    # )

    # # apply dbscan to drop tiny clusters that survive
    # if True:
    #     labels = np.array(pcd.cluster_dbscan(
    #         eps=0.1,  # max distance between two points for them to be considered in the same cluster
    #         min_points=1000,  # min number of points to form a cluster
    #         print_progress=False
    #     ))
    #     if labels.size > 0 and labels.max() >= 0:
    #         # get counts for each label (excluding noise: label = -1)
    #         unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    #         target_labels = []
    #         for label, count in zip(unique_labels, counts):
    #             if count > 1000:
    #                 target_labels.append(label)

    #         # # sort by size, descending
    #         # sorted_labels = unique_labels[np.argsort(-counts)]
    #         # print(f"sorted_labels: {sorted_labels}")
    
    #         # # pick top-k labels
    #         # top_k_labels = sorted_labels[:top_k]
    #         # print(f"counts:", counts[top_k_labels])
    
    #         # Get indices for points belonging to top-k clusters
    #         # mask = np.isin(labels, top_k_labels)
    #         mask = np.isin(labels, np.array(target_labels))
    #         pcd = pcd.select_by_index(np.where(mask)[0])
    #         print(f"After dbscan filtering: {pcd}")
    
    return pcd

    
def create_voxel_grid(pc_local, pc_global, rgb, voxel_size, bounds):
    """
    Vectorized voxelization: each point is mapped to its voxel index directly,
    and we aggregate by voxel to get mean position and color.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    grid_dims = (
        int(np.ceil((x_max - x_min) / voxel_size)),
        int(np.ceil((y_max - y_min) / voxel_size)),
        int(np.ceil((z_max - z_min) / voxel_size)),
    )
    assert grid_dims == (64, 64, 32), "Grid dim mismatch. Must be (128, 128, 8)."

    # Compute voxel indices for each point
    ix = ((pc_local[:, 0] - x_min) / voxel_size).astype(int)
    iy = ((pc_local[:, 2] - y_min) / voxel_size).astype(int)
    iz = ((pc_local[:, 1] - z_min) / voxel_size).astype(int)

    # Filter out points that fall outside the grid
    valid_mask = (
        (ix >= 0) & (ix < grid_dims[0]) &
        (iy >= 0) & (iy < grid_dims[1]) &
        (iz >= 0) & (iz < grid_dims[2])
    )
    ix, iy, iz = ix[valid_mask], iy[valid_mask], iz[valid_mask]
    pc_global = pc_global[valid_mask]
    rgb = rgb[valid_mask]

    # prepare voxel storage
    voxel_pc = np.zeros((*grid_dims, 6), dtype=np.float32)
    counts = np.zeros(grid_dims, dtype=np.int32)

    # Flatten voxel indices for grouping
    flat_indices = ix * (grid_dims[1] * grid_dims[2]) + iy * grid_dims[2] + iz

    # Sum up xyz and rgb values per voxel
    np.add.at(voxel_pc.reshape(-1, 6), flat_indices, np.hstack([pc_global, rgb]))
    np.add.at(counts.ravel(), flat_indices, 1)

    # Compute means where counts > 0
    mask_nonzero = counts > 0
    voxel_pc[mask_nonzero] /= counts[mask_nonzero][..., None]

    return voxel_pc
    

def main(args: argparse.Namespace):
    # read result file
    data = np.load(args.result_file)
    print(f"Read result file: {data}")

    # process point cloud data
    video, depths, intrinsics, inv_extrinsics = process_data(data)

    # pull camera parameters
    fx, fy = intrinsics[0][0, 0], intrinsics[0][1, 1]
    cx, cy = intrinsics[0][0, 2], intrinsics[0][1, 2]

    # set bounds for voxelization
    width_min = -args.voxel_size * 32
    width_max = args.voxel_size * 32
    depth_min = 0.3
    depth_max = depth_min + args.voxel_size * 64
    height_max = 0.3
    height_min = height_max - args.voxel_size * 32
    bounds = (
        width_min,
        depth_min,
        height_min,
        width_max,
        depth_max,
        height_max
    )

    # define point cloud related parameters
    width_min = -args.voxel_size * 64
    depth_min = 0.3  # z-axis
    depth_max = depth_min + args.voxel_size * 128
    height_max = 0.3
    height_min = height_max - args.voxel_size * 8

    # process each frame
    for i_frame in range(video.shape[0]):
        print(f"Processing frame: {i_frame}")

        # # apply median depth to reduce speckle
        # depth_median = cv2.medianBlur(depths[i_frame], ksize=5)

        # flatten rgb and depth
        rgb_flat = video[i_frame].reshape(-1, 3) / 255.0
        depth_flat = depths[i_frame].flatten()

        # filter based on valid depth data
        valid_depth = (depth_flat > depth_min) & (depth_flat < depth_max)
        rgb_flat = rgb_flat[valid_depth]
        z = depth_flat[valid_depth]
        print(f"Depth filter: {depth_flat.shape} -> {z.shape}")

        # compute projected point cloud
        width, height = video.shape[2], video.shape[1]
        xs = np.tile(np.arange(width, dtype=np.float32), height)
        ys = np.repeat(np.arange(height, dtype=np.float32), width)
        x = ((xs[valid_depth] - cx) * z) / fx
        y = ((ys[valid_depth] - cy) * z) / fy

        # homogeneous camera points (p_cx) -> frame0
        pc_wrt_local = np.stack([x, y, z, np.ones_like(z)], axis=1)
        pc_wrt_c0 = (inv_extrinsics[i_frame] @ pc_wrt_local.T).T[:, :3]

        # voxelize point cloud using pc_wrt_local
        # with features of xyz values and rgb values
        pc_voxel = create_voxel_grid(
            pc_wrt_local,
            pc_wrt_c0,
            rgb_flat,
            args.voxel_size,
            bounds
        )
        pc_voxel_vis_xyz, pc_voxel_vis_rgb = [], []
        for i in range(pc_voxel.shape[0]):
            for j in range(pc_voxel.shape[1]):
                for k in range(pc_voxel.shape[2]):
                    if pc_voxel[i, j, k, 0] != 0:
                        pc_voxel_vis_xyz.append(pc_voxel[i, j, k, :3])
                        pc_voxel_vis_rgb.append(pc_voxel[i, j, k, 3:])

        # normalize pc_voxel for training stability
        pc_voxel[:, :, :, 2] = pc_voxel[:, :, :, 2] / 10.
        print(f"Global min x: {pc_voxel[:, :, :, 0].min():.2f} / max x: {pc_voxel[:, :, :, 0].max():.2f}")
        print(f"Global min y: {pc_voxel[:, :, :, 1].min():.2f} / max x: {pc_voxel[:, :, :, 1].max():.2f}")
        print(f"Global min z: {pc_voxel[:, :, :, 2].min():.2f} / max x: {pc_voxel[:, :, :, 2].max():.2f}")

        # make into pcd
        pcd_local = o3d.geometry.PointCloud()
        pcd_local.points = o3d.utility.Vector3dVector(pc_wrt_local[:, :3])
        pcd_local.colors = o3d.utility.Vector3dVector(rgb_flat.astype(np.float32))

        pcd_global = o3d.geometry.PointCloud()
        pcd_global.points = o3d.utility.Vector3dVector(pc_wrt_c0)
        pcd_global.colors = o3d.utility.Vector3dVector(rgb_flat.astype(np.float32))

        pcd_voxel = o3d.geometry.PointCloud()
        pcd_voxel.points = o3d.utility.Vector3dVector(np.array(pc_voxel_vis_xyz))
        pcd_voxel.colors = o3d.utility.Vector3dVector(np.array(pc_voxel_vis_rgb))

        # # apply filtering
        # pcd = filter_pcd(pcd)
    
        # save point cloud
        o3d.io.write_point_cloud(f"data/pc_{str(i_frame).zfill(3)}_local.pcd", pcd_local)
        o3d.io.write_point_cloud(f"data/pc_{str(i_frame).zfill(3)}_global.pcd", pcd_global)
        o3d.io.write_point_cloud(f"data/pc_{str(i_frame).zfill(3)}_voxel.pcd", pcd_voxel)

        # save voxel grid
        np.save(f"data/pc_{str(i_frame).zfill(3)}.npy", pc_voxel)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_file',
        type=str,
        required=True,
        help='Path to the result file input (e.g., resule.npz file)'
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=0.04,
        help='Target voxel size'
    )
    args = parser.parse_args()

    # run main
    main(args)


# # filter based on relative height to camera
# # to remove objects that do not matter (e.g., trees)
# ground_height = 0.3
# valid_height = (pc_wrt_local[:, 1] <= ground_height) & (pc_wrt_local[:, 1] > ground_height - args.voxel_size * 8)
# pc_wrt_local = pc_wrt_local[valid_height][:, :3]
# pc_wrt_c0 = pc_wrt_c0[valid_height]
# rgb_flat = rgb_flat[valid_height]
# print(f"min x, max x: {pc_wrt_local[:, 0].min():.2f}, {pc_wrt_local[:, 0].max():.2f}")
# print(f"min y, max y: {pc_wrt_local[:, 1].min():.2f}, {pc_wrt_local[:, 1].max():.2f}")
# print(f"min z, max z: {pc_wrt_local[:, 2].min():.2f}, {pc_wrt_local[:, 2].max():.2f}")
