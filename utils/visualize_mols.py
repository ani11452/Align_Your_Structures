import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
from .dataset_configs import get_dataset_info, get_bond_order, geom_predictor


def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0,
                    alpha=alpha)


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color,
                  dataset_info):
    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    # ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2
    # areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)  # , linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                           dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]],
                    dataset_info['atom_decoder'][s[1]])
            if 'qm9' in dataset_info['name']:
                draw_edge_int = get_bond_order(atom1, atom2, dist)
                line_width = (3 - 2) * 2 * 2
            elif dataset_info['name'] == 'geom':
                draw_edge_int = geom_predictor(pair, dist)
                # Draw edge outputs 1 / -1 value, convert to True / False.
                line_width = 2
            else:
                raise Exception('Wrong dataset_info name')
            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    # linewidth_factor = draw_edge_int  # Prop to number of
                    # edges.
                    linewidth_factor = 1
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        linewidth=line_width * linewidth_factor,
                        c=hex_bg_color, alpha=alpha)

def plot_data3d(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='white', alpha=1.):  # Changed default to white
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                  hex_bg_color, dataset_info)

    if 'qm9' in dataset_info['name']:
        max_value = positions.abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = positions.abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=False)  # Changed to False

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()

def plot_ours(molecule, output_path, dataset_name, remove_h):
    atom_number, x = molecule  # [N], [N, 3]

    positions = x.view(-1, 3)
    positions_centered = positions - positions.mean(dim=0, keepdim=True)

    dataset_info = get_dataset_info(dataset_name, remove_h)
    atom_type = torch.from_numpy(
        np.array([dataset_info['atomic_nb'].index(i.item()) for i in atom_number])
    ).numpy()

    plot_data3d(
        positions_centered,
        atom_type,
        dataset_info=get_dataset_info(dataset_name, remove_h),
        spheres_3d=True,
        bg='white',
        save_path=output_path,
    )


if __name__ == '__main__':
    import os
    output_path = 'vis/molecule_test'
    os.makedirs(output_path, exist_ok=True)
    dataset_name = 'qm9'
    
    # Test case 1: Water molecule (H₂O)
    atom_number = torch.from_numpy(np.array([8, 1, 1]))  # O, H, H
    # Position the oxygen at the center and hydrogens at a typical O-H bond distance
    x = torch.tensor([
        [0.0, 0.0, 0.0],           # Oxygen at center
        [0.0, 0.8, -0.5],          # Hydrogen 1
        [0.0, -0.8, -0.5]          # Hydrogen 2
    ], dtype=torch.float)
    molecule = (atom_number, x)
    plot_ours(molecule, f'{output_path}/water.png', dataset_name, remove_h=False)
    
    # atom_number = torch.from_numpy(np.array([1, 1, 6, 7, 8, 9]))
    # x = torch.randn(6, 3) * 3
    # molecule = (atom_number, x)
    # output_path = 'vis/exp_temp'
    # os.makedirs(output_path, exist_ok=True)
    # dataset_name = 'qm9'
    # remove_h = False
    # plot_ours(molecule, f'{output_path}/1.png', dataset_name, remove_h)

    # remove_h = True
    # atom_number = torch.from_numpy(np.array([6, 7, 8, 9]))
    # x = torch.randn(4, 3) * 3
    # molecule = (atom_number, x)
    # plot_ours(molecule, f'{output_path}/2.png', dataset_name, remove_h)
    