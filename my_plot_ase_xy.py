import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb


def plot_ase_xy(
    atoms,
    cutoff=1.5,
    atom_radius=0.25,

    # Wavefunction overlay (optional): plotted iff psi is not None
    psi=None,
    wf_scale=1.0,
    wf_alpha=0.45,
    prob_mode="sqrt",          # "sqrt" => radius ~ |psi|, "linear" => radius ~ |psi|^2
    show_phase_colorbar=True,

    # Bond currents (optional): plotted iff J is not None
    J=None,
    current_dir="j_to_i",      # "i_to_j" | "j_to_i" | "sign"
    current_scale=1.,
    current_alpha=0.7,
    current_threshold=0.0,
    currents_on_bonds_only=True,

    # Current style: line + triangle head (paper-style)
    head_length=0.4,          # Å (base), scaled weakly with |J|
    head_width=0.3,           # Å (base), scaled weakly with |J|
    lw_min=0.6,                # minimum shaft linewidth
    lw_max=3.0,                # maximum shaft linewidth

    highlight_atoms=None,      # List of atom indices to highlight
    highlight_radius=None,     # Radius for highlighted atoms
    
    ax=None,
):
    
    pos = atoms.get_positions()
    # xy = pos[:, [0, 2]].astype(float) # Original: x and z
    # Rotate 90 degrees clockwise: (x, z) -> (z, -x)
    # But since we usually want positive axes, let's just swap them and maybe flip if needed.
    # 90 deg clockwise: new_x = z, new_y = -x. 
    # Let's try just swapping x and z for now as a "rotation" of the view, 
    # effectively plotting (z, x) or (z, -x).
    # If the user wants 90 deg clockwise rotation of the image:
    # x' = y
    # y' = -x
    # origin is usually flexible in plots.
    
    # Original map: x -> plot_x, z -> plot_y
    # We want: 
    #   old_plot_x (axis 0) -> new_plot_y (axis 1)
    #   old_plot_y (axis 2) -> -new_plot_x (axis 0) (or just new_plot_x depending on preference)

    # Let's simply swap the columns extracted to rotate the view.
    # If we map x->y and z->x, it's a 90 degree rotation/reflection.
    
    # x_new = z (pos[:, 2])
    # y_new = -x (pos[:, 0])  <-- standard 90 deg CW rotation x,y -> y, -x
    
    x_coords = pos[:, 0]
    z_coords = pos[:, 2]
    
    # 90 degrees clockwise: (x, y) -> (y, -x)
    # Here our "y" is z.
    # So (x, z) -> (z, -x)
    
    xy = np.column_stack((z_coords, -x_coords)).astype(float)
    N = len(atoms)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # --- Bonds (neighbors within cutoff) ---
    bond_segs = []
    bonded = set()
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(xy[i] - xy[j]) <= cutoff:
                bond_segs.append([xy[i], xy[j]])
                bonded.add((i, j))
                bonded.add((j, i))

    if bond_segs:
        ax.add_collection(
            LineCollection(bond_segs, colors="black", linewidths=0.6, alpha=0.9, zorder=1)
        )

    # --- Atom outlines ---
    highlight_set = set(highlight_atoms) if highlight_atoms is not None else set()
    for i in range(N):
        if i in highlight_set:
            ec = "black"
            lw = 2.0
            r_use = highlight_radius if highlight_radius is not None else atom_radius
        else:
            ec = "black"
            lw = 1.0
            r_use = atom_radius

        ax.add_patch(
            Circle(
                xy[i],
                radius=r_use,
                edgecolor=ec,
                facecolor="none",
                linewidth=lw,
                zorder=2,
            )
        )

    # --- Wavefunction overlay (if present) ---
    if psi is not None:
        psi = np.asarray(psi)
        if psi.shape != (N,):
            raise ValueError(f"psi must have shape ({N},), got {psi.shape}")

        phase = np.angle(psi)  # [-pi, pi]
        hue = (phase + np.pi) / (2 * np.pi)  # [0,1)
        rgb = hsv_to_rgb(np.column_stack([hue, np.ones(N), np.ones(N)]))

        prob = np.abs(psi) ** 2
        if prob_mode == "sqrt":
            rad = wf_scale * np.sqrt(prob)   # ~ |psi|
        elif prob_mode == "linear":
            rad = wf_scale * prob            # ~ |psi|^2
        else:
            raise ValueError("prob_mode must be 'sqrt' or 'linear'")

        for i in range(N):
            r = float(rad[i])
            if r <= 0:
                continue
            ax.add_patch(
                Circle(
                    xy[i],
                    radius=r,
                    edgecolor="none",
                    facecolor=rgb[i],
                    alpha=wf_alpha,
                    zorder=3,
                )
            )

        if show_phase_colorbar:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize, ListedColormap

            hues = np.linspace(0, 1, 256, endpoint=False)
            cmap = ListedColormap(
                hsv_to_rgb(np.column_stack([hues, np.ones_like(hues), np.ones_like(hues)]))
            )
            sm = ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("phase(ψ)")
            cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # --- Currents: shaft line + triangular head (if present) ---
    if J is not None:
        J = np.asarray(J)
        if J.shape != (N, N):
            raise ValueError(f"J must have shape {(N, N)}, got {J.shape}")

        # Collect eligible pairs (i>j) and magnitudes for normalization
        pairs = []
        mags = []
        for i in range(N):
            for j in range(i):  # i > j
                if currents_on_bonds_only and ((i, j) not in bonded):
                    continue
                a = abs(J[i, j])
                if a > current_threshold:
                    pairs.append((i, j))
                    mags.append(a)

        if mags:
            maxv = max(mags)
            shaft_segs = []
            shaft_lws = []

            for (i, j), a in zip(pairs, mags):
                Jij = J[i, j]
                rel = a / maxv if maxv > 0 else 1.0

                # Choose direction for this pair
                if current_dir == "i_to_j":
                    src, dst = i, j
                elif current_dir == "j_to_i":
                    src, dst = j, i
                elif current_dir == "sign":
                    src, dst = (i, j) if Jij >= 0 else (j, i)
                else:
                    raise ValueError("current_dir must be 'i_to_j', 'j_to_i', or 'sign'")

                p0, p1 = xy[src], xy[dst]
                d = p1 - p0
                L = np.linalg.norm(d)
                if L == 0:
                    continue
                u = d / L
                perp = np.array([-u[1], u[0]])

                # Keep off atom outline circles
                start = p0 + u * atom_radius
                end = p1 - u * atom_radius

                # Shaft linewidth scales strongly with magnitude
                lw = (lw_min + (lw_max - lw_min) * rel) * current_scale

                # Head size scales weakly with magnitude (keeps head subtle)
                hl = head_length * (0.7 + 0.6 * rel) * current_scale
                hw = head_width  * (0.7 + 0.6 * rel) * current_scale

                # Stop shaft at the base of the head
                end_shaft = end - u * hl

                shaft_segs.append([start, end_shaft])
                shaft_lws.append(lw)

                # Triangle head (tip at end, base centered at end_shaft)
                base_center = end_shaft
                left = base_center + perp * (hw / 2.0)
                right = base_center - perp * (hw / 2.0)
                tri = np.vstack([end, left, right])

                ax.add_patch(
                    Polygon(
                        tri,
                        closed=True,
                        facecolor="red",
                        edgecolor="none",
                        alpha=current_alpha,
                        zorder=4,
                    )
                )

            # Draw all shafts in one collection for clean rendering
            if shaft_segs:
                ax.add_collection(
                    LineCollection(
                        shaft_segs,
                        colors="red",
                        linewidths=shaft_lws,
                        alpha=current_alpha,
                        zorder=4,
                    )
                )

    # --- Ax formatting ---
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.margins(0.1)
    #ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("z (Å)")
    ax.set_ylabel("-x (Å)")
    ax.autoscale_view()
    #ax.margins(0.1)

    return fig, ax
