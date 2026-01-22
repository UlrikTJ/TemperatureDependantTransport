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

    ax=None,
):
    """
    Plot ASE atoms in the xz-plane with:
      - Atom circles (black outline, no fill)
      - Bonds to neighbors within `cutoff` (thin black lines)
      - Optional wavefunction overlay (psi on atoms):
          * color by phase (HSV wheel)
          * radius by probability (|psi| or |psi|^2 depending on prob_mode)
      - Optional bond-current arrows from matrix J:
          * drawn as shaft line + small triangular head ("paper-style")
          * size scales with |J_ij|
          * direction controlled by current_dir for i>j entries

    Parameters
    ----------
    atoms : ase.Atoms
    cutoff : float
        Bond cutoff in Å.
    atom_radius : float
        Radius of the atom outline circles (in plot coordinate units, Å).

    psi : (N,) complex array or None
        Wavefunction values on atoms. If None, wavefunction overlay is omitted.
    wf_scale : float
        Global scale for wavefunction circle radii.
    wf_alpha : float
        Alpha for wavefunction circles.
    prob_mode : {"sqrt","linear"}
        "sqrt": radius ~ |psi| (default)
        "linear": radius ~ |psi|^2
    show_phase_colorbar : bool
        Show phase colorbar if psi is provided.

    J : (N,N) array or None
        Bond current matrix. If None, current arrows are omitted.
    current_dir : {"i_to_j","j_to_i","sign"}
        For each pair (i>j):
          - "i_to_j": arrow i -> j
          - "j_to_i": arrow j -> i
          - "sign": direction depends on sign of J[i,j]
                    (positive i->j, negative j->i)
    current_scale : float
        Global scaling of arrow linewidths and head sizes.
    current_alpha : float
        Alpha for current arrows.
    current_threshold : float
        Skip arrows with |J_ij| <= threshold.
    currents_on_bonds_only : bool
        If True, only draw current arrows for pairs within `cutoff`.

    head_length, head_width : float
        Base arrowhead geometry in Å (scaled weakly with |J|).
    lw_min, lw_max : float
        Shaft linewidth range (scaled with |J|).

    ax : matplotlib Axes or None
        If None, creates a new figure.

    Returns
    -------
    fig, ax
    """
    pos = atoms.get_positions()
    xy = pos[:, [0, 2]].astype(float)
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
    for i in range(N):
        ax.add_patch(
            Circle(
                xy[i],
                radius=atom_radius,
                edgecolor="black",
                facecolor="none",
                linewidth=1.0,
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
                        facecolor="black",
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
                        colors="black",
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
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("z (Å)")
    ax.autoscale_view()
    #ax.margins(0.1)

    return fig, ax
