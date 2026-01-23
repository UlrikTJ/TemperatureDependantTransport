import numpy as np
import nglview as nv
from matplotlib import cm
import nglview as nv
import ipywidgets as widgets
from ipywidgets import interactive

def plotstate(atoms1, psi, scale=1.0, atom_sizes=None,
              cmap_name="twilight",
              sphere_opacity=0.8,
              bond_style="licorice",
              bond_color="black",
              bond_width=2):
    """
    NGLview visualization of a (complex) wavefunction on atoms.

    - Phase(arg psi) -> color (cyclic colormap)
    - |psi| -> sphere radius
    - Semi-transparent spheres
    - Bond lines shown

    Parameters
    ----------
    atoms1 : ase.Atoms
    psi : complex array, shape (n_atoms,)
    scale : float
        Overall radius scaling
    atom_sizes : array-like or None
        Base atom radii (defaults to 1.0 for all)
    cmap_name : str
        Cyclic colormap name ("twilight" recommended)
    sphere_opacity : float in (0,1)
        Opacity of wavefunction spheres
    bond_style : {"line", "licorice", "ball+stick"}
    bond_color : str
    bond_width : float
        Linewidth for "line" bonds

    Returns
    -------
    view : nglview.NGLWidget
    """
    

    psi = np.asarray(psi)
    na = len(atoms1)
    if psi.shape[0] != na:
        raise ValueError("psi length must match number of atoms")

    if atom_sizes is None:
        atom_sizes = np.ones(na)
    atom_sizes = np.asarray(atom_sizes, float)

    # Create view
    view = nv.show_ase(atoms1)

    # Phase and amplitude
    phase = np.angle(psi)                       # [-pi, pi]
    t = (phase + np.pi) / (2*np.pi)             # [0,1]
    amp = np.abs(psi)
    amp /= amp.max() + 1e-15

    cmap = getattr(cm, cmap_name)

    # Remove default representation once
    view.clear_representations()

    # Wavefunction spheres (reliable per-atom reps)
    for ia in range(na):
        r, g, b, _ = cmap(float(t[ia]))
        color = f"#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}"

        radius = float(atom_sizes[ia] * (0.3 + 0.7 * amp[ia]) * scale)

        view.add_representation(
            "spacefill",
            selection=[ia],
            color=color,
            radius=radius,
            opacity=float(sphere_opacity)
        )

    # Bonds / connectivity
    if bond_style == "line":
        view.add_representation(
            "line",
            color=bond_color,
            linewidth=bond_width
        )
    elif bond_style == "licorice":
        view.add_representation(
            "licorice",
            color=bond_color,
            radius=0.15
        )
    elif bond_style == "ball+stick":
        view.add_representation("ball+stick")
    
    view.add_representation("licorice", color="grey", radius=0.15)
    return view



def _shape_add_arrow_compat(shape, p0, p1, color, radius=0.1, opacity=1.0):
    """
    Call shape.add_arrow with whichever signature this nglview version supports.
    """
    # Try newest-style kwargs first
    try:
        return shape.add_arrow(p0, p1, color=color, radius=radius, opacity=opacity)
    except TypeError:
        pass

    # Try positional: (p0, p1, color, radius, opacity)
    try:
        return shape.add_arrow(p0, p1, color, radius, opacity)
    except TypeError:
        pass

    # Try positional without opacity: (p0, p1, color, radius)
    try:
        return shape.add_arrow(p0, p1, color, radius)
    except TypeError:
        pass

    # Fallback: (p0, p1, color)
    return shape.add_arrow(p0, p1, color)


def add_bond_current_arrows_from_matrix(
    view, atoms, J,
    threshold=0.0,
    symmetric_mode="upper",
    direction_axis="auto",
    direction_rule="matrix",
    scale=1.0,
    trim=0.25,
    rmin=0.03, rmax=0.20,
    color_pos=(1.0, 0.0, 0.0),   # RGB in [0,1]
    color_neg=(0.0, 0.0, 1.0),
    max_arrows=None,
    scale_by="max",
):
    """
    Draw bond-current arrows from an NxN current matrix J using nglview.shape.add_arrow.

    Your nglview version expects:
        shape.add_arrow(position1, position2, color_rgb, radius, name=None)

    Parameters
    ----------
    view : nglview.NGLWidget
        Existing widget (e.g. from your plotstate()).
    atoms : ase.Atoms
        Structure. PBC ignored (as requested).
    J : (N, N) array-like float
        Bond current matrix. Typical conventions:
        - Directed: J[i,j] is current from i -> j (often antisymmetric: J = -J^T).
        - You decide what sign means; this function visualizes it.

    threshold : float
        Only draw arrows for entries with |J[i,j]| >= threshold.

    symmetric_mode : {"upper", "all"}
        How to interpret / traverse the matrix:
        - "upper": consider only pairs with i < j.
          Use this when your matrix has redundancy (e.g. antisymmetric/symmetric)
          to avoid drawing two arrows per bond.
        - "all": consider all i != j.
          Use this if you truly store independent directed currents in both J[i,j] and J[j,i].

        Practical tip:
        If J is antisymmetric (J[i,j] = -J[j,i]) then "upper" is almost always what you want.

    direction_axis : {"auto","x","y","z"} or array-like (3,)
        Defines the global “positive direction” axis used by direction_rule="along_axis".
        - "auto" (default): pick the longest bounding-box axis of the structure.
          (i.e. whichever of x/y/z has the largest extent in atom positions)
        - "x","y","z": use the corresponding Cartesian axis.
        - custom vector: e.g. (1,1,0) or [0,0,1]. It will be normalized.

    direction_rule : {"matrix","along_axis"}
        Controls how arrow direction is chosen:

        - "matrix" (default):
            Arrow direction comes directly from the sign of J[i,j] for the chosen pair (i,j).
            If J[i,j] >= 0: arrow i -> j (colored color_pos)
            If J[i,j] <  0: arrow j -> i (colored color_neg)
            This is the natural choice if you trust the matrix convention “J[i,j] is i->j”.

        - "along_axis":
            Enforces a global notion of forward/backward along `direction_axis`.
            For each considered pair (i,j), we first define a canonical bond orientation:
                from the atom with smaller projection onto +axis  --> larger projection.
            Then we **project** the matrix sign onto that convention:
                J_eff = J[i,j]   if (r_j - r_i)·axis >= 0
                      = -J[i,j]  if (r_j - r_i)·axis <  0
            Finally:
                if J_eff >= 0: arrow along +axis (forward) using color_pos
                if J_eff <  0: arrow opposite (backward) using color_neg

            Use this when you want “positive current means flow to the right” (or along device axis),
            regardless of how (i,j) ordering was chosen.

    scale : float
        Overall visual scale factor multiplying radii and trim.

    trim : float (Å)
        Shorten arrows by `trim` at BOTH ends (so arrowheads don’t sit inside spheres).
        Increase if your atoms are big.

    rmin, rmax : float (Å)
        Minimum and maximum arrow radius. Actual radius scales with |J|.

    color_pos, color_neg : (3,) RGB tuple/list in [0,1]
        Arrow colors for positive/negative (after applying direction_rule).
        Note: your nglview expects numeric RGB, not strings.

    max_arrows : int or None
        If set, only draw the strongest `max_arrows` by |J| (after thresholding).
        Helps avoid clutter for dense matrices.

    scale_by : {"max","percentile90","percentile95"}
        How to map |J| to radii:
        - "max": radii scale to the maximum |J| encountered.
        - "percentile90"/"percentile95": more robust if you have outliers.

    Returns
    -------
    view : nglview.NGLWidget
        Same view, with arrows added.

    Notes / other ways to change the plot
    ------------------------------------
    - Reduce clutter: increase `threshold`, or set `max_arrows`.
    - Make arrows thicker/thinner: adjust `rmin`, `rmax`, `scale`.
    - Make arrows not collide with atoms: increase `trim`.
    - Change meaning of "positive": use `direction_rule="along_axis"` and set `direction_axis`.
    """

    import numpy as np

    pos = atoms.get_positions()
    J = -np.asarray(J, float).T
    N = len(atoms)
    if J.shape != (N, N):
        raise ValueError(f"J must have shape ({N},{N}), got {J.shape}")

    # --- choose axis ---
    if isinstance(direction_axis, str):
        da = direction_axis.lower()
        if da == "auto":
            mins = pos.min(axis=0)
            maxs = pos.max(axis=0)
            ext = maxs - mins
            k = int(np.argmax(ext))  # 0:x, 1:y, 2:z
            axis = np.zeros(3, float)
            axis[k] = 1.0
        elif da in ("x", "y", "z"):
            axis = np.zeros(3, float)
            axis[{"x": 0, "y": 1, "z": 2}[da]] = 1.0
        else:
            raise ValueError("direction_axis must be 'auto','x','y','z', or a length-3 vector")
    else:
        axis = np.asarray(direction_axis, float).reshape(3)
        nrm = np.linalg.norm(axis)
        if nrm < 1e-15:
            raise ValueError("direction_axis vector has near-zero norm")
        axis = axis / nrm

    # --- collect candidate edges ---
    edges = []
    vals = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if symmetric_mode == "upper" and not (i < j):
                continue
            v = J[i, j]
            if abs(v) >= threshold:
                edges.append((i, j))
                vals.append(v)

    if not edges:
        return view

    vals = np.asarray(vals, float)
    absvals = np.abs(vals)

    # Optionally keep only strongest arrows
    if max_arrows is not None and len(edges) > int(max_arrows):
        order = np.argsort(absvals)[::-1][: int(max_arrows)]
        edges = [edges[k] for k in order]
        vals = vals[order]
        absvals = absvals[order]

    # Robust scaling
    if scale_by == "max":
        denom = absvals.max()
    elif scale_by == "percentile90":
        denom = np.percentile(absvals, 90)
    elif scale_by == "percentile95":
        denom = np.percentile(absvals, 95)
    else:
        raise ValueError("scale_by must be 'max', 'percentile90', or 'percentile95'")
    denom = float(denom) if denom > 0 else 1.0

    def radius_from_abs(x):
        w = float(x) / denom
        if w > 1.0:
            w = 1.0
        return (rmin + (rmax - rmin) * w) * float(scale)

    # --- draw arrows ---
    for (i, j), val, absv in zip(edges, vals, absvals):
        ri, rj = pos[i], pos[j]
        bond = rj - ri

        if direction_rule == "matrix":
            # direction from sign of J[i,j]
            if val >= 0:
                p0, p1 = ri, rj
                col = color_pos
            if val < 0: 
                p0, p1 = rj, ri
                col = color_neg
            if val == 0:
                p0, p1 = rj, ri
                col = [0,0,0]

        elif direction_rule == "along_axis":
            # canonical forward direction along +axis based on projections
            si = float(np.dot(ri, axis))
            sj = float(np.dot(rj, axis))

            if sj >= si:
                forward_p0, forward_p1 = ri, rj
                proj_sign = 1.0 if float(np.dot(bond, axis)) >= 0 else -1.0
            else:
                forward_p0, forward_p1 = rj, ri
                proj_sign = 1.0 if float(np.dot(-bond, axis)) >= 0 else -1.0

            # effective value in "forward" convention
            J_eff = proj_sign * val

            if J_eff >= 0:
                p0, p1 = forward_p0, forward_p1
                col = color_pos
            if J_eff < 0:
                p0, p1 = forward_p1, forward_p0
                col = color_neg
            if J_eff == 0.:
                col = [0,0,0]
        else:
            raise ValueError("direction_rule must be 'matrix' or 'along_axis'")

        # trim
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 1e-12:
            continue
        u = v / L

        t = float(trim) * float(scale)
        p0t = (p0 + u * t).tolist()
        p1t = (p1 - u * t).tolist()

        view.shape.add_arrow(
            p0t,
            p1t,
            list(col),                 # numeric RGB list
            float(radius_from_abs(absv))
        )

    return view
