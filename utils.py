import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import plotly.graph_objects as go
from colorsys import rgb_to_hls

def largest_connected_component(A, verbose=False):
    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)
    component_sizes = np.bincount(labels)
    lcc_label = np.argmax(component_sizes)
    lcc_mask = labels == lcc_label

    if verbose:
        print(f"Size of graph: {A.shape[0]}")
        print(f"Number of connected components: {n_components}")
        print(f"Size of largest connected component: {np.max(component_sizes)}")
    if len(component_sizes) > 1:
        print(f"Size of other connected components: {np.sort(component_sizes)[::-1]}")

    return lcc_mask



def knn_graph(X, k, directed=False):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    rows = []
    cols = []
    data = []

    for i, (neighbors, dist) in enumerate(zip(indices, distances)):
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
        data.extend(dist)

    A = csr_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]))

    if not directed:
        A = A.maximum(A.T)

    return A

def chatterjee_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    if n < 2:
        raise ValueError("Need at least two observations to compute correlation.")
    
    # 1) Order the indices of x
    idx = np.argsort(x)
    # 2) Compute (0-based) ranks of y
    rank_y = np.argsort(np.argsort(y))
    # 3) Reorder y-ranks by increasing x
    r_ordered = rank_y[idx]
    # 4) Sum absolute successive differences
    diffs = np.abs(np.diff(r_ordered))
    S = np.sum(diffs)
    # 5) Plug into Chatterjee’s formula
    return 1 - (3 * S) / (n**2 - 1)

def rotate_3d(points, rotation, units="degrees"):
    theta_x, theta_y, theta_z = rotation
    if units == "degrees":
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
    elif units != "radians":
        raise ValueError("Units must be 'degrees' or 'radians'")

    # Rotation matrices around x, y, z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
                   
    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Rotate points
    rotated_points = np.dot(points, R.T)
    return rotated_points




def interactive_3d_plot(points, labels=None, color_values=None, title="", point_size=5, opacity=1, colormap='Viridis'):

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=color_values,  # Color by the Y values
            colorscale=colormap,
            opacity=opacity
        ),
        text=labels,  # Add statements for hover text
        hoverinfo='text',
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        title=title
    )

    fig.show()

    return None



def plot_scatter(x, y, colors=None, cmap='viridis', y_min=None, y_max=None,
                        xlabel='', ylabel='',
                        title=None, figsize=(6,6), dpi=300,
                        marker_size=2, alpha=0.1, text_box=None):
    """
    Enhanced scatter plot:
    - colors: hex codes (strings) or numeric array (mapped via `cmap`)
    - y_max:  maximum y-axis value (axes start at 0 if provided)
    - x-axis always starts from 0; max is auto
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Determine color mapping
    if colors is not None:
        arr = np.array(colors)
        if arr.dtype.kind in {'U','S','O'}:
            sc = ax.scatter(x, y, s=marker_size, alpha=alpha, color=list(arr))
        else:
            sc = ax.scatter(x, y, s=marker_size, alpha=alpha, c=arr, cmap=cmap)
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label('Value', rotation=-90, va="bottom")
    else:
        sc = ax.scatter(x, y, s=marker_size, alpha=alpha)
    
    # Convert to arrays
    x_arr = np.array(x)
    y_arr = np.array(y)

    # Axis limits
    x_min, x_max = 0, np.max(x_arr)
    y_min_plot = y_min if y_min is not None else 0
    y_max_plot = y_max if y_max is not None else np.max(y_arr)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_plot, y_max_plot)
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add custom text box in bottom-right
    if text_box:
        bbox_props = dict(boxstyle='square,pad=0.75', facecolor='white', edgecolor='black', alpha=1)
        ax.text(0.95, 0.05, text_box,
                transform=ax.transAxes,
                fontsize=10,
                va='bottom', ha='right',
                bbox=bbox_props)
    

    # Uniform black spines on all sides
    lw = ax.spines['left'].get_linewidth()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color('black')
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    fig.tight_layout()
    return fig, ax

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def distance_plot(
    DX, DY, labels, cmap = plt.cm.viridis, colors=None,
    corr_coef='pearson', square_distances=True, xlabel="",
    ylabel=""):

    xs = []
    ys = []
    ls = []
    ns = []
    for i in range(len(DY)):
        for d in np.unique(DY[i,:]):
            x = d
            y = np.mean(DX[i,DY[i,:] == d])
            if colors is None:
                l = labels[i]
            else:
                l = colors[i]
            xs.append(x)
            ys.append(y)
            ls.append(l)

    xs = np.array(xs)
    ys = np.array(ys)
    ls = np.array(ls)

    # shuffle the data
    random_indices = np.random.permutation(len(xs))
    xs = xs[random_indices]
    ys = ys[random_indices]
    ls = ls[random_indices]

    if colors is None:
        norm = plt.Normalize(min(ls), max(ls))
        rgb_colors = cmap(norm(ls))

        plt_colors = []
        for r, g, b, a in rgb_colors:
            plt_colors.append(rgb_to_hex((r, g, b)))
        plt_colors = np.array(plt_colors)
    else:
        plt_colors = ls

    # compute the correlation
    if corr_coef == 'pearson':
        corr = np.corrcoef(xs, ys)[0, 1]
        corr_txt = fr"$\rho = {corr:.3f}$"
    elif corr_coef == 'chatterjee':
        corr = chatterjee_corr(xs, ys)
        corr_txt = fr"$\xi = {corr:.3f}$"
    else:
        raise ValueError("corr_coef must be 'pearson' or 'chatterjee'")

    # square the distances if requested
    if square_distances:
        xs = xs**2

    # create the plot
    fig, ax = plot_scatter(
        xs, ys, alpha=0.1, marker_size=0.1,
        xlabel=xlabel,
        ylabel=ylabel,
        colors = plt_colors,
        y_min=min(ys),
        text_box = corr_txt,
        figsize=(3.5,3.5),
    )

    return fig, ax



def hex_to_rgb(hex_codes):
    # Remove the '#' if it exists and convert hex codes to RGB values
    rgb_values = np.array([tuple(int(hex_code.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for hex_code in hex_codes])
    return rgb_values

def hex_to_hls(hex_codes):
    # Convert hex codes to RGB values
    rgb_values = hex_to_rgb(hex_codes)
    # Normalize RGB values to the range [0, 1]
    rgb_normalized = rgb_values / 255.0
    # Convert RGB to HLS
    hls_values = np.array([rgb_to_hls(r, g, b) for r, g, b in rgb_normalized])
    return hls_values
