import numpy as np
from torch import nn
import torch
import seaborn as sns
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
from skimage import exposure, transform
from branching_model import Agent
import colour
import importlib

importlib.reload(Agent)
LEARNING_RATE = 10**-1


TIME_COL = "time"
SIZE_COL = "size"
MUTATION_SIZE_COL = "mutation_size"
PHENO_COL = "phenotype"
ID_COL = "id"
CLONE_ID_COL = "clone_id"
PARENT_ID_COL = "parent_id"
VAF_COL = "vaf"
ROOT_PARENT_ID = -1

CLONE_STR = "clone"
CELL_STR = "cell"

PHENO_CSPACE = "JzAzBz"
PHENO_CRANGE = (0, 0.025)
PHENO_LRANGE = (0.004, 0.015)



"""
This module provides the GenBary class to visualize high dimensional
data in 2 dimensions using generalized barycentric coordinates.
"""
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull


class GenBary:
    """
    copied from https://github.com/fraunhofer-izi/multi_bary_plot/blob/master/multi_bary_plot/GenBary.py

    This class can turn n-dimensional data into a
    2-d plot with generalized barycentric coordinates.

    Parameters
    ----------
    data : pandas.DataFrame
        Coordinates in at least 3 dimensions and an optional
        value column.
    value_column : string, optional
        The name of the optional value column in the `data`.
        If no value column is given, `imshow` is not available
        and `scatter` does not color the points automatically.
    coordinate_columns : list of strings, optional
        The coloumns of data that contain the positional values.
        If None is given, all columns but the `value_column` are
        used as `coordinate_columns`.
    res : int, optional
        The number of pixels along one axes; defaults to 500.
    ticks : list of numericals, optional
        The ticks of the colorbar.

    Returns
    -------
    GenBary : instance
        An instance of the GenBary.

    Usage
    -----
    vec = list(range(100))
    pdat = pd.DataFrame({'class 1':vec,
                         'class 2':list(reversed(vec)),
                         'class 3':[50]*100,
                         'val':vec})
    bp = GenBary(pdat, 'val')
    fig, ax, im = bp.plot()
    """

    def __init__(self, data, value_column=None, coordinate_columns=None,
                 res=500, ticks=None):
        if value_column is not None and \
           value_column not in data.columns.values:
            raise ValueError('`value_column` must be '
                             + 'a column name of `data`.')
        if coordinate_columns is not None:
            if not isinstance(coordinate_columns, list) or \
               len(coordinate_columns) < 3:
                raise ValueError('`coordinate_columns` must be a list'
                                 + 'of at least three column names of `data`.')
        if coordinate_columns is not None and \
           not all([cc in data.columns.values for cc in coordinate_columns]):
            raise ValueError('All `coordinate_columns` must be '
                             + 'column names of `data`.')
        if not isinstance(res, (int, float)):
            raise ValueError('`res` must be numerical.')
        self.res = int(res)
        numerical = ['float64', 'float32', 'int64', 'int32']
        if not all([d in numerical for d in data.dtypes]):
            raise ValueError('The data must be numerical.')
        if value_column is None and coordinate_columns is None:
            coords = data
            self.values = None
        elif coordinate_columns is None:
            coords = data.drop([value_column], axis=1)
            self.values = data[value_column].values
        elif value_column is None:
            coords = data[coordinate_columns]
            self.values = None
        else:
            coords = data[coordinate_columns]
            self.values = data[value_column].values
        self.ticks = ticks
        norm = np.sum(coords.values, axis=1, keepdims=True)
        ind = np.sum(np.isnan(coords), axis=1) == 0
        ind = np.logical_and(ind, (norm != 0).flatten())
        if self.values is not None:
            ind = np.logical_and(ind, ~np.isnan(self.values))
            self.values = self.values[ind]
        norm = norm[ind]
        coords = coords[ind]
        self.coords = coords.values / norm
        self.vert_names = list(coords.columns.values)
        self.nverts = self.coords.shape[1]
        if self.nverts < 3:
            raise ValueError('At least three dimensions are needed.')

    @property
    def grid(self):
        """The grid of pixels to raster in imshow."""
        x = np.linspace(-1, 1, self.res)
        return np.array(np.meshgrid(x, 0-x))

    @property
    def mgrid(self):
        """Melted x and y coordinates of the pixel grid."""
        grid = self.grid
        return grid.reshape((grid.shape[0],
                             grid.shape[1]*grid.shape[2]))

    @property
    def vertices(self):
        """The vertices of the barycentric coordinate system."""
        n = self.nverts
        angles = np.array(range(n))*np.pi*2/n
        vertices = [[np.sin(a), np.cos(a)] for a in angles]
        vertices = pd.DataFrame(vertices, columns=['x', 'y'],
                                index=self.vert_names)
        return vertices

    @property
    def hull(self):
        """The edges of the confex hull for plotting."""
        return ConvexHull(self.vertices).simplices

    @property
    def points_2d(self):
        """The 2-d coordinates of the given points."""
        parts = np.dot(self.coords, self.vertices)
        pdat = pd.DataFrame(parts, columns=['x', 'y'])
        pdat['val'] = self.values
        return pdat

    def _vals_on_grid(self):
        """The unmasked pixel colors."""
        p2 = self.points_2d
        dist = cdist(self.mgrid.T, p2[['x', 'y']].values)
        ind = np.argmin(dist, axis=1)
        vals = p2['val'][ind]
        return vals.values.reshape(self.grid.shape[1:])

    @property
    def in_hull(self):
        """A mask of the grid for the part outside
        the simplex."""
        pixel = self.mgrid.T
        inside = np.repeat(True, len(pixel))
        for simplex in self.hull:
            vec = self.vertices.values[simplex]
            vec = vec.mean(axis=0, keepdims=True)
            shifted = pixel - vec
            below = np.dot(shifted, vec.T) < 0
            inside = np.logical_and(inside, below.T)
        return inside.reshape(self.grid.shape[1:])

    @property
    def plot_values(self):
        """The Pixel colors masked to the inside of
        the barycentric coordinate system."""
        values = self._vals_on_grid()
        return np.ma.masked_where(~self.in_hull, values)

    @property
    def text_position(self):
        """Dimensions label positions in plot."""
        half = int(np.floor(self.nverts/2))
        odd = (self.nverts & 1) == 1
        tp = self.vertices.copy() * 1.05
        i = tp.index
        tp['v_align'] = 'center'
        tp.loc[i[0], 'v_align'] = 'bottom'
        tp.loc[i[half], 'v_align'] = 'top'
        if odd:
            tp.loc[i[half+1], 'v_align'] = 'top'
        tp['h_align'] = 'center'
        tp.loc[i[1:half], 'h_align'] = 'left'
        tp.loc[i[half+1+odd:], 'h_align'] = 'right'
        return tp

    def draw_polygon(self, ax=None):
        """Draws the axes and lables of the coordinate system."""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        vertices = self.vertices
        for simplex in self.hull:
            ax.plot(vertices.values[simplex, 0],
                    vertices.values[simplex, 1], 'k-')
        for index, row in self.text_position.iterrows():
            ax.text(row['x'], row['y'], index,
                    ha=row['h_align'], va=row['v_align'])
        return ax

    def imshow(self, colorbar=True, fig=None, ax=None, **kwargs):
        """

        Plots the data in barycentric coordinates and colors pixels
        according to the closest given value.

        Parameters
        ----------
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axes is not None.
        fig : matplotlib.figure, optional
            The figure to plot in.
        ax : matplotlib.axes, optional
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.imshow.

        Returns
        -------
        fig, ax, im
            The matplotlib Figure, AxesSubplot,
            and AxesImage of the plot.

        """
        if self.values is None:
            raise ValueError('No value column supplied.')
        if fig is None and ax is not None and colorbar:
            warnings.warn('axes but no figure is supplied,'
                          + ' so a colorbar cannot be plotted.')
            colorbar = False
        elif fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.axis('off')
        im = ax.imshow(self.plot_values, extent=[-1, 1, -1, 1], **kwargs)
        ax = self.draw_polygon(ax)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            fig.colorbar(im, cax=cax, orientation='horizontal',
                         ticks=self.ticks)
        # manual limits because of masked data
        v = self.vertices
        xpad = (v['x'].max()-v['x'].min()) * .05
        ax.set_xlim([v['x'].min()-xpad, v['x'].max()+xpad])
        ypad = (v['y'].max()-v['y'].min()) * .05
        ax.set_ylim([v['y'].min()-ypad, v['y'].max()+ypad])
        ax.set_aspect('equal')
        return fig, ax, im

    def scatter(self, color=None, colorbar=None, fig=None,
                ax=None, **kwargs):
        """

        Scatterplot of the data in barycentric coordinates.

        Parameters
        ----------
        color : bool, optional
            Color points by given values. Ignored if no value column
            is given.
        colorbar : bool, optional
            If true a colorbar is plotted on the bottom of the image.
            Ignored if figure is None and axes is not None.
        fige : matplotlib.figure, optional
            The figure to plot in.
        ax : matplotlib.axes, optional
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.scatter. The keyword argument c
            overwrites given values in the data.

        Returns
        -------
        fig, ax, pc
            The matplotib Figure, AxesSubplot,
            and PathCollection of the plot.

        """
        color_info = self.values is not None or 'c' in kwargs.keys()
        if color is None and color_info:
            color = True
        elif color is None:
            color = False
        if color and not color_info:
            raise ValueError('No value column for color supplied.')
        if color and colorbar is None:
            colorbar = True
        elif colorbar is None:
            colorbar = False
        if fig is None and ax is not None and colorbar:
            warnings.warn('axes but no figure is supplied,'
                          + ' so a colorbar cannot be plotted.')
            colorbar = False
        elif fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')
        ax.axis('off')
        p2 = self.points_2d
        if color and 'c' not in kwargs.keys():
            pc = ax.scatter(p2['x'], p2['y'], c=p2['val'], **kwargs)
        else:
            pc = ax.scatter(p2['x'], p2['y'], **kwargs)
        ax = self.draw_polygon(ax)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=.2)
            fig.colorbar(pc, cax=cax, orientation='horizontal',
                         ticks=self.ticks)
        return fig, ax, pc

    def plot(self, fig=None, ax=None, **kwargs):
        """

        Plots the data in barycentric coordinates.

        Parameters
        ----------
        fig : matplotlib.figure, optional
            The figure to plot in.
        ax : matplotlib.axes, optional
            The axes to plot in.
        **kwargs
            Other keyword arguments are passed on to
            matplotlib.pyplot.plot.

        Returns
        -------
        fig, ax, ll
            The matplotlib Figure, AxesSubplot,
            and list of Line2D of the plot.

        """
        if fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')
        ax.axis('off')
        p2 = self.points_2d
        ll = ax.plot(p2['x'], p2['y'], **kwargs)
        ax = self.draw_polygon(ax)
        return fig, ax, ll


def color_dxdy(dx, dy, c_range=PHENO_CRANGE, l_range=PHENO_LRANGE, cspace=PHENO_CSPACE):
    """
   Color displacement, where larger displacements are more colorful,
   and, if scale_l=True,  brighter.

    Parameters
    ----------
    dx: array
        1D Array containing the displacement in the X (column) direction

    dy: array
        1D Array containing the displacement in the Y (row) direction

    c_range: (float, float)
        Minimum and maximum colorfulness in JzAzBz colorspace

    l_range: (float, float)
        Minimum and maximum luminosity in JzAzBz colorspace

    scale_l: boolean
        Scale the luminosity based on magnitude of displacement

    Returns
    -------
    displacement_rgb : array
        RGB (0, 255) color for each displacement, with the same shape as dx and dy

    """

    initial_shape = dx.shape

    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    if np.all(dx==0) and np.all(dy==0):
        # No displacements. Return grey image
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            bg_rgb = colour.convert(np.dstack([l_range[0], 0, 0]), cspace, 'sRGB')*255

        displacement_rgb = np.full((*initial_shape, 3), bg_rgb).astype(np.uint8)
        return displacement_rgb

    eps = np.finfo("float").eps
    magnitude = np.sqrt(dx ** 2 + dy ** 2 + eps)
    C = exposure.rescale_intensity(magnitude, in_range=(0, magnitude.max()), out_range=tuple(c_range))
    H = np.arctan2(dy.T, dx.T)
    A, B = C * np.cos(H), C * np.sin(H)
    J = exposure.rescale_intensity(magnitude, in_range=(0, magnitude.max()), out_range=tuple(l_range))

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(np.dstack([J, A+eps, B+eps]), cspace, 'sRGB')

    displacement_rgb = (255*np.clip(rgb, 0, 1)).astype(np.uint8).reshape((*initial_shape, 3))

    return displacement_rgb


def displacement_legend(wh=500):

    X = np.linspace(-1, 1, wh)
    Y = np.linspace(-1, 1, wh)

    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    C = np.sin(R)

    C = exposure.rescale_intensity(C, out_range=(0, 1))

    grad = np.linspace(-1, 1, X.shape[0])
    grad = np.resize(grad, X.shape)
    dx = grad*C
    dy = grad.T * C

    displacement_legend = color_dxdy(dx, dy, PHENO_CRANGE, PHENO_LRANGE, cspace=PHENO_CSPACE)

    return displacement_legend


def get_square_verts():
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 1, 0])

    return np.dstack([x, y])






class Recorder(object):
    def __init__(self):
        self.clone_mutation_df = None
        self.cell_mutation_df = None
        self.size_df = None

    def get_id(self, agent, resolution=CLONE_STR):
        if resolution == CLONE_STR:
            mut_id = agent.clone_id
        else:
            mut_id = agent.id
        return mut_id

    def get_mutation_df(self, agent_list, time_pt, resolution=CLONE_STR):

        id_dict = {a.id: self.get_id(a, resolution=resolution) for a in agent_list}
        edge_dict = {}

        mutant_ids = np.unique(list(id_dict.values()))
        mutant_ids = sorted(mutant_ids)
        clone_clusters = {i: [] for i in mutant_ids}
        for agent in agent_list:
            mut_id = id_dict[agent.id]
            clone_clusters[mut_id].append(agent)
            parent = agent.parent
            if parent is None:
                edge_dict[mut_id] = ROOT_PARENT_ID
            while parent is not None:
                assert parent != agent
                try:
                    parent_mut_id = id_dict[parent.id]
                    clone_clusters[parent_mut_id].append(agent)

                except KeyError:
                    parent_mut_id = self.get_id(parent, resolution)
                    # if these are dead agents, then it could be that the parent isn't dead too

                if parent_mut_id != mut_id:
                    if mut_id not in edge_dict:
                        edge_dict[mut_id] = parent_mut_id

                parent = parent.parent

        # No time to figure out why clones end up in the same clone list multiple times :(
        filtered_clone_clusters = {idx: set(clone_clusters[idx]) for idx in clone_clusters}
        mutation_sizes = {idx: sum([1 if a.status == CELL_STR else a.n_cells for a in filtered_clone_clusters[idx]]) for idx in mutant_ids}
        phenotypes = {idx: np.vstack([a.phenotype.detach().numpy() for a in filtered_clone_clusters[idx]]).mean(axis=0)  for idx in mutant_ids}
        n_pheno = len(phenotypes[mut_id])
        phenotype_cols = ["S", *[f"R{i}" for i in range(1, n_pheno)]]

        phenotype_df = pd.DataFrame([phenotypes[idx] for idx in mutant_ids], columns=phenotype_cols)
        phenotype_df[CLONE_ID_COL] = mutant_ids

        mutation_df = pd.DataFrame(
            {
                TIME_COL: time_pt,
                CLONE_ID_COL: mutant_ids,
                PARENT_ID_COL: [edge_dict[idx] for idx in mutant_ids],
                MUTATION_SIZE_COL: [mutation_sizes[idx] for idx in mutant_ids]
            }

        )

        mutation_df = mutation_df.merge(phenotype_df, on=CLONE_ID_COL)

        return mutation_df

    def add_treatment_col(self, df, doses):

        doses_np = doses.squeeze().detach().numpy()
        doses_df = pd.DataFrame(np.vstack([doses_np for i in range(df.shape[0])]),
                                columns=[f"T{i+1}" for i in range(len(doses_np))])

        df = df.join(doses_df)

        return df


    def record_time_pt(self, agent_list, time_pt, doses):

        sizes = {agent.id: 0 for agent in agent_list}
        phenotypes = {agent.id: None for agent in agent_list}
        clone_ids = {agent.id: agent.clone_id for agent in agent_list}

        agent_ids = [agent.id for agent in agent_list]

        edge_dict = {
            agent.id: agent.parent.id if agent.parent is not None else ROOT_PARENT_ID
            for agent in agent_list
        }

        for agent in agent_list:
            size = 1 if agent.status == "cell" else agent.n_cells
            # mutation_sizes[agent.id] += size
            sizes[agent.id] = size
            phenotypes[agent.id] = agent.phenotype.detach().numpy()

        phenotypes = np.vstack([phenotypes[idx] for idx in agent_ids])
        phenotype_cols = ["S", *[f"R{i}" for i in range(1, phenotypes.shape[1])]]
        _pheno_df = pd.DataFrame(phenotypes, columns=phenotype_cols)

        time_pt_size_df = pd.DataFrame(
            {
                TIME_COL: time_pt,
                ID_COL: agent_ids,
                CLONE_ID_COL: [clone_ids[idx] for idx in agent_ids],
                PARENT_ID_COL: [edge_dict[idx] for idx in agent_ids],
                SIZE_COL: [sizes[idx] for idx in agent_ids]
            }
        ).join(_pheno_df)

        time_pt_size_df = self.add_treatment_col(time_pt_size_df, doses)

        time_pt_clone_mutation_df = self.get_mutation_df(agent_list=agent_list, time_pt=time_pt, resolution=CLONE_STR)
        time_pt_clone_mutation_df = self.add_treatment_col(time_pt_clone_mutation_df, doses)

        clone_res_vaf = self.calculate_vaf(time_pt_size_df, time_pt_clone_mutation_df)
        vaf_col_pos = list(time_pt_clone_mutation_df).index(MUTATION_SIZE_COL) + 1
        time_pt_clone_mutation_df.insert(loc=vaf_col_pos, column=VAF_COL, value=clone_res_vaf)

        self.size_df = self.update_df(self.size_df, time_pt_size_df)
        self.clone_mutation_df = self.update_df(self.clone_mutation_df, time_pt_clone_mutation_df)

        if agent_list[0].status == CELL_STR:
            time_pt_cell_mutation_df = self.get_mutation_df(agent_list=agent_list, time_pt=time_pt, resolution=CELL_STR)
            time_pt_cell_mutation_df = self.add_treatment_col(time_pt_cell_mutation_df, doses)
            cell_res_vaf = self.calculate_vaf(time_pt_size_df, time_pt_cell_mutation_df)
            vaf_col_pos = list(time_pt_cell_mutation_df).index(MUTATION_SIZE_COL) + 1
            time_pt_cell_mutation_df.insert(loc=vaf_col_pos, column=VAF_COL, value=cell_res_vaf)
            self.cell_mutation_df = self.update_df(self.cell_mutation_df, time_pt_cell_mutation_df)

    def update_df(self, df1, df2):
        if df1 is not None and df2 is not None:
            new_df = pd.concat([df1, df2])
        elif df1 is not None and df2 is None:
            new_df = df1
        elif df1 is None and df2 is not None:
            new_df = df2
        else:
            new_df = None

        return new_df

    def write_csv(self, dst_dir, prefix=""):
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

        size_fout = os.path.join(dst_dir, f"{prefix}_size.csv").replace(f"{os.sep}_", os.sep)
        if os.path.isfile(size_fout):
            os.remove(size_fout)
        self.size_df.to_csv(size_fout, index=False)

        clone_mutaton_fout = os.path.join(dst_dir, f"{prefix}_clone_mutations.csv").replace(f"{os.sep}_", os.sep)
        if os.path.isfile(clone_mutaton_fout):
            os.remove(clone_mutaton_fout)
        self.clone_mutation_df.to_csv(clone_mutaton_fout, index=False)

        if self.cell_mutation_df is not None:
            cell_mutaton_fout = os.path.join(dst_dir, f"{prefix}_cell_mutations.csv").replace(f"{os.sep}_", os.sep)
            if os.path.isfile(cell_mutaton_fout):
                os.remove(clone_mutaton_fout)
            self.cell_mutation_df.to_csv(cell_mutaton_fout, index=False)

    def long_to_wide(self, cname):
        info_cols = [ID_COL, PARENT_ID_COL]
        info_df = self.df[info_cols].drop_duplicates()

        long_df = self.df[[ID_COL, TIME_COL, cname]]
        wide_df = long_df.pivot(index=ID_COL, columns=TIME_COL, values=cname)
        wide_df.fillna(0, inplace=True)
        wide_df = info_df.merge(wide_df, on=ID_COL)

        return wide_df

    def calculate_vaf(self, size_df, mutation_df):
        """
        resolution : str
            cell or clone
        """

        n_cells = sum(size_df[SIZE_COL])
        n_seq = 2*n_cells
        vaf = mutation_df[MUTATION_SIZE_COL]/n_seq
        return vaf


if __name__ == "__main__":

    def generate_tree(n_agents=10):
        recorder = Recorder()
        self = recorder
        agent_list = []
        initial_cell = Agent.Agent(is_cell=True, id=0, clone_id=0, learning_rate=LEARNING_RATE)
        initial_cell.time_created = 0
        agent_list.append(initial_cell)
        # current_max_id = initial_cell.id
        n_created = 1
        n_clones = 1
        treatment_time = 10
        no_doses = torch.zeros(Agent.N_TREATMENTS).reshape(1, -1)
        # np_treatment = np.zeros(Agent.N_TREATMENTS)
        # np_treatment[0] = 1
        # treament_dose = torch.tensor(np_treatment).reshape(1, -1)
        treament_dose = torch.zeros_like(no_doses)
        treament_dose[0][0] = 1.0
        doses = no_doses
        time = 0
        while n_clones < n_agents:
            if time > treatment_time:
                doses = treament_dose
            for cell in agent_list:
                cell.update_phenotype(doses)
                p = cell.calc_growth_rate(cell.phenotype, doses, resistance_cost=0.2, resistance_benefit=0.5)
                cell.p = p
                if hasattr(cell, "pheno_changes"):
                    cell.pheno_changes.append(cell.phenotype.detach().numpy())
                else:
                    cell.pheno_changes = [cell.phenotype.detach().numpy()]
                if p > 0:
                    divide = np.random.binomial(1, p) == 1
                    if divide:
                        mutate_cell = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                        if mutate_cell:
                            n_created += 1
                            mutate_clone = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                            if mutate_clone:
                                n_clones += 1

                            new_cell = cell.copy(new_id=n_created-1, new_clone_id=n_clones-1)
                            new_cell.mutate()

                            new_cell.time_created = time
                            agent_list.append(new_cell)

            recorder.record_time_pt(agent_list, time_pt=time, doses=doses)
            time += 1

        return recorder, agent_list


    recorder, agent_list = generate_tree()
    self = recorder

    dst_dir = os.path.join(os.getcwd(), "tests/csv_files")
    recorder.write_csv(dst_dir, prefix="live")

    time_df = recorder.size_df[recorder.size_df.time==40]
    pheno_cols = [x for x in list(time_df) if x == "S" or x.startswith("R")]
    pheno_df = time_df[pheno_cols]
    # pheno_df.loc[len(pheno_df.index)] = np.zeros(len(pheno_cols))
    # pheno_df.loc[len(pheno_df.index)] = np.ones(len(pheno_cols))

    projector = GenBary(pheno_df, coordinate_columns=pheno_cols)
    pheno2d = projector.points_2d

    pheno_map = displacement_legend()

    # Warp coords to be on un-rotated square
    square_verts = get_square_verts()
    rotator = transform.SimilarityTransform()
    rotator.estimate(square_verts, projector.vertices.values)

    projector.vert_names

    plt.scatter(projector.vertices["x"], projector.vertices["y"])
    plt.show()





    # plt.scatter(pheno2d.x, pheno2d.y)
    # plt.show()


