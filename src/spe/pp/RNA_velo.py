import numpy as np
import torch
from scipy.sparse import issparse


def _velovi_velocity(adata, n_samples: int=25, save_vae: bool=False, **velo_kwargs):
    
    try:
        from scvi.external.velovi import VELOVI
    except ImportError:
        raise ImportError('scvi-tools must be installed to use the spe.pp.RNA_velocity function with mode dynamo.')

    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(**velo_kwargs)

    latent_time = vae.get_latent_time(n_samples=n_samples)
    velocities = vae.get_velocity(n_samples=n_samples, return_mean=False)

    t = latent_time
    scaling = 20 / t.max(0)
    scaling = scaling.to_numpy()

    adata.layers["velocity"] = np.mean(velocities, axis=0) / scaling[None]
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
                              torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
                          ) * scaling

    # below needs to be appropriately updated
    adata.var["fit_u0"] = np.zeros_like(adata.var["fit_alpha"])
    adata.var["fit_s0"] = np.zeros_like(adata.var["fit_alpha"])

    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0
    if save_vae: adata.uns['vae'] = vae
    return adata


def RNA_velocity(origdata, mode: str='stochastic', color: str=None, show: bool=True, **velo_kwargs):
    """
    Prepare a scanpy AnnData object and calculate the RNA velocity, with several possible modes.
    :param origdata: the original scanpy AnnData object. In order to calculate the RNA velocity, this object needs to
                     have layers with the 'spliced' and 'unspliced' counts
    :param mode: the RNA velocity mode to use
    :param color: which observable the streamplot should be colored by
    :param show: whether or not to show the streamplot after calculating the velocities
    :param **velo_kwargs: keyword arguments to pass to the velocity calculation
    :return: the scanpy AnnData object with the calculated RNA velocity, stored in the layer 'velocity'
    """

    try:
        import scanpy as sc
        import scvelo as scv
    except ImportError:
        raise ImportError('scanpy and scvelo must be installed to use the spe.pp.RNA_velocity function.')

    adata = origdata.copy()

    sc.settings.verbosity = 0  # show errors(0), warnings(1), info(2), hints(3)
    scv.settings.verbosity = 0  # show errors(0), warnings(1), info(2), hints(3)

    mode = mode.lower()

    if mode in ['stochastic', 'dynamical', 'velovi'] and 'Ms' not in list(adata.layers.keys()):
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        scv.pp.moments(adata)

    if 'dynamical' in mode: scv.tl.recover_dynamics(adata)

    if mode in ['stochastic', 'dynamical']: scv.tl.velocity(adata, mode=mode, **velo_kwargs)

    elif mode in ['dynamo', 'dynamo-negbin', 'dynamo-gmm', 'dynamo-ransac', 'dynamo-ols', 'dynamo-rlm']:
        try:
            import dynamo as dyn
        except ImportError:
            raise ImportError('dynamo must be installed to use the spe.pp.RNA_velocity function with mode dynamo.')
        dyn.pp.recipe_monocle(adata)
        tmpmode = mode
        if tmpmode == 'dynamo': tmpmode = 'auto'
        tmpmode = tmpmode.replace('dynamo-', '')
        model = 'stochastic' if tmpmode in ['auto', 'negbin', 'gmm', 'rlm'] else 'deterministic'
        adata = dyn.tl.dynamics(adata, model=model, est_method=tmpmode, cores=4, assumption_mRNA='kinetic',  **velo_kwargs)
        adata.layers['velocity'] = adata.layers['velocity_S'].toarray()

    elif mode == 'velovi':
        adata = _velovi_velocity(adata, **velo_kwargs)

    else: raise NotImplementedError(f'RNA velocity mode {mode} not implemented.')

    scv.tl.velocity_graph(adata)

    sc.tl.pca(adata)
    if 'X_umap' not in list(adata.obsm.keys()):
        sc.tl.umap(adata)
    if show:
        scv.tl.velocity_embedding(adata, basis='umap')
        scv.pl.velocity_embedding_stream(adata, basis='umap', color=color, legend_loc='best' if color is not None else None)

    return adata


def prepare_RNA_velocity_embedding(adata, dim: int, basis: str='pca', direct_proj: bool=False, subsample: float=None, mode: str=None, **velo_kwargs):
    """
    Prepare a scanpy AnnData object and calculate the RNA velocity, with several possible modes.
    :param adata: the scanpy AnnData object. In order to calculate the RNA velocity, this object needs to
                  have layers with the 'spliced' and 'unspliced' counts
    :param dim: the number of dimensions for the embedding
    :param basis: the embedding basis to use for the velocity embedding
    :param direct_proj: whether to use direct projection for the PCA embedding (generally not recommended)
    :param subsample: the fraction of cells to subsample for the velocity embedding
    :param mode: the RNA velocity mode to use
    :param **velo_kwargs: keyword arguments to pass to the velocity calculation
    :return: the scanpy AnnData object with the calculated RNA velocity, stored in the layer 'velocity', the embedding coordinates, and the embedding of the velocities in those coordinates
    """

    try:
        import scanpy as sc
        import scvelo as scv
    except ImportError:
        raise ImportError('scanpy and scvelo must be installed to use the spe.pp.prepare_RNA_velocity_embedding function.')

    if adata.layers.get('velocity') is None or mode is not None:
        if mode is None: mode = 'stochastic'
        adata = RNA_velocity(adata, mode=mode, show=False, **velo_kwargs)

    if subsample is not None:
        adata = adata.copy()
        sc.pp.subsample(adata, fraction=subsample)
    if basis == 'pca': sc.tl.pca(adata, n_comps=dim)
    
    scv.tl.velocity_embedding(adata, basis=basis)

    if direct_proj:  # get direct PCA projection of velocities if requested
        PCs = adata.varm['PCs']
        V = adata.layers['velocity']
        if issparse(V): V = V.toarray()
        V = np.nan_to_num(V, nan=0., posinf=0., neginf=0.)
        adata.obsm['velocity_pca'] = V@PCs
    
    return adata, adata.obsm[f'X_{basis}'], adata.obsm[f'velocity_{basis}']