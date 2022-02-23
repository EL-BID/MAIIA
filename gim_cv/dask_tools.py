""" dask_tools.py

    Functions that are used to support preprocessing or data transformation
    operations, usually some kind of dask array acrobatics.
"""
import gc
import numpy as np
import dask.array as da

import logging

log = logging.getLogger(__name__)


def get_full_blocks(X):
    """
    Returns the subset of the input dask array made of full blocks.

    Full blocks are assumed to be those of the max size present, and blocks
    are assumed to be aligned along axis 0.

    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0 only (typically the samples
        dimension)


    Returns
    -------
    :obj:`dask.array.Array`
        A subset of the input array where the blocks are all those with the
        max chunk size present in the original array.
    """
    # get block sizes
    size_chunks = X.chunks[0]
    full_blk_ixs = np.argwhere(np.array(size_chunks) == max(size_chunks))
    if len(full_blk_ixs) == X.numblocks[0]:
        return X
    return da.concatenate([X.blocks[ix] for ix in full_blk_ixs], axis=0)


def get_incomplete_blocks(X):
    """
    Returns the subset of the input dask array made of incomplete blocks.

    Incomplete blocks are assumed to be those smaller than the max size present,
    and blocks are assumed to be aligned along axis 0.

    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0 only (typically the samples
        dimension)

    Returns
    -------
    :obj:`dask.array.Array`
        A subset of the input array where the blocks are all those with size
        smaller than the max chunk size present in the original array.
    """
    # get block sizes
    size_chunks = X.chunks[0]
    inc_blk_ixs = np.argwhere(np.array(size_chunks) != max(size_chunks))
    if len(inc_blk_ixs) == 1:
        return X.blocks[inc_blk_ixs[0]]
    elif len(inc_blk_ixs) > 1:
        return da.concatenate([X.blocks[ix] for ix in inc_blk_ixs], axis=0)


def shuffle_blocks(X):
    """
    Shuffles the order of the blocks present in a dask array

    Chunks are assumed to be along axis 0 only (the samples dimension).

    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0 only (typically the samples
        dimension)

    Returns
    -------
    :obj:`dask.array.Array`
        The input array with blocks reordered by a random permutation
    """
    blk_inds = list(np.ndindex(X.numblocks))
    log.debug("Shuffling blocks in dask array...")
    rand_perm = np.random.permutation(len(blk_inds))
    blk_inds = [blk_inds[p] for p in rand_perm]
    _X = da.concatenate(
        [X.blocks[blix] for blix in blk_inds], axis=0
    )
    return _X


def fold_interleave(X, repeats=4):
    """
    Performes iterative pseudo-shuffling operation on a dask array.

    Chunks are assumed to be along axis 0 only (the samples dimension).

    This function folds a dask array on itself along axis 0, then
    interleaves the vertically adjacent elements (like shuffling a deck of cards
    by splitting it in two, placing one next to the other then interleaving both
    halves), repeating `repeats` times.

    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension
    repeats: int
        The number of times to perform the folding shuffle operation

    Returns
    -------
    :obj:`dask.array.Array`
        The pseudo-shuffled input array

    See Also
    --------
    `stack_interleave_flatten`
    """
    n_blks = X.numblocks[0]
    n_fold = n_blks // 2
    for _ in range(repeats):
        p1, p2 = X.blocks[:n_fold], X.blocks[n_fold:2*n_fold]
        _X = stack_interleave_flatten([p1, p2])
        # if there's an odd number of blocks, put the last one (which was excluded)
        # at the beginning of the array so it gets shuffled next iteration
        if n_blks % 2 != 0:
            X = da.concatenate([X.blocks[-1], _X])
        else:
            X = _X
    return X


def combine_incomplete_blocks(X):
    """
    Merges incomplete blocks of a dask array into complete blocks

    Incomplete blocks are assumed to be those smaller than the max size present,
    and blocks are assumed to be divided only along axis 0 (e.g. training
    examples).

    New blocks formed by combining incomplete blocks are stuck on the end of the
    output array.

    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension

    Returns
    -------
    :obj:`dask.array.Array`
        The input dask array, rechunked and reassembled with as many blocks with
        the max original size present as possible.
    """
    # get block sizes
    size_chunks = X.chunks[0]
    # identify incomplete blocks in X
    inc_blk_ixs = np.argwhere(np.array(size_chunks) != max(size_chunks))
    # if there are a few incomplete blocks, put em together and try to get full blocks
    if len(inc_blk_ixs) > 1:
        inc_blocks = da.concatenate([X.blocks[ix] for ix in inc_blk_ixs], axis=0)
        inc_blocks = inc_blocks.rechunk((max(size_chunks), -1, -1, -1))
        extra_full_blocks = [b for b in inc_blocks.blocks if b.chunksize[0] == max(size_chunks)]
    elif len(inc_blk_ixs) == 1:
        extra_full_blocks = []
        inc_blocks = X.blocks[inc_blk_ixs[0]]
    else:
        return X
    # identify full blocks in original X
    full_blk_ixs = np.argwhere(np.array(size_chunks) == max(size_chunks))
    # get the original full blocks
    full_blocks = da.concatenate([X.blocks[ix] for ix in full_blk_ixs], axis=0)
    # if we formed new full blocks by combining incomplete ones,  stick em together in a new arr
    if extra_full_blocks:
        extra_full_blocks = da.concatenate(extra_full_blocks)
        log.debug("extra full blocks formed by combining incomplete ones:", extra_full_blocks)
        full_blocks = da.concatenate([full_blocks, extra_full_blocks], axis=0)
        inc_blocks = da.concatenate([b for b in inc_blocks.blocks if b.chunksize[0] != max(size_chunks)])
    return da.concatenate([full_blocks, inc_blocks])



def shuffle_blocks_together(X, y):
    """
    Shuffles the blocks of a pair of dask arrays by the same random permutation

    Used for synchronised shuffling operations on training inputs and labels.

    Blocks are assumed to be aligned along axis 0 only, as is typical for
    arrays containing training examples.

    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0
    y: :obj:`dask.array.Array`
        A dask array with chunking along axis 0
    Returns
    -------
    X: :obj:`dask.array.Array`
        The first input dask array with its blocks shuffled
    y: :obj:`dask.array.Array`
        The second input dask array with its blocks shuffled

    """
    blk_inds = list(np.ndindex(X.numblocks))
    if len(blk_inds) == 1:
        return X, y
    log.debug("Shuffling blocks in dask array pair by same permutation...")
    rand_perm = np.random.permutation(len(blk_inds))
    blk_inds = [blk_inds[p] for p in rand_perm]
    X = da.concatenate(
        [X.blocks[blix] for blix in blk_inds], axis=0
    )
    y = da.concatenate(
        [y.blocks[blix] for blix in blk_inds], axis=0
    )
    return X, y


def shuffuhl_together(X, y, shuffle_blocks=True, repeats=4):
    """
    Shuffles two dask arrays so corresponding elements end up in the same place

    Designed to shuffle pairs of image/mask arrays to achieve a degree of
    data dispersal without invoking a full random shuffle and blowing up memory.
    Combines shuffling of blocks within the array with a repeated
    'fold interleave' operation. Accounts for complete and incomplete blocks.

    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunks along axis = 0. Typically training examples.
    y: :obj:`dask.array.Array`
        A dask array with chunks along axis = 0. Typically corresponding masks.
    shuffle_blocks: bool, optional
        Flags whether to shuffle the blocks during the fold interleave operation
        (if not, will just do once at the end).
    repeats: int, optional
        Specifies how many times to perform the `fold_interleave_together`
        operation

    Returns
    -------
    :obj:`dask.array.Array`:
        Pseudo-shuffled version of first input array
    :obj:`dask.array.Array`:
        Pseudo-shuffled version of second input array

    See Also
    --------
    `shuffle_blocks_together`
    `fold_interleave_together`
    """
    # merge any incomplete blocks in the array and stick em on the end
    X_comb, y_comb = combine_incomplete_blocks(X), combine_incomplete_blocks(y)
    # distinguish the complete and incomplete blocks
    X_incomplete_blocks, y_incomplete_blocks = (get_incomplete_blocks(X_comb),
                                                get_incomplete_blocks(y_comb))
    # fold the part of the arrays with complete blocks on itself repeats times,
    # interleaving the now vertically-stacked elements and flattening into two new arrays
    # shuffle the blocks each repetition if flagged
    X_full_blocks, y_full_blocks = get_full_blocks(X_comb), get_full_blocks(y_comb)
    X_full_shf, y_full_shf = fold_interleave_together(X_full_blocks,
                                                      y_full_blocks,
                                                      shuffle_blocks=shuffle_blocks,
                                                      repeats=repeats)
    X_to_conc, y_to_conc = [X_full_shf], [y_full_shf]
    if X_incomplete_blocks is not None:
        X_to_conc.append(X_incomplete_blocks)
    if y_incomplete_blocks is not None:
        y_to_conc.append(y_incomplete_blocks)
    # tag on the incomplete blocks and shuffle them all one more time
    return shuffle_blocks_together(da.concatenate(X_to_conc, axis=0),
                                   da.concatenate(y_to_conc, axis=0))


def fold_interleave_together(X, y, shuffle_blocks=True, repeats=4):
    """
    Performes synchronised pseudo-shuffling operation on a pair of dask arrays.

    Chunks are assumed to be along axis 0 only (the samples dimension). All
    chunks are assumed to be the same size!

    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension. Typically training inputs/images.
    y : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension. Typically training labels/masks.
    shuffle_blocks: bool
        Flag controlling whether block orders are additionally shuffled each
        iteration.
    repeats: int
        The number of times to perform the folding shuffle operation

    Returns
    -------
    :obj:`dask.array.Array`
        The pseudo-shuffled first input array
    :obj:`dask.array.Array`
        The pseudo-shuffled second input array
    """
    n_blks = X.numblocks[0]
    if n_blks == 1:
        return X, y
    n_fold = n_blks // 2
    for ii in range(repeats):
        if shuffle_blocks:
            X, y = shuffle_blocks_together(X, y)
        Xp1, Xp2 = X.blocks[:n_fold], X.blocks[n_fold:2*n_fold]
        yp1, yp2 = y.blocks[:n_fold], y.blocks[n_fold:2*n_fold]
        _X = stack_interleave_flatten([Xp1, Xp2])
        _y = stack_interleave_flatten([yp1, yp2])
        assert _X.chunksize[:-1] == _y.chunksize[:-1]
        # if there's an odd number of blocks, put the last one (which was excluded)
        # at the beginning of the array so it gets shuffled next iteration
        if n_blks % 2 != 0:
            X = da.concatenate([X.blocks[-1], _X]).rechunk(X.chunks)
            y = da.concatenate([y.blocks[-1], _y]).rechunk(y.chunks)
        else:
            X = _X.rechunk(X.chunks)
            y = _y.rechunk(y.chunks)
    return X, y


def stack_interleave_flatten(arrs):
    """
    Performes folding and interleaving operation on a dask array.

    Chunks are assumed to be along axis 0 only (the samples dimension).

    This function folds a dask array on itself along axis 0, then
    interleaves the vertically adjacent elements (like shuffling a deck of cards
    by splitting it in two, placing one next to the other then interleaving both
    halves).

    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension

    Returns
    -------
    :obj:`dask.array.Array`
        The input array, with its elements now alternating between those of the
        first and second halves of the array

    See Also
    --------
    `fold_interleave`
    `fold_interleave_together`
    """
    assert all(len(a.shape) == len(arrs[0].shape) for a in arrs), (
        "arrays must have the same number of dimensions"
    )
    static_ax_ixs = range(2, len(arrs[0].shape) + 1)
    stk = da.stack(arrs, axis=0).transpose(1,0,*static_ax_ixs)
    rshp = (stk.shape[i] for i in static_ax_ixs)
    stk = stk.reshape((stk.shape[0]*stk.shape[1], *rshp))
    return stk


def chunk_generator(X):
    """
    Generator function for yielding blocks from a dask array.

    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array

    Yields
    ------
    :obj:`numpy.ndarray`
        Numpy arrays corresponding to each block of the dask array in order of
        the block indices
    """
    blk_inds = list(np.ndindex(X.numblocks))
    for blk_ind in blk_inds:
        yield X.blocks[blk_ind].compute()


def pair_chunk_generator(X, y, shuffle_blocks=False):
    """
    Generator function to yield corresponding blocks of a pair of dask arrays

    Useful for machine learning model training data generators.

    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array
    y : :obj:`dask.array.Array`
        A dask array with the same number of blocks as X whose elements
        correspond

    Yields
    ------
    tuple of :obj:`np.ndarray` :
        A pair of numpy arrays corresponding to the blocks of the inputs X and y
    """
    assert X.numblocks == y.numblocks, ("X and y must have the same numblocks!")
    # get the index pairs along the target axis which make up each chunk
    #cumulative_chunk_lens = np.cumsum(np.array(np.concatenate([[0], X.chunks[0]]))) # axis=0
    #inds = [range(*z) for z in zip(cumulative_chunk_lens, cumulative_chunk_lens[1:])]
    blk_inds = list(np.ndindex(X.numblocks))
    if shuffle_blocks:
        rand_perm = np.random.permutation(len(blk_inds))
        blk_inds = [blk_inds[p] for p in rand_perm]
    # we iterate over the block number and indices (shared for images/masks all darrs)
    for i, blk_ind in enumerate(blk_inds):
        #blk_inds = inds[i]
        yield X.blocks[blk_ind].compute(), y.blocks[blk_ind].compute()
