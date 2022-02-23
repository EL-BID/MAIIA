import sys
import numpy as np
import pytest
import dask.array as da

import gim_cv.dask_tools as dask_tools


@pytest.fixture(scope='module')
def X0():
    """A sample dask array with shape (samples, rows, columns, channels)"""
    size_0 = 100
    chunk_size_0 = 13
    X0 = da.random.random((size_0, 24, 24, 3),
                          chunks=(chunk_size_0, 24, 24, 3))
    return X0


@pytest.fixture(scope='module')
def X1():
    """A sample dask array with shape (samples, rows, columns, channels)"""
    size_1 = 23
    chunk_size_1 = 6
    X1 = da.random.random((size_1, 24, 24, 3),
                          chunks=(chunk_size_1, 24, 24, 3))
    return X1


@pytest.fixture(scope='module')
def X_conc(X0, X1):
    """A sample dask array with shape (samples, rows, columns, channels)
       formed by concatenating two such arrays with different chunk sizes
    """
    return da.concatenate([X0, X1], axis=0)


def test_get_full_blocks(X0, X1, X_conc):
    """ check subset of dask array returned which contains only the full blocks
    """
    #size_0, size_1 = 100, 23
    #chunk_size_0, chunk_size_1 = 13, 6
    #X0 = da.random.random((size_0, 24, 24, 3),
    #                      chunks=(chunk_size_0, 24, 24, 3))
    #X1 = da.random.random((size_1, 24, 24, 3),
    #                      chunks=(chunk_size_1, 24, 24, 3))
    #X_conc = da.concatenate([X0, X1], axis=0)
    X0_full = dask_tools.get_full_blocks(X0)
    X1_full = dask_tools.get_full_blocks(X1)
    X_full = dask_tools.get_full_blocks(X_conc)
    # check individual arrays have one incomplete block and the correct shapes
    # X0
    exp_n_full_blocks_X0 = X0.shape[0] // X0.chunks[0][0]
    exp_full_chunks_X0 = (
        tuple(X0.chunks[0][0] for _ in range(exp_n_full_blocks_X0)),
        (24,), (24,), (3,)
    )
    assert X0_full.numblocks[0] == exp_n_full_blocks_X0
    assert X0_full.chunks == exp_full_chunks_X0
    # X1
    exp_n_full_blocks_X1 = X1.shape[0] // X1.chunks[0][0]#chunk_size_1
    exp_full_chunks_X1 = (
        tuple(X1.chunks[0][0] for _ in range(exp_n_full_blocks_X1)),
        (24,), (24,), (3,)
    )
    assert X1_full.numblocks[0] == exp_n_full_blocks_X1
    assert X1_full.chunks == exp_full_chunks_X1
    # concatenating both, the behaviour should be to define the full blocks to
    # match the larger of the two
    exp_n_full_blocks_X_conc = exp_n_full_blocks_X0
    exp_full_chunks_X_conc = 0
    assert X_full.numblocks[0] == exp_n_full_blocks_X_conc


def test_get_incomplete_blocks(X0, X1, X_conc):
    """ check subset of dask array returned which contains only the incomplete
        blocks
    """
    X0_incomplete = dask_tools.get_incomplete_blocks(X0)
    X1_incomplete = dask_tools.get_incomplete_blocks(X1)
    X_incomplete = dask_tools.get_incomplete_blocks(X_conc)
    # check individual arrays have one incomplete block and the correct shapes
    # X0
    exp_n_full_blocks_X0 = X0.shape[0] // X0.chunks[0][0]
    exp_n_incomplete_blocks_X0 = 1
    exp_incomplete_chunks_X0 = (
        (X0.shape[0] - exp_n_full_blocks_X0 * X0.chunks[0][0],),
        (24,), (24,), (3,)
    )
    assert X0_incomplete.numblocks[0] == 1
    assert X0_incomplete.chunks == exp_incomplete_chunks_X0
    # X1
    exp_n_full_blocks_X1 = X1.shape[0] //  X1.chunks[0][0]
    exp_n_incomplete_blocks_X1 = 1
    exp_incomplete_chunks_X1 = (
        (X1.shape[0] - exp_n_full_blocks_X1 * X1.chunks[0][0],),
        (24,), (24,), (3,)
    )
    assert X1_incomplete.numblocks[0] == 1
    assert X1_incomplete.chunks == exp_incomplete_chunks_X1
    # concatenating both, the behaviour should be to define the full blocks to
    # match the larger of the two
    # all the smaller array's blocks are incomplete
    exp_n_incomplete_blocks_X_conc = 1 + X1.numblocks[0]
    assert X_incomplete.numblocks[0] == exp_n_incomplete_blocks_X_conc


def test_shuffle_blocks(X_conc):
    Xs = dask_tools.shuffle_blocks(X_conc)
    assert not (X_conc == Xs).compute().all()
    assert X_conc.shape == Xs.shape


def test_stack_interleave_flatten():
    """The first two elements of the interleaved array are the first
       elements from each of the individual arrays
    """
    X = da.random.random((100, 2, 2,3 ), chunks=10)
    X1 = da.random.random((100, 2, 2, 3), chunks=10)
    sif  = dask_tools.stack_interleave_flatten([X, X1])
    assert (
        (sif[0:2].compute() == np.stack([X[0].compute(), X1[0].compute()])).all()
    )

def test_fold_interleave():
    """Second element of folded array should be first element of second half of
       the blocks of original array, counting along axis 0
    """
    X = da.random.random((100, 2, 2,3 ), chunks=10)
    fi = dask_tools.fold_interleave(X, repeats=1).compute()
    assert (fi[1] == X.blocks[5].compute()[0]).all()


@pytest.mark.skip(reason='NYI')
def test_combine_incomplete_blocks():
    pass


@pytest.mark.skip(reason='NYI')
def test_interleave_blocks():
    pass

@pytest.mark.skip(reason='NYI')
def test_shuffle_blocks_together():
    pass


@pytest.mark.skip(reason='NYI')
def test_shuffuhl_together():
    pass


@pytest.mark.skip(reason='NYI')
def test_fold_interleave_together():
    pass


@pytest.mark.skip(reason='NYI')
def test_separate_incomplete_blocks():
    pass


@pytest.mark.skip(reason='NYI')
def test_interleave_identical_blocks():
    pass





@pytest.mark.skip(reason='NYI')
def test_chunk_generator():
    pass


@pytest.mark.skip(reason='NYI')
def test_pair_chunk_generator():
    pass
