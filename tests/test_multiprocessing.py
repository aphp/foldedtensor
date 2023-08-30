from multiprocessing.reduction import ForkingPickler

import pytest
import torch
import torch.multiprocessing as mp

from foldedtensor import as_folded_tensor


def workerB(X, Y, done, result, device):
    print("Started B, waiting for X", device, flush=True)
    tensor = X.get()
    assert tensor.is_shared()
    assert tensor.indexer.is_shared()
    new_tensor = tensor + 1
    assert new_tensor.indexer is tensor.indexer
    print("B put back tensor in Y", flush=True)
    Y.put(new_tensor)
    del tensor
    done.get()
    del new_tensor
    result.put(True)


def workerA(X, Y, done, result, device):
    print("Started A", device, flush=True)
    t = as_folded_tensor(
        data=[[4, 3]],
        data_dims=("samples", "lines"),
        full_names=("samples", "lines"),
        dtype=torch.long,
    ).to(device)
    X.put(t)
    print("A waiting for Y", flush=True)
    Y.get()
    print("A received from Y", flush=True)
    done.put(True)
    result.put(True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_share_device(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X = mp.Queue()
    Y = mp.Queue()
    done = mp.Queue()
    result = mp.Queue()
    a = mp.Process(target=workerA, args=(X, Y, done, result, device))
    b = mp.Process(target=workerB, args=(X, Y, done, result, device))
    a.start()
    b.start()
    a.join()
    b.join()

    assert result.get()
    assert result.get()
    print("Done", flush=True)

    X.close()
    Y.close()
    done.close()
    result.close()
    X.join_thread()
    Y.join_thread()
    done.join_thread()
    result.join_thread()


def test_forking_pickler():
    ft = as_folded_tensor(
        [
            [4, 3],
        ],
        data_dims=("samples", "lines"),
        full_names=("samples", "lines"),
        dtype=torch.long,
    ).share_memory_()
    print(ForkingPickler.dumps(ft))
