"""Helpers for splitting repeated Monte Carlo work into batches."""


def split_batch_sizes(total: int, n_batches: int) -> list[int]:
    """Split ``total`` samples into ``n_batches`` nearly equal positive sizes."""
    if total <= 0:
        raise ValueError("total must be positive")
    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    if n_batches > total:
        raise ValueError("n_batches cannot exceed total")

    base, remainder = divmod(total, n_batches)
    return [base + (1 if batch_idx < remainder else 0) for batch_idx in range(n_batches)]
