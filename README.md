## Notes

*TLDR*: Let the compiler auto-vectorize `comp_grad()` for a more portable code, or use `comp_grad_vectorized()` to squeeze even more theoretical performance, always maintaining a **row-major layout**. Both reduce TLB misses, and are cache friendly enough to beat tiled variants, I suspect because of the fact that the stencil has unit radius.

- Compiling `comp_grad()` with -O3 uses vectorization in the inner loop by default, but always using unaligned move instructions, and with extra logic to handle potential leftover scalars. Manually vectorizing with intrinsics inside `comp_grad_vectorized()` and working on **aligned** and possibly padded data should theoretically be better, but most of the times leads to an almost negligible advantage in execution time.

- Tiled layouts could be introduced to try to exploit **only** the temporal locality of the stencil, with a tile being able to fit entirely into the L1 cache, since vectorization reduces the need to consider the spatial locality. Unfortunately, they would probably cause headaches when having to deal with the ordering of the points required by the linear solvers, moreover `comp_grad_tiled()`, auto-vectorized with -O3, is still behind the non tiled versions, and even though it's dealing with less LL cache loads and misses, **TLB misses** seem to be the major bottleneck. This is most notably true for `comp_grad_vectorized_tiled_loop()`, where the data is layed out in row-major but the processing happens one tile at a time. I would only expect TLB misses for the tiled layout to happen due to loads across the tile boundary, while the row-major one suffers them a lot when jumping along the outer dimension. For a contiguos row-major access, TLB misses are vastly reduced, since a few pages are being entirely streamed to the cache at any time. Hardware prefetching is also more reliable for contiguous memory accesses, potentially hidìng TLB and cache misses delays.

## TODO

- Try to avoid the unaligned load to compute the finite difference along the inner dimension, consider loading contiguous vectors and transposing in-register before performing the vector sub.
