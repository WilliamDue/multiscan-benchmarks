# Multiscan Benchmarks
This repository will explore performance differences in Futhark  and CUDA in respect to Multiscan.

## Running The Benchmarks

To run the benchmarks you need to have Futhark, CUDA and CUB.
All benchmarks can be ran from the root folder using `make`.
To run individual benchmarks navigate into a folder and run `make`.
To clean up files created use `make clean` in the same manner as `make`.

## Multiscan
The Single-Pass scan [^1] allows for the fusion of multiple  map-scans ending in a single map-reduce or map-scatter. This construct will be refered to as a Multiscan.

For the ability to fuse two or more map-scans then it should hold for any neighbouring pair of map-scans in a sequence of map-scans that:
* The first map-scans output is used as input in the second map-scan.
* The operations used in the second map-scan does no indexing into the first map-scans output array. 

For the ability to fuse a map-reduce or map-scatter at the end then:
* The output of the multiple map-scans should be used as input. For map-scatter this means the elements that will be copied into some other array.
* The arrays used in map-scatter must not dependent on knowing the full output in any of the map-scans.
* The operations used in the map-reduce must not index into the results of the map-scans.

### Notes
* This is mostlikely not a compelete list of rules.
* "The operation used in the second map-scan does no indexing into the first map-scans output array." It should be possible to index into a past array if the index is less or equal to the element at the current index. This seems confusing in relation to writing futhark code.

## Citations
[^1]: Harris, Mark, and Michael Garland. "Single-Pass Parallel Prefix Scan with Decoupled Look-Back." Proceedings of the 2016 ACM/IEEE Conference on Supercomputing (SC), 2016. [https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
