import "lib/github.com/diku-dk/segmented/segmented"

def segscan_op op (v0, f0, i0) (v1, f1, i1) =
  (if f1 then v1 else v0 `op` v1, f0 || f1, i0 i64.+ i1)

def segreduce' [n] 't (op: t -> t -> t) (ne: t)
                     (flags: [n]bool) (vals: [n]t) =
  let (segscans, _, idxs) =
    rotate 1 flags
    |> map i64.bool 
    |> zip3 vals flags
    |> scan (segscan_op op) (ne, false, 0)
    |> unzip3
  let index i j = if flags[(j + 1) % n] then i-1 else -1
  let result = scatter (copy vals) (map2 index idxs (iota n)) segscans
  in if n != 0 then result[:idxs[n - 1]] else result

entry flags [n] (as : [n]i32) =
  map (0i32<) as

-- ==
-- input @ ../randomints_sparse_500MiB.in
-- output @ ../randomints_sparse_500MiB.out
-- input @ ../randomints_dense_500MiB.in
-- output @ ../randomints_dense_500MiB.out
-- input @ ../randomints_moderate_500MiB.in
-- output @ ../randomints_moderate_500MiB.out
-- input @ ../randomints_empty_500MiB.in
-- output @ ../randomints_empty_500MiB.out
-- input @ ../randomints_full_500MiB.in
-- output @ ../randomints_full_500MiB.out
entry main [n] (as: [n]i32) (flags: [n]bool) =
  segreduce' (+) 0 flags as

-- ==
-- entry: intscan
-- input @ ../randomints_sparse_500MiB.in
-- input @ ../randomints_dense_500MiB.in
-- input @ ../randomints_moderate_500MiB.in
-- input @ ../randomints_empty_500MiB.in
-- input @ ../randomints_full_500MiB.in
entry intscan [n] (as: [n]i32) (_: [n]bool): [n]i32 =
  scan (+) 0 as

-- ==
-- entry: expected
-- input @ ../randomints_sparse_500MiB.in
-- output @ ../randomints_sparse_500MiB.out
-- input @ ../randomints_dense_500MiB.in
-- output @ ../randomints_dense_500MiB.out
-- input @ ../randomints_moderate_500MiB.in
-- output @ ../randomints_moderate_500MiB.out
-- input @ ../randomints_empty_500MiB.in
-- output @ ../randomints_empty_500MiB.out
-- input @ ../randomints_full_500MiB.in
-- output @ ../randomints_full_500MiB.out
entry expected [n] (as: [n]i32) (flags: [n]bool) =
  segmented_reduce (+) 0 flags as