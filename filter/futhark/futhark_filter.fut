-- | Filter implementation which is almost the same as in SOACS.
-- The difference is since we need it to have a "streamable quality"
-- for map-scan-map-scatter then we need to first scatter and then cutoff.
def filter' [n] 'a (p: a -> bool) (as: [n]a): *[]a =
  let flags = map (\x -> if p x then 1 else 0) as
  let offsets = scan (+) 0 flags
  let result =
    scatter (copy as)
            (map2 (\f o -> if f==1 then o-1 else -1) flags offsets)
            as
  let m = if n == 0 then 0 else offsets[n-1]
  in result[:m]

-- ==
-- input @ ../../data/randomints_sparse_500MiB.in
-- output @ ../randomints_sparse_500MiB.out
-- input @ ../../data/randomints_dense_500MiB.in
-- output @ ../randomints_dense_500MiB.out
-- input @ ../../data/randomints_moderate_500MiB.in
-- output @ ../randomints_moderate_500MiB.out
-- input @ ../../data/randomints_empty_500MiB.in
-- output @ ../randomints_empty_500MiB.out
-- input @ ../../data/randomints_full_500MiB.in
-- output @ ../randomints_full_500MiB.out
entry main [n] (as: [n]i32): *[]i32 =
  filter' (0i32<) as

-- ==
-- entry: intscan
-- input @ ../../data/randomints_sparse_500MiB.in
-- input @ ../../data/randomints_dense_500MiB.in
-- input @ ../../data/randomints_moderate_500MiB.in
-- input @ ../../data/randomints_empty_500MiB.in
-- input @ ../../data/randomints_full_500MiB.in
entry intscan [n] (as: [n]i32): *[]i32 =
  scan (+) 0 as

-- ==
-- entry: expected
-- input @ ../../data/randomints_sparse_500MiB.in
-- output @ ../randomints_sparse_500MiB.out
-- input @ ../../data/randomints_dense_500MiB.in
-- output @ ../randomints_dense_500MiB.out
-- input @ ../../data/randomints_moderate_500MiB.in
-- output @ ../randomints_moderate_500MiB.out
-- input @ ../../data/randomints_empty_500MiB.in
-- output @ ../randomints_empty_500MiB.out
-- input @ ../../data/randomints_full_500MiB.in
-- output @ ../randomints_full_500MiB.out
entry expected [n] (as: [n]i32): *[]i32 =
  filter (0i32<) as