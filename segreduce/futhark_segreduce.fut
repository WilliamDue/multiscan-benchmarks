import "lib/github.com/diku-dk/segmented/segmented"

def segscan_op op (v0, f0, i0) (v1, f1, i1) =
  (if f1 then v1 else v0 `op` v1, f0 || f1, i0 i64.+ i1)

def segreduce' [n] 't (op: t -> t -> t) (ne: t)
                     (flags: [n]bool) (vals: [n]t) =
  let (segscans, _, offsets) =
    flags
    |> map i64.bool
    |> zip3 vals flags
    |> scan (segscan_op op) (ne, false, 0)
    |> unzip3
  let index f i = if f then i-1 else -1
  let is = map2 index (rotate 1 flags) offsets
  let result = scatter (#[scratch] [vals][0]) is segscans
  let count =
    scatter (#[scratch] [0])
            (map (\j -> if j == n - 1 then 0 else -1) (iota n))
            offsets
  in result[:count[0]]

entry flags [n] (as : [n]i32) =
  map2 (\i a -> i == 0 || 0i32 < a) (iota n) as

-- ==
-- input @ randomints_full_500MiB.in
-- output @ randomints_full_500MiB.out
-- input @ randomints_dense_500MiB.in
-- output @ randomints_dense_500MiB.out
-- input @ randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
-- input @ randomints_sparse_500MiB.in
-- output @ randomints_sparse_500MiB.out
-- input @ randomints_empty_500MiB.in
-- output @ randomints_empty_500MiB.out
entry main [n] (as: [n]i32) (flags: [n]bool) =
  segreduce' (+) 0 flags as

-- ==
-- entry: expected
-- input @ randomints_full_500MiB.in
-- output @ randomints_full_500MiB.out
-- input @ randomints_dense_500MiB.in
-- output @ randomints_dense_500MiB.out
-- input @ randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
-- input @ randomints_sparse_500MiB.in
-- output @ randomints_sparse_500MiB.out
-- input @ randomints_empty_500MiB.in
-- output @ randomints_empty_500MiB.out
entry expected [n] (as: [n]i32) (flags: [n]bool) =
  segmented_reduce (+) 0 flags as