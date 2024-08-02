def segscan_op op (v0, f0, i0) (v1, f1, i1) =
  (if f1 then v1 else v0 `op` v1, f0 || f1, i0 i64.+ i1)

def segreduce [n] 't (op: t -> t -> t) (ne: t)
                     (flags: [n]bool) (vals: [n]t) =
  let (segscans, _, idxs) =
    map i64.bool flags
    |> zip3 vals flags
    |> scan (segscan_op op) (ne, false, 0)
    |> unzip3
  let index i j = if flags[(j + 1) % n] then i-1 else -1
  let result = scatter (copy vals) (map2 index idxs (iota n)) segscans
  in if n != 0 then result[:idxs[n - 1]] else result
