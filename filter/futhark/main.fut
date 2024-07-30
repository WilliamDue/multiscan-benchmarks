-- | Filter implementation which is almost the same as in SOACS.
-- The difference is since we need it to have a "streamable quality"
-- for map-scan-map-scatter then we need to first scatter and then cutoff.
def filter [n] 'a (p: a -> bool) (as: [n]a): *[]a =
  let flags = map (\x -> if p x then 1 else 0) as
  let offsets = scan (+) 0 flags
  let result =
    scatter (copy as)
            (map2 (\f o -> if f==1 then o-1 else -1) flags offsets)
            as
  let m = if n == 0 then 0 else offsets[n-1]
  in result[:m]
