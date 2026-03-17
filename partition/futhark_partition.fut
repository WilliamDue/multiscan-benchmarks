-- ==
-- entry: main
-- input @ ../data/randomints_full_500MiB.in
-- output @ randomints_full_500MiB.out
-- input @ ../data/randomints_dense_500MiB.in
-- output @ randomints_dense_500MiB.out
-- input @ ../data/randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
-- input @ ../data/randomints_sparse_500MiB.in
-- output @ randomints_sparse_500MiB.out
-- input @ ../data/randomints_empty_500MiB.in
-- output @ randomints_empty_500MiB.out
entry main [n] (as: [n]i32): [n]i32 =
  partition (0i32<) as
  |> uncurry (++)
  |> sized n