-- | An Alpacc LISP lexer which was modified such that it does not
-- filter out certain tokens.

import "lib/github.com/diku-dk/containers/opt"

module type lexer_context = {
  module endomorphism_module: integral
  module terminal_module: integral
  val identity_endomorphism: endomorphism_module.t
  val endomorphism_size: i64
  val endo_mask: endomorphism_module.t
  val terminal_mask: endomorphism_module.t
  val accept_mask: endomorphism_module.t
  val produce_mask: endomorphism_module.t
  val endo_offset: endomorphism_module.t
  val terminal_offset: endomorphism_module.t
  val accept_offset: endomorphism_module.t
  val produce_offset: endomorphism_module.t
  val transitions_to_endomorphisms: [256]endomorphism_module.t
  val compositions: [endomorphism_size * endomorphism_size]endomorphism_module.t
  val dead_terminal: terminal_module.t
}

module mk_lexer(L: lexer_context) = {
  type endomorphism = L.endomorphism_module.t
  type terminal = L.terminal_module.t

  def get_value (mask: endomorphism)
                (offset: endomorphism)
                (a: endomorphism):
                endomorphism =
    let a' = mask L.endomorphism_module.& a
    in a' L.endomorphism_module.>> offset
                                    
  def is_accept (a: endomorphism): bool =
    get_value L.accept_mask L.accept_offset a
    |> L.endomorphism_module.to_i64
    |> bool.i64

  def is_produce (a: endomorphism): bool =
    get_value L.produce_mask L.produce_offset a
    |> L.endomorphism_module.to_i64
    |> bool.i64

  def to_terminal (a: endomorphism): terminal =
    get_value L.terminal_mask L.terminal_offset a
    |> L.endomorphism_module.to_i64
    |> L.terminal_module.i64

  def to_index (a: endomorphism): i64 =
    get_value L.endo_mask L.endo_offset a
    |> L.endomorphism_module.to_i64
    
  def compose (a: endomorphism) (b: endomorphism): endomorphism =
    #[unsafe]
    let a' = to_index a
    let b' = to_index b
    in copy L.compositions[b' * L.endomorphism_size + a']
    
  def trans_to_endo (c: u8): endomorphism =
    copy L.transitions_to_endomorphisms[u8.to_i64 c]

  def traverse [n] (str: [n]u8): *[n]endomorphism =
    map trans_to_endo str
    |> scan compose L.identity_endomorphism
    
  def lex_with_dead [n'] (offset: i32)
                         (str: [n']u8):
                         [](u32, terminal) =
    let n = i32.i64 n'
    let endos = traverse str
    let is = filter (\i -> i == 0 || is_produce endos[i]) (0i32..<n)
    let new_size = length is
    in tabulate new_size (
                  \i ->
                    let end = if i == new_size - 1 then n else is[i + 1]
                    let endo = endos[end - 1]
                    in (u32.i32 (offset + end - 1), to_terminal endo)
                )
    
  def lex [n'] (str: [n']u8): opt ([](u32, terminal)) =
    let result = lex_with_dead 0 str
    let is_valid =
      length result == 0 ||
      (last result).1 L.terminal_module.!= L.dead_terminal
    in if is_valid
       then some result
       else #none
}

-- End of lexer.fut

module lexer = mk_lexer {
  module terminal_module = u8
  module endomorphism_module = u16

  type endomorphism = endomorphism_module.t
  type terminal = terminal_module.t
  
  def identity_endomorphism: endomorphism = 74
  def dead_terminal: terminal = 4
  def endo_mask: endomorphism = 15
  def endo_offset: endomorphism = 0
  def terminal_mask: endomorphism = 112
  def terminal_offset: endomorphism = 4
  def accept_mask: endomorphism = 128
  def accept_offset: endomorphism = 7
  def produce_mask: endomorphism = 256
  def produce_offset: endomorphism = 8

  def endomorphism_size: i64 = 12

  def transitions_to_endomorphisms : [256]endomorphism = sized 256 [75,
75,
75,
75,
75,
75,
75,
75,
75,
128,
128,
75,
75,
128,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
128,
75,
75,
75,
75,
75,
75,
75,
161,
178,
75,
75,
75,
75,
75,
75,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
75,
75,
75,
75,
75,
75,
75,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
75,
75,
75,
75,
75,
75,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
147,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75,
75]

  def compositions : [endomorphism_size * endomorphism_size]endomorphism = [132u16, 392u16, 392u16, 392u16, 132u16, 392u16, 392u16, 392u16, 132u16, 392u16, 128u16, 75u16,
421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 161u16, 75u16,
438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 178u16, 75u16,
407u16, 407u16, 407u16, 153u16, 407u16, 407u16, 407u16, 153u16, 407u16, 153u16, 147u16, 75u16,
132u16, 132u16, 132u16, 132u16, 132u16, 132u16, 132u16, 132u16, 132u16, 132u16, 132u16, 75u16,
421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 421u16, 75u16,
438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 438u16, 75u16,
407u16, 407u16, 407u16, 407u16, 407u16, 407u16, 407u16, 407u16, 407u16, 407u16, 407u16, 75u16,
392u16, 392u16, 392u16, 392u16, 392u16, 392u16, 392u16, 392u16, 392u16, 392u16, 392u16, 75u16,
153u16, 153u16, 153u16, 153u16, 153u16, 153u16, 153u16, 153u16, 153u16, 153u16, 153u16, 75u16,
128u16, 161u16, 178u16, 147u16, 132u16, 421u16, 438u16, 407u16, 392u16, 153u16, 74u16, 75u16,
75u16, 75u16, 75u16, 75u16, 75u16, 75u16, 75u16, 75u16, 75u16, 75u16, 75u16, 75u16] :> [endomorphism_size * endomorphism_size]endomorphism
}

-- ==
-- input @ ../../data/tokens_dense_500MiB.in
-- input @ ../../data/tokens_moderate_500MiB.in
-- input @ ../../data/tokens_sparse_500MiB.in
entry main (s : []u8) : ([]u32, []u8) =
  match lexer.lex s
  case #some r -> unzip r
  case #none -> ([], [])

entry test (s : []u8) : i64 =
  match lexer.lex s
  case #some r -> length r
  case #none -> 0