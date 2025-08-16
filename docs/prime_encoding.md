# Prime Encodings for Codons

Prime-product and base-count encodings provide complementary lenses on the same genetic composition.

## Prime-Product Encoding
- **Mechanism:** multiply a distinct prime for each nucleotide. The product is invariant to permutation, much like a conserved mass.
- **Advantages:** compact scalar, easy hashing for group comparisons.
- **Limitations:** grows quickly with longer sequences and obscures individual base contributions.

## Base-Count Encoding
- **Mechanism:** record the vector `(n_A, n_C, n_G, n_T)`.
- **Advantages:** linear structure amenable to algebra and the Preservation Constraint Equation; explicit base contributions; simple to extend with order annotations when needed.
- **Limitations:** requires more storage and still discards order without an accompanying index.

Both encodings collapse permutations of a codon into the same group. When order matters, retain the original string alongside either encoding.
