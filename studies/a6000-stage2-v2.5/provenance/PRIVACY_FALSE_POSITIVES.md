# Reviewed synthetic identifiers

UUID-shaped constants in `code/tests/v2/test_stage2_heldout.py` are deliberate
synthetic identities: repeated digit/letter patterns at lines 39 and 477.
They test pair mismatch and alias handling; they are not the server's UUIDs.
The `GPU-00000000-1111-2222-3333-444444444444` literal in existing redaction
tests/export code is also synthetic. These frozen test fixtures are retained
unchanged, not confused with privacy leaks.

The `A6000@PCI` grammar in mock native tests is a device/name delimiter, not
an email. Real device UUIDs, personal home paths, hostnames and private
network addresses in the new scientific records are transformed through
the declared derivative rules.

Installed-package names in historical environment inventories are facts
about that environment. Material allocation/delegation evidence is not
rewritten to invent exclusive or unaided authorship. This privacy review is
not a claim that every historical tool-name string has been removed.
