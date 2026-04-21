//! Episodic causal graph (Trace).
//!
//! Directed graph where nodes are CellIds and edges represent causal relationships.
//! Edge types: CAUSED_BY, FOLLOWS, CONTRADICTS, SUPPORTS.
//! Stored as adjacency lists in a separate segment file.
