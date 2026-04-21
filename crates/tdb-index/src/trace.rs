//! Episodic causal graph (Trace).
//!
//! Directed graph where nodes are CellIds and edges represent causal relationships.
//! Stored as adjacency lists. Supports transitive ancestor queries for causal reasoning.

use std::collections::{HashMap, HashSet, VecDeque};

use tdb_core::CellId;

/// Types of causal relationships between memory cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EdgeType {
    CausedBy = 0,
    Follows = 1,
    Contradicts = 2,
    Supports = 3,
}

impl EdgeType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::CausedBy),
            1 => Some(Self::Follows),
            2 => Some(Self::Contradicts),
            3 => Some(Self::Supports),
            _ => None,
        }
    }
}

/// A single directed edge in the Trace graph.
#[derive(Debug, Clone)]
pub struct TraceEdge {
    pub src: CellId,
    pub dst: CellId,
    pub edge_type: EdgeType,
    pub timestamp: u64,
}

/// Directed causal graph over memory cells.
///
/// Provides adjacency-list-based storage with outgoing/incoming edge queries
/// and transitive ancestor traversal for causal chain reasoning.
pub struct TraceGraph {
    /// Outgoing edges: src → [(dst, edge_type, timestamp)]
    outgoing: HashMap<CellId, Vec<TraceEdge>>,
    /// Incoming edges: dst → [(src, edge_type, timestamp)]
    incoming: HashMap<CellId, Vec<TraceEdge>>,
    /// Total number of edges.
    count: usize,
}

impl TraceGraph {
    pub fn new() -> Self {
        Self {
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            count: 0,
        }
    }

    /// Add a directed edge from `src` to `dst`.
    pub fn add_edge(&mut self, src: CellId, dst: CellId, edge_type: EdgeType, timestamp: u64) {
        let edge = TraceEdge {
            src,
            dst,
            edge_type,
            timestamp,
        };

        self.outgoing.entry(src).or_default().push(edge.clone());
        self.incoming.entry(dst).or_default().push(edge);
        self.count += 1;
    }

    /// Get outgoing edges from a node, optionally filtered by edge type.
    pub fn outgoing(&self, src: CellId, edge_type_filter: Option<EdgeType>) -> Vec<&TraceEdge> {
        self.outgoing
            .get(&src)
            .map(|edges| {
                edges
                    .iter()
                    .filter(|e| edge_type_filter.is_none_or(|t| e.edge_type == t))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get incoming edges to a node, optionally filtered by edge type.
    pub fn incoming(&self, dst: CellId, edge_type_filter: Option<EdgeType>) -> Vec<&TraceEdge> {
        self.incoming
            .get(&dst)
            .map(|edges| {
                edges
                    .iter()
                    .filter(|e| edge_type_filter.is_none_or(|t| e.edge_type == t))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all transitive ancestors of a node following edges of the given type.
    ///
    /// For `CausedBy` edges: A → B → C means "C was caused by B, B was caused by A",
    /// so ancestors of C following `CausedBy` are {A, B}.
    ///
    /// Uses BFS on the `outgoing` edges from the target node
    /// (since `CausedBy` edges point from child → parent, outgoing from C finds B, etc.)
    pub fn ancestors(&self, node: CellId, edge_type: EdgeType) -> Vec<CellId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Seed with direct outgoing edges of the given type.
        for edge in self.outgoing(node, Some(edge_type)) {
            if visited.insert(edge.dst) {
                queue.push_back(edge.dst);
            }
        }

        // BFS transitively.
        while let Some(current) = queue.pop_front() {
            for edge in self.outgoing(current, Some(edge_type)) {
                if visited.insert(edge.dst) {
                    queue.push_back(edge.dst);
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.count
    }

    /// Number of distinct nodes (that have at least one edge).
    pub fn node_count(&self) -> usize {
        let mut nodes = HashSet::new();
        for &src in self.outgoing.keys() {
            nodes.insert(src);
        }
        for &dst in self.incoming.keys() {
            nodes.insert(dst);
        }
        nodes.len()
    }
}

impl Default for TraceGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_query_edges() {
        let mut g = TraceGraph::new();
        g.add_edge(0, 1, EdgeType::CausedBy, 100);
        g.add_edge(0, 2, EdgeType::Follows, 200);

        let out = g.outgoing(0, None);
        assert_eq!(out.len(), 2);

        let out_caused = g.outgoing(0, Some(EdgeType::CausedBy));
        assert_eq!(out_caused.len(), 1);
        assert_eq!(out_caused[0].dst, 1);

        let inc = g.incoming(1, None);
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].src, 0);
    }

    #[test]
    fn test_ancestors_transitive() {
        let mut g = TraceGraph::new();
        // Chain: D(3) → C(2) → B(1) → A(0)
        g.add_edge(3, 2, EdgeType::CausedBy, 300);
        g.add_edge(2, 1, EdgeType::CausedBy, 200);
        g.add_edge(1, 0, EdgeType::CausedBy, 100);

        let ancestors = g.ancestors(3, EdgeType::CausedBy);
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&0));
        assert!(ancestors.contains(&1));
        assert!(ancestors.contains(&2));
    }
}
