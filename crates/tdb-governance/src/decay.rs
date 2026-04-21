//! Recency decay function for the AKL.
//!
//! r = exp(-Δt / τ), where τ = 30 days (~21-day half-life).
//! Applied as a multiplier during retrieval scoring.
