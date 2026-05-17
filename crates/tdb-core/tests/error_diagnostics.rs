//! ATDD for typed error context + miette diagnostics.
//!
//! Two properties under test:
//!
//! 1. `TardigradeError` implements `miette::Diagnostic`, so Rust
//!    consumers (and any future Rust CLI / server) can pretty-print
//!    errors with codes, source locations, and help text.
//! 2. New structured variants carry actionable context in their
//!    `Display` output and a `#[help]` annotation with a fix hint.
//!    Existing variants keep their old `Display` shape for
//!    backward compatibility.

use miette::Diagnostic;
use tdb_core::error::TardigradeError;

#[test]
fn tardigrade_error_implements_miette_diagnostic() {
    fn assert_diagnostic<E: Diagnostic>() {}
    assert_diagnostic::<TardigradeError>();
}

#[test]
fn empty_corpus_variant_carries_owner_and_help() {
    let err = TardigradeError::EmptyCorpus { owner: 7 };
    let display = format!("{err}");
    assert!(display.contains("owner"), "Display should mention 'owner': {display}");
    assert!(display.contains('7'), "Display should mention the owner id: {display}");
    let help = err.help().expect("EmptyCorpus should carry a help hint");
    let help_text = format!("{help}");
    assert!(!help_text.is_empty(), "help text should be non-empty");
}

#[test]
fn no_query_key_variant_carries_hint() {
    let err = TardigradeError::NoQueryKey { hint: "set capture_fn".to_string() };
    let display = format!("{err}");
    assert!(display.contains("capture_fn"), "hint should appear in Display: {display}");
    let help = err.help().expect("NoQueryKey should carry a help hint");
    assert!(!format!("{help}").is_empty());
}

#[test]
fn existing_variants_keep_their_display_shape() {
    // Backward-compat: the existing CellNotFound shape must not
    // change. Code that grep'd for "cell not found: N" still works.
    let err = TardigradeError::CellNotFound(42);
    let display = format!("{err}");
    assert!(display.starts_with("cell not found"), "shape regressed: {display}");
    assert!(display.contains("42"));
}

#[test]
fn existing_snapshot_variants_still_work() {
    let err = TardigradeError::NotATardigradeSnapshot { reason: "missing magic".to_string() };
    let display = format!("{err}");
    assert!(display.contains("missing magic"), "{display}");
}

#[test]
fn diagnostic_codes_are_unique_per_variant() {
    // Variants with a `#[diagnostic(code = "...")]` annotation
    // should expose distinct codes so consumers can match on them.
    let a = TardigradeError::EmptyCorpus { owner: 1 };
    let b = TardigradeError::NoQueryKey { hint: "x".into() };
    let a_code = a.code().map(|c| format!("{c}"));
    let b_code = b.code().map(|c| format!("{c}"));
    assert!(a_code.is_some(), "EmptyCorpus should have a diagnostic code");
    assert!(b_code.is_some(), "NoQueryKey should have a diagnostic code");
    assert_ne!(a_code, b_code, "codes must be unique");
}
