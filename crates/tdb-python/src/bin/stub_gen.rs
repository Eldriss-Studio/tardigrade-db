//! Stub generator binary — emits `.pyi` files from the `PyO3` surface.
//!
//! Invoked from the Maturin build pipeline. Walks every item annotated
//! with `#[gen_stub_pyclass]` / `#[gen_stub_pymethods]` /
//! `#[gen_stub_pyfunction]` in the library crate and writes Python type
//! stubs into `python/tardigrade_db/`.
//!
//! Run manually: `cargo run -p tdb-python --bin stub_gen --release`
//! Or via maturin: `maturin develop` (configured to invoke this binary).

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = _native::stub_info()?;
    stub.generate()?;
    Ok(())
}
