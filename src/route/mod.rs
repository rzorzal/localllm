//! Local-vs-cloud routing: a pure decision function plus its policy knobs.
//!
//! The decision logic (`decide`) performs no I/O so it is unit-testable in
//! isolation. The HTTP layer turns a `Decision` into either the local engine
//! path or a cloud reverse-proxy.

pub mod policy;

pub use policy::{Profile, RoutingPolicy};
