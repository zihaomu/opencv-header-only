# TODOS

This file tracks intentionally deferred work with sufficient context for future pickup.

## TODO-2026-04-16-01 Distribution Hardening After Trust Pipeline Stabilizes
- Priority: P2
- What: Add distribution pipeline closure after Trust Pipeline is stable for 2 release cycles: `CMake package export -> vcpkg/conan`.
- Why: Current strategy optimizes for compatibility/performance trust first, but production adoption eventually needs low-friction install/discovery.
- Pros: Lowers trial/integration cost, improves external credibility as a consumable library.
- Cons: Adds maintenance burden across package metadata, version policy, and platform-specific packaging behavior.
- Context: Explicitly deferred in current plan; distribution is not a hard release gate during Trust Pipeline phase.
- Depends on / blocked by: Trust Pipeline hard gates and report artifacts must be stable (low-noise) across at least 2 release cycles.
- Suggested trigger to start: Once PR default gates (`core_basic` + `imgproc_quick_gate`) are enforced and false-positive rate is acceptable.

## TODO-2026-04-16-02 Trust Scoreboard After Evidence Pipeline Stabilizes
- Priority: P3
- What: Add a Trust Scoreboard that visualizes compatibility coverage, performance stability, and gate health trends.
- Why: Current evidence is strong but distributed; a scoreboard improves fast external comprehension and release-to-release trend visibility.
- Pros: Better communication for adopters, quicker confidence checks, easier regression storytelling.
- Cons: Requires scoring-model governance; premature rollout can create noisy or misleading signals.
- Context: Considered during plan alternatives as Approach C; intentionally deferred to avoid disrupting current Trust Pipeline delivery.
- Depends on / blocked by: Stable contract definitions, stable CI artifact schema, and low-noise gate outcomes over multiple cycles.
- Suggested trigger to start: After Trust Pipeline has run on at least 2 stable release cycles with consistent artifact quality.

