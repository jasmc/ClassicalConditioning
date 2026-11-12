"""Shared acquisition facts and diagnostic policy for the three supported assays.

Acquisition event numbers start at one. No stimulus is discarded. Exact event totals are protocol requirements. Response windows remain separate
analysis settings.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SessionPhase:
    name: str
    cs_count: int
    us_count: int
    us_duration_ms: int


@dataclass(frozen=True)
class SessionProtocol:
    cs_duration_s: float = 10.0
    phases: tuple[SessionPhase, ...] = (
        SessionPhase("priming", 4, 15, 50),
        SessionPhase("pretraining", 10, 3, 50),
        SessionPhase("training", 50, 46, 100),
        SessionPhase("testing", 30, 14, 50),
    )
    training_catch_cs: tuple[int, ...] = (25, 39, 53, 59)
    analysis_catch_cs: tuple[int, ...] = (25, 39, 53, 59, 65)
    final_check_count: int = 1
    final_check_duration_ms: int = 500
    first_us_discarded: bool = False
    control_min_cs_interval_s: float = 5.0
    control_min_cs_us_onset_separation_s: float = 0.5

    @property
    def first_training_us(self) -> int:
        return 1 + sum(p.us_count for p in self.phases[:2])

    @property
    def paired_count(self) -> int:
        return self.phases[2].cs_count - len(self.training_catch_cs)

    @property
    def cs_count(self) -> int:
        return sum(p.cs_count for p in self.phases)

    @property
    def regular_us_count(self) -> int:
        return sum(p.us_count for p in self.phases)

    @property
    def complete_us_count(self) -> int:
        return self.regular_us_count + self.final_check_count


@dataclass(frozen=True)
class ProtocolVerificationPolicy:
    timing_tolerance_ms: int = 200
    pulse_duration_tolerance_ms: int = 0  # Nominal-duration differences are warnings.
    discrepancy_action: str = "warn_and_continue"
    control_assay_assignment: str = "explicit_experiment_configuration"


SESSION = SessionProtocol()
VERIFICATION_POLICY = ProtocolVerificationPolicy()


def protocol_settings() -> dict:
    return {
        "session": asdict(SESSION),
        "verification": asdict(VERIFICATION_POLICY),
        "expected_cs_count": SESSION.cs_count,
        "regular_us_count": SESSION.regular_us_count,
        "complete_us_count": SESSION.complete_us_count,
        "expected_us_count": SESSION.complete_us_count,
        "event_count_rule": "exact_equality",
        "training_us_ids": list(range(SESSION.first_training_us,
                                      SESSION.first_training_us + SESSION.paired_count)),
        "control_rule": "US schedule retained; training CS onsets randomized per fish",
        "source": "Paper Methods: general structure and control protocols; author confirmed US 19 starts training on 2026-10-31",
    }
