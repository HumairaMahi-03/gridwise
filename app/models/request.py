"""Request models for POST /optimize-energy.

All structural validation happens here so that malformed scenarios are rejected
with HTTP 422 before any LLM call or solver run is attempted.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import HOURS_IN_DAY


class HourRecord(BaseModel):
    """One hourly slice of the 24-hour scenario."""

    model_config = ConfigDict(extra="forbid")

    hour: int = Field(..., ge=0, le=HOURS_IN_DAY - 1, description="Hour of day, 0-23.")
    demand_kwh: float = Field(..., ge=0, description="Campus demand for the hour, kWh.")
    solar_kwh: float = Field(..., ge=0, description="Available solar generation, kWh.")
    tariff_bdt_per_kwh: float = Field(..., ge=0, description="Grid tariff, BDT per kWh.")


class BatteryConfig(BaseModel):
    """Battery capabilities and starting state."""

    model_config = ConfigDict(extra="forbid")

    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_logical_relationships(self) -> "BatteryConfig":
        if not (self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh):
            raise ValueError(
                "battery must satisfy "
                "0 <= minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh"
            )
        return self


class OptimizationRequest(BaseModel):
    """A 24-hour campus energy scenario plus free-text operator notes."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(..., min_length=1)
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourRecord] = Field(..., min_length=HOURS_IN_DAY, max_length=HOURS_IN_DAY)
    battery: BatteryConfig

    @field_validator("scenario_id")
    @classmethod
    def scenario_id_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("scenario_id must be a non-empty string")
        return cleaned

    @field_validator("operator_notes")
    @classmethod
    def notes_not_blank(cls, value: List[str]) -> List[str]:
        for index, note in enumerate(value):
            if not note or not note.strip():
                raise ValueError(f"operator_notes[{index}] must be a non-empty string")
        return [note.strip() for note in value]

    @field_validator("hours")
    @classmethod
    def hours_cover_full_day(cls, value: List[HourRecord]) -> List[HourRecord]:
        seen = [record.hour for record in value]
        if len(set(seen)) != len(seen):
            raise ValueError("hours must not contain duplicate hour values")
        if sorted(seen) != list(range(HOURS_IN_DAY)):
            raise ValueError("hours must contain exactly one record for each hour 0-23")
        return value

    def hours_in_order(self) -> List[HourRecord]:
        """Hourly records sorted ascending by hour, regardless of input order."""
        return sorted(self.hours, key=lambda record: record.hour)
