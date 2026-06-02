"""A simple Formula 1 race simulation built with Mesa.

The goal of this script is to demonstrate a multi-agent system rather than
realistic Formula 1 physics. It includes:

- DriverAgent: chooses pace and accumulates race time
- TeamAgent: decides pit stops based on tire wear and race conditions
- WeatherAgent: changes weather over the course of the race
- TrackAgent: translates weather into track grip
- RaceControlAgent: randomly triggers and manages Safety Car periods

The simulation runs for a fixed number of laps, prints lap-by-lap updates,
and generates several Matplotlib visuals:

- Lap number vs. lap time for every driver
- Driver position over laps
- Tire wear over laps
- Weather and Safety Car timeline
- Final standings table

Additional multi-agent features:
- Soft, Medium, and Hard tire strategies
- Driver personalities: aggressive, balanced, conservative
- Overtaking attempts between close drivers
- Random race-control incidents affecting individual drivers

Run this file directly:
    python f1_simulation.py
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Tuple

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")
lines = importlib.import_module("matplotlib.lines")
mesa = importlib.import_module("mesa")
Agent = mesa.Agent
Model = mesa.Model


# --- Simulation constants -------------------------------------------------

BASE_LAP_TIME = 90.0
PIT_STOP_PENALTY = 22.0
SAFETY_CAR_PENALTY = 12.0
DEFAULT_LAPS = 24
OUTPUT_DIR = Path(".")
LAP_TIME_PLOT = OUTPUT_DIR / "f1_lap_times.png"
POSITION_PLOT = OUTPUT_DIR / "f1_positions.png"
TIRE_WEAR_PLOT = OUTPUT_DIR / "f1_tire_wear.png"
EVENTS_PLOT = OUTPUT_DIR / "f1_events.png"
STANDINGS_PLOT = OUTPUT_DIR / "f1_standings.png"

WEATHER_WEIGHTS = {
    "Sunny": 0.60,
    "Cloudy": 0.25,
    "Rain": 0.15,
}

TRACK_GRIP = {
    "Sunny": 1.00,
    "Cloudy": 0.96,
    "Rain": 0.72,
}

PACE_WEAR = {
    "push": 11.0,
    "normal": 7.0,
    "save tires": 4.5,
}

PACE_TIME_BONUS = {
    "push": -1.8,
    "normal": 0.0,
    "save tires": 1.4,
}

TIRE_TYPES = {
    "Soft": {"bonus": -1.5, "wear": 1.4},
    "Medium": {"bonus": 0.0, "wear": 1.0},
    "Hard": {"bonus": 1.0, "wear": 0.7},
}

INCIDENT_PROBABILITY = 0.03
INCIDENT_PENALTY = 10.0
OVERTAKE_GAP = 1.0


# --- Helper functions -----------------------------------------------------


def format_seconds(value: float) -> str:
    """Format a lap or race time with two decimal places."""

    return f"{value:0.2f}s"


# --- Agents ---------------------------------------------------------------


class DriverAgent(Agent):
    """A race driver who chooses a pace and accumulates lap times."""

    def __init__(self, model: Model, name: str, skill: int, aggression: int, personality: str) -> None:
        super().__init__(model)
        self.name = name
        self.skill = max(0, min(100, skill))
        self.aggression = max(0, min(100, aggression))
        self.personality = personality
        self.tire_wear = 0.0
        self.current_tire = "Medium"
        self.total_time = 0.0
        self.position = 0
        self.current_pace = "normal"
        self.lap_times: List[float] = []

    def choose_pace(self, weather: str, grip: float, safety_car_active: bool) -> str:
        """Choose a simple driving pace for the current lap.

        The decision intentionally blends skill, aggression, tire wear, and
        conditions to keep the simulation readable rather than realistic.
        """

        if safety_car_active:
            return "normal"

        if self.personality == "aggressive":
            if grip > 0.93 and self.tire_wear < 80:
                return "push"
            if weather == "Rain" and self.tire_wear > 35:
                return "save tires"
            return "normal"

        if self.personality == "conservative":
            if weather == "Rain" or self.tire_wear > 55:
                return "save tires"
            if grip > 0.97 and self.tire_wear < 30:
                return "normal"
            return "normal"

        if weather == "Rain" and self.tire_wear > 42:
            return "save tires"

        performance_score = self.skill + self.aggression - (self.tire_wear * 0.45)
        if grip > 0.95 and performance_score > 120:
            return "push"
        if self.tire_wear >= 68 or performance_score < 78:
            return "save tires"
        return "normal"

    def apply_lap_result(self, lap_time: float, wear_gain: float) -> None:
        """Update the driver's total time, tire wear, and history."""

        self.total_time += lap_time
        self.tire_wear = min(100.0, self.tire_wear + wear_gain)
        self.lap_times.append(lap_time)

    def add_time_penalty(self, penalty: float) -> None:
        """Apply an incident penalty to the last lap and the race total."""

        self.total_time += penalty
        if self.lap_times:
            self.lap_times[-1] += penalty

    def pit_stop(self, new_tire: str) -> None:
        """Reset tires after a pit stop and fit a new compound."""

        self.tire_wear = 0.0
        self.current_tire = new_tire


class TeamAgent(Agent):
    """A team that monitors a single driver and decides pit stops."""

    def __init__(self, model: Model, name: str, driver: DriverAgent) -> None:
        super().__init__(model)
        self.name = name
        self.driver = driver

    def should_pit(self, weather: str, safety_car_active: bool) -> bool:
        """Return True when the team wants to pit its driver."""

        base_threshold = {
            "aggressive": 78.0,
            "balanced": 68.0,
            "conservative": 58.0,
        }.get(self.driver.personality, 68.0)

        wear_threshold = base_threshold
        if weather == "Rain":
            wear_threshold -= 12.0
        if safety_car_active:
            wear_threshold -= 10.0
        if self.driver.current_tire == "Soft":
            wear_threshold -= 8.0
        elif self.driver.current_tire == "Hard":
            wear_threshold += 6.0
        return self.driver.tire_wear >= wear_threshold

    def choose_tire(self, weather: str, safety_car_active: bool) -> str:
        """Choose a tire compound for the next stint."""

        if weather == "Rain":
            return "Hard"
        if safety_car_active:
            return "Hard" if self.driver.personality == "conservative" else "Medium"
        if self.driver.personality == "aggressive":
            return "Soft" if weather == "Sunny" else "Medium"
        if self.driver.personality == "conservative":
            return "Hard"
        return "Medium" if weather == "Cloudy" else "Soft"


class WeatherAgent(Agent):
    """Controls the current weather and occasionally changes it."""

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.condition = "Sunny"

    def step(self) -> bool:
        """Possibly change the weather and return True if it changed."""

        new_condition = self.model.random.choices(
            population=list(WEATHER_WEIGHTS.keys()),
            weights=list(WEATHER_WEIGHTS.values()),
            k=1,
        )[0]
        changed = new_condition != self.condition
        self.condition = new_condition
        return changed


class TrackAgent(Agent):
    """Converts weather into grip and a small lap-time effect."""

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.grip = TRACK_GRIP["Sunny"]

    def update_conditions(self, weather: str) -> float:
        """Update the grip value based on weather and return it."""

        self.grip = TRACK_GRIP[weather]
        return self.grip


class RaceControlAgent(Agent):
    """Randomly triggers and manages Safety Car periods."""

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.safety_car_active = False
        self.laps_remaining = 0

    def step(self) -> Tuple[bool, bool]:
        """Update the race control state.

        Returns:
            A tuple of (started_this_lap, ended_this_lap).
        """

        started = False
        ended = False

        if self.safety_car_active:
            self.laps_remaining -= 1
            if self.laps_remaining <= 0:
                self.safety_car_active = False
                ended = True
        else:
            # Avoid an immediate Safety Car at race start; allow it only after a few laps.
            if self.model.current_lap > 3 and self.model.random.random() < 0.16:
                self.safety_car_active = True
                self.laps_remaining = self.model.random.randint(1, 3)
                started = True

        return started, ended

    def maybe_trigger_incident(self, drivers: List[DriverAgent]) -> tuple[bool, str | None, DriverAgent | None]:
        """Randomly trigger a driver incident and return the event message."""

        if self.safety_car_active or self.model.current_lap <= 2:
            return False, None, None

        if self.model.random.random() < INCIDENT_PROBABILITY:
            victim = self.model.random.choice(drivers)
            victim.add_time_penalty(INCIDENT_PENALTY)
            message = f"{victim.name} spun at Turn 3 (+10s)"
            return True, message, victim

        return False, None, None


# --- Model ----------------------------------------------------------------


class F1RaceModel(Model):
    """A Mesa model that simulates a small Formula 1 race."""

    def __init__(self, num_laps: int = DEFAULT_LAPS, seed: int | None = None) -> None:
        super().__init__(rng=seed)
        self.num_laps = num_laps
        self.current_lap = 0

        # Core race state agents.
        self.weather_agent = WeatherAgent(self)
        self.track_agent = TrackAgent(self)
        self.race_control_agent = RaceControlAgent(self)

        # Driver/team line-up.
        driver_specs = [
            ("Maya Vega", 92, 68),
            ("Leo Santos", 84, 82),
            ("Ava Chen", 77, 55),
            ("Kai Novak", 68, 72),
            ("Jules Patel", 81, 48),
        ]

        self.drivers: List[DriverAgent] = []
        self.teams: List[TeamAgent] = []
        for name, skill, aggression in driver_specs:
            personality = "aggressive" if aggression >= 75 else "balanced" if aggression >= 55 else "conservative"
            driver = DriverAgent(self, name=name, skill=skill, aggression=aggression, personality=personality)
            team = TeamAgent(self, name=f"{name} Racing", driver=driver)
            self.drivers.append(driver)
            self.teams.append(team)

        self.lap_history: Dict[str, List[float]] = {driver.name: [] for driver in self.drivers}
        self.tire_history: Dict[str, List[str]] = {driver.name: [] for driver in self.drivers}
        self.position_history: Dict[str, List[int]] = {driver.name: [] for driver in self.drivers}
        self.tire_wear_history: Dict[str, List[float]] = {driver.name: [] for driver in self.drivers}
        self.pit_stop_history: Dict[str, List[bool]] = {driver.name: [] for driver in self.drivers}
        self.weather_history: List[str] = []
        self.grip_history: List[float] = []
        self.safety_car_history: List[bool] = []
        self.weather_change_laps: List[int] = []
        self.safety_car_laps: List[int] = []
        self.incident_laps: List[int] = []
        self.current_order: List[DriverAgent] = list(self.drivers)
        self.overtake_count = 0

    def step(self) -> None:
        """Run a single lap of the race."""

        self.current_lap += 1

        weather_changed = self.weather_agent.step()
        grip = self.track_agent.update_conditions(self.weather_agent.condition)
        safety_car_started, safety_car_ended = self.race_control_agent.step()

        self.weather_history.append(self.weather_agent.condition)
        self.grip_history.append(grip)
        self.safety_car_history.append(self.race_control_agent.safety_car_active)
        if weather_changed:
            self.weather_change_laps.append(self.current_lap)
        if self.race_control_agent.safety_car_active:
            self.safety_car_laps.append(self.current_lap)

        print(f"\nLap {self.current_lap}/{self.num_laps}")
        if weather_changed:
            print(f"  Weather changed to {self.weather_agent.condition} (grip {grip:.2f})")
        else:
            print(f"  Weather: {self.weather_agent.condition} (grip {grip:.2f})")

        if safety_car_started:
            print(f"  [Race Control] Safety Car deployed for {self.race_control_agent.laps_remaining} lap(s)")
        elif safety_car_ended:
            print("  [Race Control] Safety Car ended")
        elif self.race_control_agent.safety_car_active:
            print(f"  [Race Control] Safety Car remains active ({self.race_control_agent.laps_remaining} lap(s) left)")

        for team in self.teams:
            driver = team.driver
            new_tire = team.choose_tire(
                weather=self.weather_agent.condition,
                safety_car_active=self.race_control_agent.safety_car_active,
            )
            pit_stop = team.should_pit(
                weather=self.weather_agent.condition,
                safety_car_active=self.race_control_agent.safety_car_active,
            )
            lap_tire = driver.current_tire
            pace = driver.choose_pace(
                weather=self.weather_agent.condition,
                grip=grip,
                safety_car_active=self.race_control_agent.safety_car_active,
            )
            driver.current_pace = pace

            lap_time = self._calculate_lap_time(driver, grip, pace)
            tire_bonus = TIRE_TYPES[driver.current_tire]["bonus"]
            tire_wear_factor = TIRE_TYPES[driver.current_tire]["wear"]
            wear_gain = PACE_WEAR[pace] * tire_wear_factor
            wear_gain += 2.0 if self.weather_agent.condition == "Rain" else 0.8 if self.weather_agent.condition == "Cloudy" else 0.0
            wear_gain += max(0.0, (1.0 - grip) * 4.0)
            lap_time += tire_bonus

            if pit_stop:
                lap_time += PIT_STOP_PENALTY

            if self.race_control_agent.safety_car_active:
                lap_time += SAFETY_CAR_PENALTY

            # Record the lap after all penalties are applied.
            driver.apply_lap_result(lap_time, wear_gain)
            if pit_stop:
                driver.pit_stop(new_tire)
            self.lap_history[driver.name].append(lap_time)
            self.tire_history[driver.name].append(driver.current_tire)
            self.tire_wear_history[driver.name].append(driver.tire_wear)
            self.pit_stop_history[driver.name].append(pit_stop)

            pit_text = " | PIT STOP" if pit_stop else ""
            print(
                f"  {driver.name:<12} pace={pace:<10} tire={lap_tire:<6} lap={format_seconds(lap_time):>8} "
                f"total={format_seconds(driver.total_time):>8} "
                f"wear={driver.tire_wear:5.1f}{pit_text}"
            )

        incident_occurred, incident_text, victim = self.race_control_agent.maybe_trigger_incident(self.drivers)
        if incident_occurred and incident_text and victim is not None:
            self.incident_laps.append(self.current_lap)
            self.lap_history[victim.name][-1] += INCIDENT_PENALTY
            self.current_order = sorted(self.drivers, key=lambda driver: driver.total_time)
            print(f"  [Race Control] {incident_text}")

        self._attempt_overtakes()
        self._update_positions()
        for driver in self.current_order:
            self.position_history[driver.name].append(driver.position)

        leader = self.current_order[0]
        print(f"  Leader: {leader.name} | total race time {format_seconds(leader.total_time)}")

    def _calculate_lap_time(self, driver: DriverAgent, grip: float, pace: str) -> float:
        """Compute a simple lap time based on skill, wear, weather, and pace."""

        skill_bonus = (driver.skill - 50) * 0.14
        wear_penalty = driver.tire_wear * 0.18
        grip_penalty = (1.0 - grip) * 18.0
        pace_adjustment = PACE_TIME_BONUS[pace]
        variability = self.random.uniform(-0.4, 0.4)

        lap_time = BASE_LAP_TIME - skill_bonus + wear_penalty + grip_penalty + pace_adjustment + variability
        if self.weather_agent.condition == "Rain":
            lap_time += 6.0
        elif self.weather_agent.condition == "Cloudy":
            lap_time += 1.0
        return max(70.0, lap_time)

    def _attempt_overtakes(self) -> None:
        """Try to swap nearby drivers when the gap is small enough."""

        order = sorted(self.drivers, key=lambda driver: driver.total_time)
        for index in range(len(order) - 1):
            ahead = order[index]
            behind = order[index + 1]
            gap = behind.total_time - ahead.total_time
            if gap >= OVERTAKE_GAP:
                continue

            chance = 0.20
            chance += (behind.skill - ahead.skill) * 0.004
            chance += (behind.aggression - ahead.aggression) * 0.003
            if behind.current_tire == "Soft":
                chance += 0.05
            if ahead.current_tire == "Hard":
                chance += 0.03
            if behind.personality == "aggressive":
                chance += 0.05
            if behind.personality == "conservative":
                chance -= 0.04
            chance = max(0.05, min(0.85, chance))

            if self.random.random() < chance:
                order[index], order[index + 1] = order[index + 1], order[index]
                self.overtake_count += 1
                print(f"  Lap {self.current_lap}: {behind.name} overtook {ahead.name} for P{index + 1}")

        self.current_order = order

    def _update_positions(self) -> None:
        """Sort drivers by total race time and assign positions."""

        for index, driver in enumerate(self.current_order, start=1):
            driver.position = index

    def run_race(self) -> None:
        """Run the full race."""

        print("Starting F1 race simulation\n")
        print("Grid:")
        for driver in self.drivers:
            print(
                f"  - {driver.name} | skill={driver.skill:>3} | aggression={driver.aggression:>3} "
                f"| personality={driver.personality:<11} | tire={driver.current_tire}"
            )

        for _ in range(self.num_laps):
            self.step()

        self._print_final_standings()
        self._plot_results()
        self.summarize_race()

    def _print_final_standings(self) -> None:
        """Print the final race classification."""

        print("\nFinal standings:")
        print(f"{'Pos':<4} {'Driver':<12} {'Total Time':>12} {'Best Lap':>12} {'Tire Wear':>10}")
        print("-" * 54)
        standings = sorted(self.drivers, key=lambda driver: driver.total_time)
        for index, driver in enumerate(standings, start=1):
            best_lap = min(driver.lap_times)
            print(
                f"{index:<4} {driver.name:<12} {format_seconds(driver.total_time):>12} "
                f"{format_seconds(best_lap):>12} {driver.tire_wear:>9.1f}%"
            )

    def _plot_results(self) -> None:
        """Generate the lap charts and standings table with Matplotlib."""

        self._plot_lap_times()
        self._plot_positions()
        self._plot_tire_wear()
        self._plot_events_timeline()
        self._plot_standings_table()

        print(f"\nSaved lap chart to: {LAP_TIME_PLOT.resolve()}")
        print(f"Saved position chart to: {POSITION_PLOT.resolve()}")
        print(f"Saved tire wear chart to: {TIRE_WEAR_PLOT.resolve()}")
        print(f"Saved events timeline to: {EVENTS_PLOT.resolve()}")
        print(f"Saved standings table to: {STANDINGS_PLOT.resolve()}")

    def _plot_lap_times(self) -> None:
        """Plot lap time curves with pit stop and Safety Car annotations."""

        fig, ax = plt.subplots(figsize=(11, 6))
        lap_numbers = list(range(1, self.num_laps + 1))
        for driver in self.drivers:
            ax.plot(lap_numbers, self.lap_history[driver.name], marker="o", linewidth=1.7, label=driver.name)
            pit_laps = [lap for lap, did_pit in enumerate(self.pit_stop_history[driver.name], start=1) if did_pit]
            pit_values = [self.lap_history[driver.name][lap - 1] for lap in pit_laps]
            if pit_laps:
                ax.scatter(pit_laps, pit_values, color="red", marker="x", s=70, zorder=5)

        for lap in self.safety_car_laps:
            ax.axvspan(lap - 0.5, lap + 0.5, color="gray", alpha=0.08)

        ax.set_title("Formula 1 Race Simulation - Lap Times")
        ax.set_xlabel("Lap number")
        ax.set_ylabel("Lap time (seconds)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(LAP_TIME_PLOT, dpi=160)
        plt.close(fig)

    def _plot_positions(self) -> None:
        """Plot how the race position changed lap by lap."""

        fig, ax = plt.subplots(figsize=(11, 6))
        lap_numbers = list(range(1, self.num_laps + 1))
        for driver in self.drivers:
            ax.plot(lap_numbers, self.position_history[driver.name], marker="o", linewidth=1.7, label=driver.name)

        ax.invert_yaxis()
        ax.set_yticks(range(1, len(self.drivers) + 1))
        ax.set_title("Formula 1 Race Simulation - Position Over Laps")
        ax.set_xlabel("Lap number")
        ax.set_ylabel("Position (1 = leader)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(POSITION_PLOT, dpi=160)
        plt.close(fig)

    def _plot_tire_wear(self) -> None:
        """Plot tire degradation to show why pit stops happen."""

        fig, ax = plt.subplots(figsize=(11, 6))
        lap_numbers = list(range(1, self.num_laps + 1))
        for driver in self.drivers:
            ax.plot(lap_numbers, self.tire_wear_history[driver.name], marker="o", linewidth=1.7, label=driver.name)

        ax.set_title("Formula 1 Race Simulation - Tire Wear")
        ax.set_xlabel("Lap number")
        ax.set_ylabel("Tire wear")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(TIRE_WEAR_PLOT, dpi=160)
        plt.close(fig)

    def _plot_events_timeline(self) -> None:
        """Plot weather and Safety Car events on a compact timeline."""

        weather_colors = {"Sunny": "gold", "Cloudy": "lightgray", "Rain": "royalblue"}
        fig, ax = plt.subplots(figsize=(11, 2.8))
        lap_numbers = list(range(1, self.num_laps + 1))

        for lap, weather in zip(lap_numbers, self.weather_history):
            ax.bar(lap, 1, color=weather_colors[weather], edgecolor="white", width=0.9)
            if self.safety_car_history[lap - 1]:
                ax.text(lap, 0.5, "SC", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

        ax.set_xlim(0.5, self.num_laps + 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Lap number")
        ax.set_title("Formula 1 Race Simulation - Weather and Safety Car Timeline")
        legend_handles = [
            lines.Line2D([0], [0], color=color, lw=8, label=weather)
            for weather, color in weather_colors.items()
        ]
        legend_handles.append(lines.Line2D([0], [0], color="black", marker="o", linestyle="", label="Safety Car"))
        legend_handles.append(lines.Line2D([0], [0], color="crimson", marker="x", linestyle="", label="Incident"))

        for lap in self.incident_laps:
            ax.scatter(lap, 0.82, color="crimson", marker="x", s=60, zorder=5)

        ax.legend(handles=legend_handles, loc="upper center", ncol=5, fontsize=9)
        fig.tight_layout()
        fig.savefig(EVENTS_PLOT, dpi=160)
        plt.close(fig)

    def _plot_standings_table(self) -> None:
        """Plot the final standings as a clean table."""

        standings = sorted(self.drivers, key=lambda driver: driver.total_time)
        table_rows = [
            [str(index), driver.name, format_seconds(driver.total_time), f"{driver.tire_wear:.1f}%", driver.current_tire]
            for index, driver in enumerate(standings, start=1)
        ]
        fig, ax = plt.subplots(figsize=(9.5, 3.4))
        ax.axis("off")
        table = ax.table(
            cellText=table_rows,
            colLabels=["Pos", "Driver", "Total Time", "Tire Wear", "Tire"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        ax.set_title("Final Race Standings", pad=18)
        fig.tight_layout()
        fig.savefig(STANDINGS_PLOT, dpi=160)
        plt.close(fig)

    def summarize_race(self) -> None:
        """Print a short statistics summary for presentation use."""

        total_pits = sum(sum(1 for lap in history if lap) for history in self.pit_stop_history.values())
        weather_change_count = len(self.weather_change_laps)
        safety_car_lap_count = sum(1 for active in self.safety_car_history if active)
        incident_count = len(self.incident_laps)
        print("\nRace summary:")
        print(f"  Total pit stops: {total_pits}")
        print(f"  Weather changes: {weather_change_count}")
        print(f"  Laps under Safety Car: {safety_car_lap_count}")
        print(f"  Incidents: {incident_count}")
        print(f"  Overtakes: {self.overtake_count}")


# --- Script entry point ---------------------------------------------------


def main() -> None:
    """Create and run the simulation automatically."""

    model = F1RaceModel(num_laps=DEFAULT_LAPS)
    model.run_race()


if __name__ == "__main__":
    main()

