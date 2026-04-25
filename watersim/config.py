"""Dataclass-based scene configuration objects."""

from dataclasses import dataclass


@dataclass
class KarmanConfig:
    width: int = 256
    height: int = 64
    inflowVelocity: float = 1.0
    viscosity: float = 0.0002
    obstacleRadiusFrac: float = 0.10
    frames: int = 500
    fps: int = 30


@dataclass
class SwirlConfig:
    gridSize: int = 128
    frames: int = 360
    fps: int = 30
    densityAmount: float = 200.0
    forceScale: float = 5.0


@dataclass
class DamBreakConfig:
    gridWidth: int = 192
    gridHeight: int = 96
    nParticlesPerRow: int = 50
    frames: int = 500
    fps: int = 30


@dataclass
class RayleighTaylorConfig:
    width: int = 192
    height: int = 288
    frames: int = 600
    fps: int = 30
    pertAmplitude: float = 0.015
    pertWavenumber: int = 3
    gravityStrength: float = 0.12


@dataclass
class LidCavityConfig:
    size: int = 192
    lidVelocity: float = 1.0
    viscosity: float = 0.005
    frames: int = 600
    fps: int = 30


@dataclass
class StaticConfig:
    simulationSteps: int = 50
    nParticlesPerRow: int = 30
    gridSize: int = 64
    dpi: int = 300


@dataclass
class RealtimeConfig:
    gridSize: int = 128
    dt: float = 0.1
    viscosity: float = 1e-6
    forceAmount: float = 5.0
    densityAmount: float = 100.0
