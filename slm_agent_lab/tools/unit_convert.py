"""Deterministic unit conversion across length, mass, volume, time, speed, data and temperature."""

from __future__ import annotations

# canonical unit -> (category, factor to the category's base unit)
_UNITS: dict[str, tuple[str, float]] = {
    # length (base: metre)
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "mi": ("length", 1609.344),
    "yd": ("length", 0.9144),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    # mass (base: kilogram)
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 1e-6),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "t": ("mass", 1000.0),
    # volume (base: litre)
    "l": ("volume", 1.0),
    "ml": ("volume", 0.001),
    "gal": ("volume", 3.785411784),
    "qt": ("volume", 0.946352946),
    "cup": ("volume", 0.2365882365),
    # time (base: second)
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    "day": ("time", 86400.0),
    "week": ("time", 604800.0),
    # speed (base: m/s)
    "m/s": ("speed", 1.0),
    "km/h": ("speed", 1000.0 / 3600.0),
    "mph": ("speed", 1609.344 / 3600.0),
    # data (base: byte, decimal prefixes)
    "b": ("data", 1.0),
    "kb": ("data", 1e3),
    "mb": ("data", 1e6),
    "gb": ("data", 1e9),
    "tb": ("data", 1e12),
    # temperature handled separately
    "c": ("temperature", 1.0),
    "f": ("temperature", 1.0),
    "k": ("temperature", 1.0),
}

_ALIASES: dict[str, str] = {}
for _canon, _names in {
    "m": ["m", "meter", "meters", "metre", "metres"],
    "km": ["km", "kms", "kilometer", "kilometers", "kilometre", "kilometres"],
    "cm": ["cm", "centimeter", "centimeters", "centimetre", "centimetres"],
    "mm": ["mm", "millimeter", "millimeters", "millimetre", "millimetres"],
    "mi": ["mi", "mile", "miles"],
    "yd": ["yd", "yard", "yards"],
    "ft": ["ft", "foot", "feet"],
    "in": ["in", "inch", "inches"],
    "kg": ["kg", "kgs", "kilogram", "kilograms", "kilo", "kilos"],
    "g": ["g", "gram", "grams"],
    "mg": ["mg", "milligram", "milligrams"],
    "lb": ["lb", "lbs", "pound", "pounds"],
    "oz": ["oz", "ounce", "ounces"],
    "t": ["t", "tonne", "tonnes", "metric ton", "metric tons", "ton", "tons"],
    "l": ["l", "liter", "liters", "litre", "litres"],
    "ml": ["ml", "milliliter", "milliliters", "millilitre", "millilitres"],
    "gal": ["gal", "gallon", "gallons"],
    "qt": ["qt", "quart", "quarts"],
    "cup": ["cup", "cups"],
    "s": ["s", "sec", "secs", "second", "seconds"],
    "min": ["min", "mins", "minute", "minutes"],
    "h": ["h", "hr", "hrs", "hour", "hours"],
    "day": ["d", "day", "days"],
    "week": ["wk", "week", "weeks"],
    "m/s": ["m/s", "mps", "meters per second", "metres per second"],
    "km/h": ["km/h", "kmh", "kph", "kilometers per hour", "kilometres per hour"],
    "mph": ["mph", "mi/h", "miles per hour"],
    "b": ["b", "byte", "bytes"],
    "kb": ["kb", "kilobyte", "kilobytes"],
    "mb": ["mb", "megabyte", "megabytes"],
    "gb": ["gb", "gigabyte", "gigabytes"],
    "tb": ["tb", "terabyte", "terabytes"],
    "c": ["c", "°c", "celsius", "degc", "deg c", "degrees celsius", "centigrade"],
    "f": ["f", "°f", "fahrenheit", "degf", "deg f", "degrees fahrenheit"],
    "k": ["k", "kelvin"],
}.items():
    for _n in _names:
        _ALIASES[_n] = _canon


class UnitError(ValueError):
    pass


def normalize_unit(unit: str) -> str:
    if not isinstance(unit, str):
        raise UnitError(f"unit must be a string, got {type(unit).__name__}")
    key = unit.strip().lower().replace("degrees ", "degrees ").rstrip(".")
    if key in _ALIASES:
        return _ALIASES[key]
    key2 = key.replace(" ", "")
    for alias, canon in _ALIASES.items():
        if alias.replace(" ", "") == key2:
            return canon
    raise UnitError(f"unknown unit: {unit!r}")


def _temperature(value: float, src: str, dst: str) -> float:
    if src == "c":
        kelvin = value + 273.15
    elif src == "f":
        kelvin = (value - 32.0) * 5.0 / 9.0 + 273.15
    else:
        kelvin = value
    if dst == "c":
        return kelvin - 273.15
    if dst == "f":
        return (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    return kelvin


def convert(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert value from one unit to another. Returns {"value": number, "unit": canonical_to_unit}."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise UnitError(f"value must be numeric, got {value!r}") from exc
    src, dst = normalize_unit(from_unit), normalize_unit(to_unit)
    cat_src, f_src = _UNITS[src]
    cat_dst, f_dst = _UNITS[dst]
    if cat_src != cat_dst:
        raise UnitError(f"cannot convert {cat_src} ({src}) to {cat_dst} ({dst})")
    if cat_src == "temperature":
        result = _temperature(value, src, dst)
    else:
        result = value * f_src / f_dst
    result = round(result, 6)
    if float(result).is_integer():
        result = int(result)
    return {"value": result, "unit": dst, "input": {"value": value, "unit": src}}
