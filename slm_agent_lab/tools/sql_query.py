"""Read-only SQL over a small, deterministically generated company database (SQLite, in-memory)."""

from __future__ import annotations

import random
import re
import sqlite3
import threading
from pathlib import Path

SEED = 20240601
MAX_ROWS = 50

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Support"]
DEPT_WEIGHTS = [22, 12, 8, 6, 4, 8]
CITIES = ["Bangalore", "Gurgaon", "Mumbai", "Pune", "Hyderabad", "Delhi"]
REGIONS = ["North", "South", "East", "West"]
SALARY_RANGE = {
    "Engineering": (70000, 160000),
    "Sales": (45000, 120000),
    "Marketing": (42000, 105000),
    "Finance": (50000, 130000),
    "HR": (40000, 95000),
    "Support": (35000, 80000),
}
FIRST_NAMES = [
    "Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Ananya", "Kabir", "Priya", "Vihaan", "Sneha",
    "Arjun", "Nisha", "Dev", "Pooja", "Rahul", "Kavya", "Aditya", "Riya", "Karan", "Tara",
    "Nikhil", "Sana", "Yash", "Zara", "Manav", "Isha", "Omar", "Leela", "Siddharth", "Anika",
    "Vikram", "Neha", "Raj", "Simran", "Varun", "Aisha", "Harsh", "Mira", "Aman", "Jaya",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Nair", "Patel", "Reddy", "Singh", "Mehta", "Kapoor", "Joshi",
    "Rao", "Das", "Bose", "Menon", "Gupta", "Chopra", "Malhotra", "Pillai", "Desai", "Kulkarni",
    "Bhat", "Saxena", "Mishra", "Khan", "Jain", "Agarwal", "Banerjee", "Chauhan", "Dutta", "Ghosh",
]
CUSTOMERS = [
    "Acme Corp", "Globex", "Initech", "Umbrella Ltd", "Stark Industries", "Wayne Enterprises",
    "Hooli", "Pied Piper", "Vandelay Industries", "Soylent", "Tyrell Corp", "Cyberdyne",
    "Wonka Industries", "Dunder Mifflin", "Aperture Labs",
]
# (name, category, unit_price, stock)
PRODUCTS = [
    ("Laptop Pro 14", "Computers", 1299.0, 85),
    ("Laptop Air 13", "Computers", 999.0, 120),
    ("Tablet T10", "Computers", 599.0, 70),
    ("Monitor 27", "Displays", 449.0, 60),
    ("Monitor 32", "Displays", 699.0, 35),
    ("Dock X", "Accessories", 189.0, 210),
    ("Keyboard K2", "Accessories", 89.0, 340),
    ("Mouse M1", "Accessories", 39.0, 500),
    ("Webcam W1", "Accessories", 59.0, 175),
    ("Headset H1", "Audio", 129.0, 150),
    ("Speaker S2", "Audio", 79.0, 90),
    ("Printer P3", "Printing", 349.0, 25),
]

SCHEMA = """
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, city TEXT, salary INTEGER, hire_year INTEGER);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, product TEXT, region TEXT, quantity INTEGER, unit_price REAL, order_month TEXT);
CREATE TABLE products (name TEXT PRIMARY KEY, category TEXT, unit_price REAL, stock INTEGER);
"""

SCHEMA_DESCRIPTION = (
    "employees(id, name, department, city, salary, hire_year) where department is one of "
    + "/".join(DEPARTMENTS) + " and city is one of " + "/".join(CITIES) + "; "
    "orders(id, customer, product, region, quantity, unit_price, order_month) where region is one of "
    + "/".join(REGIONS) + " and order_month is 'YYYY-MM' (e.g. '2024-12'); "
    "products(name, category, unit_price, stock)"
)

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b", re.IGNORECASE
)


class SQLError(ValueError):
    pass


def generate_rows(seed: int = SEED) -> dict[str, list[tuple]]:
    """Deterministic synthetic data. Same seed -> identical database on every machine."""
    rng = random.Random(seed)
    names: set[str] = set()
    employees = []
    emp_id = 1
    while len(employees) < 60:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name in names:
            continue
        names.add(name)
        dept = rng.choices(DEPARTMENTS, weights=DEPT_WEIGHTS, k=1)[0]
        lo, hi = SALARY_RANGE[dept]
        salary = rng.randint(lo // 1000, hi // 1000) * 1000
        employees.append((emp_id, name, dept, rng.choice(CITIES), salary, rng.randint(2012, 2024)))
        emp_id += 1

    price_of = {p[0]: p[2] for p in PRODUCTS}
    orders = []
    for order_id in range(1, 121):
        product = rng.choice(PRODUCTS)[0]
        orders.append(
            (
                order_id,
                rng.choice(CUSTOMERS),
                product,
                rng.choice(REGIONS),
                rng.randint(1, 40),
                price_of[product],
                f"2024-{rng.randint(1, 12):02d}",
            )
        )
    return {"employees": employees, "orders": orders, "products": list(PRODUCTS)}


def build_connection(seed: int = SEED) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SCHEMA)
    rows = generate_rows(seed)
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", rows["employees"])
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", rows["orders"])
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", rows["products"])
    conn.commit()
    return conn


_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            _conn = build_connection()
        return _conn


def export_sqlite(path: str | Path) -> Path:
    """Write the in-memory DB to disk for manual inspection (e.g. with the sqlite3 CLI)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    disk = sqlite3.connect(str(path))
    get_connection().backup(disk)
    disk.close()
    return path


def _validate(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise SQLError("query must be a non-empty string")
    q = query.strip().strip("`").strip()
    if q.lower().startswith("sql"):
        q = q[3:].strip()
    q = q.rstrip(";").strip()
    if ";" in q:
        raise SQLError("only a single statement is allowed")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise SQLError("only SELECT queries are allowed")
    if _FORBIDDEN.search(q):
        raise SQLError("query contains a forbidden keyword (database is read-only)")
    return q


def run_query(query: str, max_rows: int = MAX_ROWS) -> dict:
    """Execute a read-only SELECT. Returns {"columns": [...], "rows": [[...]], "row_count": n, "truncated": bool}."""
    q = _validate(query)
    conn = get_connection()
    with _lock:
        try:
            cur = conn.execute(q)
            rows = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
        except sqlite3.Error as exc:
            raise SQLError(f"SQL error: {exc}") from exc
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return {
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }


def scalar(query: str):
    """Helper for ground-truth generation: first column of first row."""
    out = run_query(query)
    if not out["rows"]:
        return None
    return out["rows"][0][0]
