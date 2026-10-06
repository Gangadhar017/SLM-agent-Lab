"""Programmatic, seeded task generation.

Every task's ground truth is computed with the *same* tool implementations the agent uses (calculator, unit
conversion, SQL over the synthetic DB, fact tables of the corpus), so there is no label noise.

Splits:
  * test  (seed S)   -> held-out evaluation set. Contains "ood" templates/documents that never appear in train.
  * train (seed S+1) -> used only to build SFT/distillation data. Same templates minus the OOD ones, new numbers.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Callable

from ..tools import calculator as calc
from ..tools import doc_search as ds
from ..tools import sql_query as sq
from ..tools import unit_convert as uc
from .schema import Task

DEFAULT_COUNTS = {
    "calc_single": 60,
    "convert_single": 60,
    "sql_single": 60,
    "doc_single": 40,
    "multi_step": 50,
    "no_tool": 20,
    "unanswerable": 10,
}

# Documents whose facts are held out of the train split entirely (tests generalisation to unseen documents).
OOD_DOCS = {"wellness-program", "printer-p3-spec", "product-return-policy", "headset-h1-spec"}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
UNIT_LONG = {
    "mi": "miles", "km": "kilometres", "m": "metres", "ft": "feet", "in": "inches", "cm": "centimetres",
    "yd": "yards", "lb": "pounds", "kg": "kilograms", "oz": "ounces", "g": "grams", "t": "metric tons",
    "c": "degrees Celsius", "f": "degrees Fahrenheit", "k": "kelvin", "gal": "gallons", "l": "litres",
    "qt": "quarts", "ml": "millilitres", "cup": "cups", "h": "hours", "min": "minutes", "s": "seconds",
    "day": "days", "week": "weeks", "mph": "miles per hour", "km/h": "kilometres per hour",
    "m/s": "metres per second", "gb": "gigabytes", "mb": "megabytes", "tb": "terabytes",
}


# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
def fmt(v) -> str:
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def num(value, abs_tol: float | None = None) -> dict:
    v = float(value)
    if abs_tol is None:
        abs_tol = 0.5 if v.is_integer() else max(0.011, 0.005 * abs(v))
    return {"type": "number", "value": value, "abs_tol": abs_tol}


def string(*any_of: str) -> dict:
    return {"type": "string", "any_of": list(any_of)}


def task_dict(template, prompt, expected, tools, gold_calls, value_text, ood=False, meta=None) -> dict:
    return {
        "template": template,
        "prompt": prompt,
        "expected": expected,
        "expected_tools": tools,
        "gold_calls": gold_calls,
        "gold_answer": f"Final answer: {value_text}",
        "ood": ood,
        "meta": meta or {},
    }


def call(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


def rnd(x, nd):
    return round(float(x), nd)


# --------------------------------------------------------------------------------------------------------------
# calculator tasks
# --------------------------------------------------------------------------------------------------------------
def calc_templates(rng: random.Random) -> list[Callable[[], dict]]:
    def percent_of():
        p = rng.choice([7.5, 12.5, 17.5, 22.5, 33, 42, 65, 85, 4.5, 9])
        n = rng.randint(1000, 99999)
        expr = f"{n} * {p} / 100"
        v = calc.calculate(expr)["result"]
        prompt = rng.choice([f"What is {p}% of {n:,}?", f"Calculate {p} percent of {n}.",
                             f"If a company earns {n:,} USD and pays {p}% in tax, how much tax does it pay?"])
        return task_dict("calc_percent_of", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v))

    def discount():
        price = rng.randint(5000, 200000) / 100
        p = rng.choice([5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 60])
        expr = f"{price} * (1 - {p} / 100)"
        v = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"An item costs {price:.2f} USD and is discounted by {p}%. What is the final price? Round to 2 decimals."
        return task_dict("calc_discount", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v))

    def sqrt():
        n = rng.randint(100, 99999)
        if math.isqrt(n) ** 2 == n:
            n += 1
        expr = f"sqrt({n})"
        v = rnd(calc.calculate(expr)["result"], 3)
        prompt = f"What is the square root of {n}? Round to 3 decimal places."
        return task_dict("calc_sqrt", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v))

    def speed():
        d = rng.randint(80, 900)
        h = rng.choice([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5])
        expr = f"{d} / {h}"
        v = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"A car travels {d} km in {fmt(h)} hours. What is its average speed in km/h? Round to 2 decimals."
        return task_dict("calc_speed", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v))

    def expression():
        a, b, c, d = rng.randint(11, 99), rng.randint(11, 99), rng.randint(3, 19), rng.randint(100, 999)
        expr = f"({a} + {b}) * {c} - {d}"
        v = calc.calculate(expr)["result"]
        prompt = rng.choice([f"Compute ({a} + {b}) × {c} − {d}.", f"What is ({a} + {b}) * {c} - {d}?",
                             f"Evaluate the expression ({a} + {b}) * {c} - {d}."])
        return task_dict("calc_expression", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v))

    def division():
        a, b = rng.randint(1000, 99999), rng.choice([3, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 53, 67, 79, 97])
        expr = f"{a} / {b}"
        v = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"What is {a:,} divided by {b}? Round to 2 decimal places."
        return task_dict("calc_division", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v))

    def compound():  # OOD template
        principal = rng.randint(2, 100) * 500
        r = rng.choice([3, 4.5, 5, 6.5, 7, 8, 9.5])
        t = rng.randint(2, 10)
        expr = f"{principal} * (1 + {r} / 100) ** {t}"
        v = rnd(calc.calculate(expr)["result"], 2)
        prompt = (f"{principal:,} USD is invested at {fmt(r)}% annual interest compounded yearly for {t} years. "
                  f"What is the final amount? Round to 2 decimals.")
        return task_dict("calc_compound", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v), ood=True)

    def circle_area():  # OOD template
        radius = rng.randint(15, 400) / 10
        expr = f"pi * {radius} ** 2"
        v = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"What is the area of a circle with radius {radius} metres? Round to 2 decimals."
        return task_dict("calc_circle_area", prompt, num(v), ["calculator"], [call("calculator", expression=expr)], fmt(v), ood=True)

    return [percent_of, discount, sqrt, speed, expression, division, compound, circle_area]


# --------------------------------------------------------------------------------------------------------------
# unit conversion tasks
# --------------------------------------------------------------------------------------------------------------
def convert_templates(rng: random.Random) -> list[Callable[[], dict]]:
    def make(template, pairs, phrasings, ood=False):
        def gen():
            a, b = rng.choice(pairs)
            value = rng.randint(15, 9999) / 10
            if rng.random() < 0.4:
                value = float(rng.randint(2, 999))
            v = uc.convert(value, a, b)["value"]
            v = rnd(v, 3)
            prompt = rng.choice(phrasings).format(v=fmt(value), a=UNIT_LONG[a], b=UNIT_LONG[b])
            gc = [call("unit_convert", value=value, from_unit=a, to_unit=b)]
            return task_dict(template, prompt, num(v), ["unit_convert"], gc, fmt(v), ood=ood, meta={"from": a, "to": b})
        return gen

    generic = ["Convert {v} {a} to {b}.", "How many {b} are {v} {a}?", "Express {v} {a} in {b}. Round to 3 decimals."]
    length = make("convert_length", [("mi", "km"), ("km", "mi"), ("ft", "m"), ("m", "ft"), ("in", "cm"), ("cm", "in"),
                                     ("yd", "m")], generic + ["A road is {v} {a} long. What is its length in {b}?"])
    mass = make("convert_mass", [("lb", "kg"), ("kg", "lb"), ("oz", "g"), ("g", "oz"), ("t", "kg")],
                generic + ["A package weighs {v} {a}. What is its mass in {b}?"])
    temp = make("convert_temperature", [("f", "c"), ("c", "f"), ("c", "k"), ("k", "c")],
                ["Convert {v} {a} to {b}.", "The temperature is {v} {a}. What is that in {b}?"])
    volume = make("convert_volume", [("gal", "l"), ("l", "gal"), ("qt", "l"), ("l", "ml"), ("cup", "ml")], generic)
    time_ = make("convert_time", [("h", "min"), ("day", "h"), ("week", "day"), ("min", "s"), ("h", "s")], generic, ood=True)
    speed_data = make("convert_speed_data", [("mph", "km/h"), ("km/h", "m/s"), ("gb", "mb"), ("tb", "gb"), ("km/h", "mph")],
                      generic, ood=True)
    return [length, mass, temp, volume, time_, speed_data]


# --------------------------------------------------------------------------------------------------------------
# SQL tasks (ground truth = the gold query executed on the same DB)
# --------------------------------------------------------------------------------------------------------------
def sql_templates(rng: random.Random) -> list[Callable[[], dict]]:
    products = [p[0] for p in sq.PRODUCTS]
    categories = sorted({p[1] for p in sq.PRODUCTS})

    def mk(template, prompt, query, expected, value_text, ood=False):
        return task_dict(template, prompt, expected, ["sql_query"], [call("sql_query", query=query)], value_text, ood=ood)

    def count_dept():
        d = rng.choice(sq.DEPARTMENTS)
        q = f"SELECT COUNT(*) FROM employees WHERE department = '{d}'"
        v = sq.scalar(q)
        prompt = rng.choice([f"How many employees work in the {d} department?", f"What is the headcount of the {d} department?",
                             f"Count the employees whose department is {d}."])
        return mk("sql_count_dept", prompt, q, num(v), fmt(v))

    def avg_salary_dept():
        d = rng.choice(sq.DEPARTMENTS)
        q = f"SELECT ROUND(AVG(salary)) FROM employees WHERE department = '{d}'"
        v = int(sq.scalar(q))
        prompt = f"What is the average salary of employees in the {d} department? Round to the nearest whole number."
        return mk("sql_avg_salary_dept", prompt, q, num(v, abs_tol=1.0), fmt(v))

    def total_salary_city():
        c = rng.choice(sq.CITIES)
        q = f"SELECT SUM(salary) FROM employees WHERE city = '{c}'"
        v = sq.scalar(q)
        prompt = rng.choice([f"What is the total annual salary paid to employees based in {c}?",
                             f"Sum the salaries of all employees located in {c}."])
        return mk("sql_total_salary_city", prompt, q, num(v), fmt(v))

    def dept_most_employees():
        q = "SELECT department FROM employees GROUP BY department ORDER BY COUNT(*) DESC LIMIT 1"
        v = sq.scalar(q)
        return mk("sql_dept_most_employees", "Which department has the most employees?", q, string(v), v)

    def top_paid_in_dept():
        d = rng.choice(sq.DEPARTMENTS)
        q = f"SELECT name FROM employees WHERE department = '{d}' ORDER BY salary DESC LIMIT 1"
        v = sq.scalar(q)
        prompt = rng.choice([f"Who is the highest-paid employee in the {d} department?",
                             f"Which {d} employee has the highest salary?"])
        return mk("sql_top_paid_in_dept", prompt, q, string(v), v)

    def count_orders_region():
        r = rng.choice(sq.REGIONS)
        q = f"SELECT COUNT(*) FROM orders WHERE region = '{r}'"
        v = sq.scalar(q)
        prompt = rng.choice([f"How many orders were placed in the {r} region?", f"How many orders does the {r} region have?"])
        return mk("sql_count_orders_region", prompt, q, num(v), fmt(v))

    def revenue_product():
        p = rng.choice(products)
        q = f"SELECT SUM(quantity * unit_price) FROM orders WHERE product = '{p}'"
        v = sq.scalar(q) or 0
        prompt = f"What is the total revenue (quantity times unit price) from all orders of '{p}'?"
        return mk("sql_revenue_product", prompt, q, num(v), fmt(v))

    def stock_product():
        p = rng.choice(products)
        q = f"SELECT stock FROM products WHERE name = '{p}'"
        v = sq.scalar(q)
        prompt = rng.choice([f"How many units of '{p}' are currently in stock?", f"What is the stock level of the {p}?"])
        return mk("sql_stock_product", prompt, q, num(v), fmt(v))

    def count_city_dept():
        d, c = rng.choice(sq.DEPARTMENTS), rng.choice(sq.CITIES)
        q = f"SELECT COUNT(*) FROM employees WHERE department = '{d}' AND city = '{c}'"
        v = sq.scalar(q)
        prompt = f"How many {d} employees are based in {c}?"
        return mk("sql_count_city_dept", prompt, q, num(v), fmt(v))

    def orders_month_qty():
        m = rng.randint(1, 12)
        q = f"SELECT SUM(quantity) FROM orders WHERE order_month = '2024-{m:02d}'"
        v = sq.scalar(q) or 0
        prompt = f"What is the total quantity of items ordered in {MONTHS[m - 1]} 2024?"
        return mk("sql_orders_month_qty", prompt, q, num(v), fmt(v))

    def hire_year_mode():  # OOD
        q = "SELECT hire_year FROM employees GROUP BY hire_year ORDER BY COUNT(*) DESC, hire_year ASC LIMIT 1"
        rows = sq.run_query("SELECT hire_year, COUNT(*) AS n FROM employees GROUP BY hire_year ORDER BY n DESC LIMIT 2")["rows"]
        if len(rows) == 2 and rows[0][1] == rows[1][1]:
            return None  # tie -> ambiguous, skip
        v = sq.scalar(q)
        return mk("sql_hire_year_mode", "In which year were the most employees hired?", q, num(v), fmt(v), ood=True)

    def max_price_category():  # OOD
        c = rng.choice(categories)
        q = f"SELECT MAX(unit_price) FROM products WHERE category = '{c}'"
        v = sq.scalar(q)
        prompt = f"What is the highest unit price among products in the '{c}' category?"
        return mk("sql_max_price_category", prompt, q, num(v), fmt(v), ood=True)

    def customer_orders():
        cust = rng.choice(sq.CUSTOMERS)
        q = f"SELECT COUNT(*) FROM orders WHERE customer = '{cust}'"
        v = sq.scalar(q)
        prompt = f"How many orders has the customer '{cust}' placed?"
        return mk("sql_customer_orders", prompt, q, num(v), fmt(v))

    return [count_dept, avg_salary_dept, total_salary_city, dept_most_employees, top_paid_in_dept, count_orders_region,
            revenue_product, stock_product, count_city_dept, orders_month_qty, hire_year_mode, max_price_category,
            customer_orders]


# --------------------------------------------------------------------------------------------------------------
# document tasks
# --------------------------------------------------------------------------------------------------------------
def all_facts() -> list[dict]:
    out = []
    for d in ds.load_corpus():
        for f in d["facts"]:
            out.append({**f, "doc_id": d["id"], "title": d["title"]})
    return out


def fact_lookup() -> dict[str, dict]:
    return {f["key"]: f for f in all_facts()}


def doc_task(fact: dict) -> dict:
    ood = fact["doc_id"] in OOD_DOCS
    return task_dict("doc_fact", fact["q"], num(fact["value"]), ["doc_search"],
                     [call("doc_search", query=fact["title"])], fmt(fact["value"]), ood=ood,
                     meta={"doc_id": fact["doc_id"], "fact_key": fact["key"]})


# --------------------------------------------------------------------------------------------------------------
# multi-step tasks
# --------------------------------------------------------------------------------------------------------------
def multi_fixed() -> list[dict]:
    """Hand-designed chains over the corpus facts. Values are derived from the fact table, not typed in."""
    F = fact_lookup()
    v = lambda k: F[k]["value"]  # noqa: E731
    t = lambda k: F[k]["title"]  # noqa: E731
    items: list[dict] = []

    def dc(prompt, key, expr, nd=2, ood=False, tools=("doc_search", "calculator")):
        value = rnd(calc.calculate(expr)["result"], nd)
        items.append(task_dict("multi_doc_calc", prompt, num(value), list(tools),
                               [call("doc_search", query=t(key)), call("calculator", expression=expr)], fmt(value),
                               ood=ood or F[key]["doc_id"] in OOD_DOCS, meta={"doc_id": F[key]["doc_id"]}))

    def dconv(prompt, key, a, b, nd=3):
        value = rnd(uc.convert(v(key), a, b)["value"], nd)
        items.append(task_dict("multi_doc_convert", prompt, num(value), ["doc_search", "unit_convert"],
                               [call("doc_search", query=t(key)), call("unit_convert", value=v(key), from_unit=a, to_unit=b)],
                               fmt(value), ood=F[key]["doc_id"] in OOD_DOCS, meta={"doc_id": F[key]["doc_id"]}))

    dc("According to the Q1 cloud cost report, how much was spent on compute in Q1, in USD?", "cloud_spend_q1",
       f"{v('cloud_spend_q1')} * {v('cloud_compute_pct')} / 100")
    dc("According to the Q1 cloud cost report, how much was spent on storage in Q1, in USD?", "cloud_spend_q1",
       f"{v('cloud_spend_q1')} * {v('cloud_storage_pct')} / 100")
    dc("Using the Q1 cloud cost report and its growth forecast, what is the forecast cloud spend for Q2 in USD?",
       "cloud_spend_q1", f"{v('cloud_spend_q1')} * (1 + {v('cloud_growth_pct')} / 100)")
    dc("Using the summer marketing campaign summary, what was the cost per lead in USD? Round to 2 decimals.",
       "campaign_budget", f"{v('campaign_budget')} / {v('campaign_leads')}")
    dc("Using the summer marketing campaign summary, how many paying customers did the campaign produce? Round to the "
       "nearest whole number.", "campaign_leads", f"round({v('campaign_leads')} * {v('campaign_conversion_pct')} / 100)", 0)
    dc("Using the electric vehicle fleet document, what is the combined battery capacity of all vans in kWh?",
       "ev_vans", f"{v('ev_vans')} * {v('ev_battery_kwh')}")
    dc("Using the electric vehicle fleet document, what is the combined maximum range of the whole fleet in kilometres?",
       "ev_vans", f"{v('ev_vans')} * {v('ev_range_km')}")
    dc("Using the learning budget document, what is the maximum amount in USD an employee may spend on conference "
       "travel per year?", "learning_budget", f"{v('learning_budget')} * {v('conference_pct')} / 100")
    dc("Using the Hyderabad data centre rack specification, what is the total power draw in kilowatts of all racks "
       "when fully loaded?", "rack_count", f"{v('rack_count')} * {v('rack_power_kw')}")
    dc("Using the Hyderabad data centre rack specification, how many rack units are there in total across all racks?",
       "rack_count", f"{v('rack_count')} * {v('rack_units')}")
    dc("Using the office locations document, what is the combined seating capacity of the Bangalore and Gurgaon offices?",
       "bangalore_seats", f"{v('bangalore_seats')} + {v('gurgaon_seats')}")
    dc("Using the office locations document, what is the combined floor area in square feet of the two offices?",
       "bangalore_sqft", f"{v('bangalore_sqft')} + {v('gurgaon_sqft')}")
    dc("Using the cafeteria document, how much does an employee pay out of pocket in INR for one standard meal after "
       "the daily subsidy?", "meal_price_inr", f"{v('meal_price_inr')} - {v('meal_subsidy_inr')}")
    dc("Using the leave policy, how many total paid leave days (annual leave plus sick leave) does a full-time employee "
       "get per year?", "annual_leave_days", f"{v('annual_leave_days')} + {v('sick_leave_days')}")
    dc("Using the warehouse logistics document, how many full truck trips are needed to move the warehouse's entire "
       "storage capacity? Round up to a whole number.", "warehouse_capacity_pallets",
       f"ceil({v('warehouse_capacity_pallets')} / {v('pallets_per_truck')})", 0)
    dc("Using the Keyboard K2 specification, how many full days does its battery last on a single charge?",
       "k2_battery_hours", f"{v('k2_battery_hours')} / 24", 0)
    dc("Using the remote work policy, what is the total internet allowance in USD a remote worker receives over a full "
       "year of 12 months?", "internet_allowance", f"{v('internet_allowance')} * 12")
    dc("Using the remote work policy, how much in USD does a remote worker receive in their first year, counting the "
       "one-time stipend plus 12 monthly internet allowances?", "home_office_stipend",
       f"{v('home_office_stipend')} + 12 * {v('internet_allowance')}")
    dc("Using the wellness program document, what is the maximum annual gym reimbursement in USD?", "gym_reimbursement",
       f"{v('gym_reimbursement')} * 12")
    dc("Using the Printer P3 specification, how many minutes does it take to print 800 pages at its rated speed?",
       "p3_ppm", f"800 / {v('p3_ppm')}")
    dc("Using the Headset H1 specification, how many full charges are needed for 150 hours of use? Round up.",
       "h1_battery_hours", f"ceil(150 / {v('h1_battery_hours')})", 0)
    dc("Using the data retention policy, for how many weeks are database backups kept?", "backup_retention_days",
       f"{v('backup_retention_days')} / 7")
    dc("Using the sales commission plan, what commission in USD does a representative earn on a single deal of 120,000 "
       "USD?", "base_commission_pct", f"120000 * ({v('base_commission_pct')} + {v('bonus_commission_pct')}) / 100")
    dc("Using the sales commission plan, what commission in USD does a representative earn on a single deal of 35,000 "
       "USD?", "base_commission_pct", f"35000 * {v('base_commission_pct')} / 100")
    dc("Using the sales commission plan, what commission in USD does a representative earn on a single deal of 75,000 "
       "USD?", "base_commission_pct", f"75000 * ({v('base_commission_pct')} + {v('bonus_commission_pct')}) / 100")
    dc("Using the support SLA document, how many minutes does the Priority 1 resolution target allow?",
       "p1_resolve_hours", f"{v('p1_resolve_hours')} * 60", 0)
    dc("Using the product return policy, how much store credit in USD does a customer receive for a 400 USD item "
       "returned on day 45?", "restocking_fee_pct", f"400 * (1 - {v('restocking_fee_pct')} / 100)")

    dconv("The warehouse logistics document gives the distance from the Pune warehouse to the Mumbai port in kilometres. "
          "What is that distance in miles?", "warehouse_port_km", "km", "mi")
    dconv("Using the electric vehicle fleet document, what is the range of one electric van in miles?", "ev_range_km", "km", "mi")
    dconv("Using the electric vehicle fleet document, how many miles does the fleet travel per week?", "ev_weekly_km", "km", "mi")
    dconv("Using the data centre rack specification, what ambient temperature in degrees Fahrenheit does the Hyderabad "
          "data centre maintain?", "dc_temp_c", "c", "f")
    dconv("Using the Laptop Pro 14 specification, what is the laptop's weight in kilograms?", "laptop_pro_weight_lb", "lb", "kg")
    dconv("Using the Monitor 27 specification, what is the monitor's weight in kilograms?", "monitor27_weight_lb", "lb", "kg")
    dconv("Using the data centre rack specification, how much does a fully loaded rack weigh in kilograms?", "rack_weight_lb", "lb", "kg")
    dconv("Using the Keyboard K2 specification, what is the keyboard's weight in grams?", "k2_weight_lb", "lb", "g")
    dconv("Using the Headset H1 specification, what is the headset's weight in grams?", "h1_weight_oz", "oz", "g")
    dconv("Using the Printer P3 specification, what is the printer's weight in kilograms?", "p3_weight_lb", "lb", "kg")
    dconv("Using the Dock X specification, what is the dock's weight in pounds?", "dockx_weight_kg", "kg", "lb")
    dconv("Using the leave policy, how many days of parental leave do primary caregivers receive?", "parental_leave_weeks", "week", "day")
    dconv("Using the IT security policy, how many hours of inactivity-free use pass before a laptop locks? Express the "
          "lock timeout in seconds.", "laptop_lock_minutes", "min", "s")

    # 4-tool chains: doc fact (weight) x DB stock -> total weight -> kg. OOD by design (held out of training).
    for product, key in [("Laptop Pro 14", "laptop_pro_weight_lb"), ("Monitor 27", "monitor27_weight_lb"),
                         ("Keyboard K2", "k2_weight_lb"), ("Printer P3", "p3_weight_lb"), ("Headset H1", "h1_weight_oz")]:
        unit = F[key]["unit"]
        stock_q = f"SELECT stock FROM products WHERE name = '{product}'"
        stock = sq.scalar(stock_q)
        expr = f"{v(key)} * {stock}"
        total = calc.calculate(expr)["result"]
        kg = rnd(uc.convert(total, unit, "kg")["value"], 1)
        prompt = (f"Use the {product} specification document to find its weight and the products table to find how many "
                  f"units are in stock. What is the total weight of all {product} units in stock, in kilograms? Round to 1 decimal.")
        items.append(task_dict("multi_doc_sql_convert", prompt, num(kg), ["doc_search", "sql_query", "calculator", "unit_convert"],
                               [call("doc_search", query=t(key)), call("sql_query", query=stock_q),
                                call("calculator", expression=expr), call("unit_convert", value=total, from_unit=unit, to_unit="kg")],
                               fmt(kg), ood=True, meta={"doc_id": F[key]["doc_id"]}))
    return items


def multi_param_templates(rng: random.Random) -> list[Callable[[], dict]]:
    F = fact_lookup()

    def perdiem():
        n = rng.randint(3, 14)
        kind, key = rng.choice([("domestic", "meal_perdiem_domestic"), ("international", "meal_perdiem_international")])
        expr = f"{F[key]['value']} * {n}"
        val = calc.calculate(expr)["result"]
        prompt = f"Using the travel policy, what is the total meal per-diem in USD for a {n}-day {kind} trip?"
        return task_dict("multi_doc_calc_perdiem", prompt, num(val), ["doc_search", "calculator"],
                         [call("doc_search", query=F[key]["title"]), call("calculator", expression=expr)], fmt(val))

    def hotel():
        n = rng.randint(2, 9)
        expr = f"{F['hotel_cap']['value']} * {n}"
        val = calc.calculate(expr)["result"]
        prompt = f"Using the travel policy, what is the maximum hotel spend in USD for a {n}-night stay without director approval?"
        return task_dict("multi_doc_calc_hotel", prompt, num(val), ["doc_search", "calculator"],
                         [call("doc_search", query=F["hotel_cap"]["title"]), call("calculator", expression=expr)], fmt(val))

    def meals():
        n = rng.randint(5, 22)
        expr = f"({F['meal_price_inr']['value']} - {F['meal_subsidy_inr']['value']}) * {n}"
        val = calc.calculate(expr)["result"]
        prompt = f"Using the cafeteria document, how much does an employee pay out of pocket in INR for standard meals on {n} working days?"
        return task_dict("multi_doc_calc_meals", prompt, num(val), ["doc_search", "calculator"],
                         [call("doc_search", query=F["meal_price_inr"]["title"]), call("calculator", expression=expr)], fmt(val))

    def avg_salary_thousands():
        d = rng.choice(sq.DEPARTMENTS)
        q = f"SELECT AVG(salary) FROM employees WHERE department = '{d}'"
        avg = sq.scalar(q)
        expr = f"{avg} / 1000"
        val = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"What is the average salary in the {d} department expressed in thousands? Round to 2 decimals."
        return task_dict("multi_sql_calc_thousands", prompt, num(val), ["sql_query", "calculator"],
                         [call("sql_query", query=q), call("calculator", expression=expr)], fmt(val))

    def combined_headcount():
        d1, d2 = rng.sample(sq.DEPARTMENTS, 2)
        q1 = f"SELECT COUNT(*) FROM employees WHERE department = '{d1}'"
        q2 = f"SELECT COUNT(*) FROM employees WHERE department = '{d2}'"
        n1, n2 = sq.scalar(q1), sq.scalar(q2)
        expr = f"{n1} + {n2}"
        val = calc.calculate(expr)["result"]
        prompt = f"What is the combined headcount of the {d1} and {d2} departments?"
        return task_dict("multi_sql_calc_headcount", prompt, num(val), ["sql_query", "calculator"],
                         [call("sql_query", query=q1), call("sql_query", query=q2), call("calculator", expression=expr)], fmt(val))

    def salary_difference():
        d1, d2 = rng.sample(sq.DEPARTMENTS, 2)
        q1 = f"SELECT AVG(salary) FROM employees WHERE department = '{d1}'"
        q2 = f"SELECT AVG(salary) FROM employees WHERE department = '{d2}'"
        a1, a2 = sq.scalar(q1), sq.scalar(q2)
        expr = f"round({a1} - {a2})"
        val = calc.calculate(expr)["result"]
        prompt = (f"What is the difference between the average salary in the {d1} department and the average salary in the "
                  f"{d2} department ({d1} minus {d2})? Round to the nearest whole number.")
        return task_dict("multi_sql_calc_salary_diff", prompt, num(val, abs_tol=1.0), ["sql_query", "calculator"],
                         [call("sql_query", query=q1), call("sql_query", query=q2), call("calculator", expression=expr)], fmt(val))

    def dept_share():
        d = rng.choice(sq.DEPARTMENTS)
        q1 = f"SELECT COUNT(*) FROM employees WHERE department = '{d}'"
        q2 = "SELECT COUNT(*) FROM employees"
        n, total = sq.scalar(q1), sq.scalar(q2)
        expr = f"{n} / {total} * 100"
        val = rnd(calc.calculate(expr)["result"], 1)
        prompt = f"What percentage of all employees work in the {d} department? Round to 1 decimal."
        return task_dict("multi_sql_calc_share", prompt, num(val, abs_tol=0.06), ["sql_query", "calculator"],
                         [call("sql_query", query=q1), call("sql_query", query=q2), call("calculator", expression=expr)], fmt(val))

    def revenue_share():
        p = rng.choice([x[0] for x in sq.PRODUCTS])
        q1 = f"SELECT SUM(quantity * unit_price) FROM orders WHERE product = '{p}'"
        q2 = "SELECT SUM(quantity * unit_price) FROM orders"
        a, b = sq.scalar(q1) or 0, sq.scalar(q2)
        expr = f"{a} / {b} * 100"
        val = rnd(calc.calculate(expr)["result"], 1)
        prompt = f"What is the total revenue from orders of '{p}' as a percentage of the total revenue from all orders? Round to 1 decimal."
        return task_dict("multi_sql_calc_revenue_share", prompt, num(val, abs_tol=0.06), ["sql_query", "calculator"],
                         [call("sql_query", query=q1), call("sql_query", query=q2), call("calculator", expression=expr)], fmt(val))

    def courier_speed():
        d = rng.randint(20, 400)
        h = rng.choice([1.5, 2, 2.5, 3, 4, 5])
        km = uc.convert(d, "mi", "km")["value"]
        expr = f"{km} / {h}"
        val = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"A courier drives {d} miles in {fmt(h)} hours. What is the average speed in kilometres per hour? Round to 2 decimals."
        return task_dict("multi_calc_convert_speed", prompt, num(val), ["unit_convert", "calculator"],
                         [call("unit_convert", value=d, from_unit="mi", to_unit="km"), call("calculator", expression=expr)], fmt(val))

    def recipe():
        c = rng.choice([1.5, 2, 2.5, 3, 4])
        n = rng.randint(3, 12)
        total_cups = c * n
        val = rnd(uc.convert(total_cups, "cup", "l")["value"], 3)
        prompt = f"A recipe uses {fmt(c)} cups of milk per batch. How many litres of milk are needed for {n} batches? Round to 3 decimals."
        return task_dict("multi_calc_convert_recipe", prompt, num(val), ["calculator", "unit_convert"],
                         [call("calculator", expression=f"{c} * {n}"), call("unit_convert", value=total_cups, from_unit="cup", to_unit="l")], fmt(val))

    def tank():
        g = rng.randint(50, 900)
        r = rng.choice([8, 10, 12, 15, 20, 25])
        litres = uc.convert(g, "gal", "l")["value"]
        expr = f"{litres} / {r}"
        val = rnd(calc.calculate(expr)["result"], 1)
        prompt = f"A water tank holds {g} gallons. If it is filled at {r} litres per minute, how many minutes does it take to fill? Round to 1 decimal."
        return task_dict("multi_calc_convert_tank", prompt, num(val), ["unit_convert", "calculator"],
                         [call("unit_convert", value=g, from_unit="gal", to_unit="l"), call("calculator", expression=expr)], fmt(val))

    def pace():  # OOD
        d = rng.choice([5, 8, 10, 12, 15, 21.1])
        m = rng.randint(25, 130)
        miles = uc.convert(d, "km", "mi")["value"]
        expr = f"{m} / {miles}"
        val = rnd(calc.calculate(expr)["result"], 2)
        prompt = f"A runner completes {fmt(d)} kilometres in {m} minutes. What is the pace in minutes per mile? Round to 2 decimals."
        return task_dict("multi_calc_convert_pace", prompt, num(val), ["unit_convert", "calculator"],
                         [call("unit_convert", value=d, from_unit="km", to_unit="mi"), call("calculator", expression=expr)], fmt(val), ood=True)

    return [perdiem, hotel, meals, avg_salary_thousands, combined_headcount, salary_difference, dept_share, revenue_share,
            courier_speed, recipe, tank, pace]


# --------------------------------------------------------------------------------------------------------------
# no-tool and unanswerable tasks
# --------------------------------------------------------------------------------------------------------------
NO_TOOL_QA: list[tuple[str, list[str]]] = [
    ("What does SQL stand for?", ["structured query language"]),
    ("What does CPU stand for?", ["central processing unit"]),
    ("What does HTTP stand for?", ["hypertext transfer protocol", "hyper text transfer protocol"]),
    ("What does RAM stand for?", ["random access memory", "random-access memory"]),
    ("What does API stand for?", ["application programming interface"]),
    ("What is the capital city of France?", ["paris"]),
    ("What is the capital city of Japan?", ["tokyo"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is the chemical formula for water?", ["h2o"]),
    ("How many days are there in a leap year?", ["366"]),
    ("How many minutes are in one hour?", ["60", "sixty"]),
    ("What is the boiling point of water in degrees Celsius at sea level?", ["100"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Who wrote the play 'Romeo and Juliet'?", ["shakespeare"]),
    ("What is the opposite of the word 'hot'?", ["cold"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("Which language is used to style web pages: HTML, CSS or SQL?", ["css"]),
    ("What does GPU stand for?", ["graphics processing unit"]),
    ("On which continent is India located?", ["asia"]),
    ("What is the plural of the word 'mouse' (the animal)?", ["mice"]),
    ("What does JSON stand for?", ["javascript object notation"]),
    ("How many hours are in a day?", ["24", "twenty-four", "twenty four"]),
    ("What is the freezing point of water in degrees Fahrenheit?", ["32"]),
    ("What does URL stand for?", ["uniform resource locator"]),
    ("What does HTML stand for?", ["hypertext markup language", "hyper text markup language"]),
    ("What is the capital city of Italy?", ["rome"]),
    ("What is the capital city of Germany?", ["berlin"]),
    ("What is the chemical symbol for sodium?", ["na"]),
    ("What is the chemical symbol for iron?", ["fe"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("How many months have 31 days?", ["7", "seven"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("Which gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2"]),
    ("What does PDF stand for?", ["portable document format"]),
    ("What does USB stand for?", ["universal serial bus"]),
    ("Which planet is closest to the Sun?", ["mercury"]),
    ("How many seconds are in one minute?", ["60", "sixty"]),
    ("What does SSD stand for?", ["solid state drive", "solid-state drive"]),
    ("Which ocean lies between Africa and Australia?", ["indian"]),
]

UNANSWERABLE: list[tuple[str, str]] = [
    ("What is the salary of the employee named Zubin Mistry?", "sql"),
    ("What is the salary of the employee named Farah Qureshi?", "sql"),
    ("Which department does the employee named Tenzin Dorje work in?", "sql"),
    ("Which city is the employee named Elena Petrova based in?", "sql"),
    ("Which employee has the employee ID 999?", "sql"),
    ("What is the salary of the employee named Marcus Oyelaran?", "sql"),
    ("According to the company documents, what is the reimbursement limit for pet insurance?", "doc"),
    ("According to the company documents, what is the warranty period of the Tablet T10?", "doc"),
    ("What is the company's policy on sabbatical leave according to the documents?", "doc"),
    ("According to the documents, what is the monthly parking fee at the Mumbai office?", "doc"),
    ("According to the product specification documents, how much does the Mouse M1 weigh?", "doc"),
    ("According to the company documents, what is the dress code policy for client meetings?", "doc"),
    ("What is the salary of the employee named Hiroshi Tanaka?", "sql"),
    ("According to the documents, how many charging stations does the Bangalore office parking have?", "doc"),
    ("Which employee has the employee ID 250?", "sql"),
    ("According to the company documents, what is the budget for the winter marketing campaign?", "doc"),
    ("What is the salary of the employee named Ingrid Olsen?", "sql"),
    ("According to the documents, what is the maximum file size allowed for email attachments?", "doc"),
    ("Which department does the employee named Kwame Mensah work in?", "sql"),
    ("According to the Speaker S2 specification document, what is its battery life?", "doc"),
]


def no_tool_task(q: str, answers: list[str]) -> dict:
    return task_dict("no_tool_general", q, string(*answers), [], [], answers[0])


def unanswerable_task(q: str, kind: str) -> dict:
    tool = "sql_query" if kind == "sql" else "doc_search"
    return task_dict("unanswerable", q, {"type": "unanswerable"}, [tool], [],
                     "The requested information is not available in the company data.", meta={"kind": kind})


# --------------------------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------------------------
def _sample_unique(rng: random.Random, gens: list[Callable[[], dict | None]], n: int, allow_ood: bool,
                   seen: set[str]) -> list[dict]:
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        g = rng.choice(gens)
        t = g()
        if t is None or (t["ood"] and not allow_ood) or t["prompt"] in seen:
            continue
        seen.add(t["prompt"])
        out.append(t)
    return out


def generate_tasks(seed: int = 0, split: str = "test", counts: dict | None = None,
                   exclude_prompts: set[str] | None = None) -> list[Task]:
    """exclude_prompts: prompts that must not be generated (pass the test prompts when generating train, so the
    two splits are disjoint even for templates with a small parameter space such as the SQL ones)."""
    counts = {**DEFAULT_COUNTS, **(counts or {})}
    rng = random.Random(seed)
    allow_ood = split == "test"
    seen: set[str] = set(exclude_prompts or ())
    pool: dict[str, list[dict]] = {}

    pool["calc_single"] = _sample_unique(rng, calc_templates(rng), counts["calc_single"], allow_ood, seen)
    pool["convert_single"] = _sample_unique(rng, convert_templates(rng), counts["convert_single"], allow_ood, seen)
    pool["sql_single"] = _sample_unique(rng, sql_templates(rng), counts["sql_single"], allow_ood, seen)

    # documents: deterministic fact split so train and test never share a fact question
    facts = all_facts()
    ood_facts = [f for f in facts if f["doc_id"] in OOD_DOCS]
    iid_facts = [f for f in facts if f["doc_id"] not in OOD_DOCS]
    test_rng = random.Random(1000)  # fixed, independent of seed: which facts are "test" never changes
    test_rng.shuffle(iid_facts)
    n_test_iid = max(0, DEFAULT_COUNTS["doc_single"] - len(ood_facts))
    test_iid, train_iid = iid_facts[:n_test_iid], iid_facts[n_test_iid:]
    if split == "test":
        chosen = (ood_facts + test_iid)[: counts["doc_single"]]
    else:
        chosen = train_iid[:]
        rng.shuffle(chosen)
        chosen = chosen[: counts["doc_single"]]
    pool["doc_single"] = [doc_task(f) for f in chosen]

    # multi-step: fixed chains are split by parity (test=even, train=odd); OOD ones only in test
    fixed = multi_fixed()
    fixed_split = [t for i, t in enumerate(fixed) if (t["ood"] and split == "test") or (not t["ood"] and (i % 2 == 0) == (split == "test"))]
    n_param = max(0, counts["multi_step"] - len(fixed_split))
    param = _sample_unique(rng, multi_param_templates(rng), n_param, allow_ood, seen)
    multi = fixed_split + param
    rng.shuffle(multi)
    pool["multi_step"] = multi[: counts["multi_step"]]

    qa = NO_TOOL_QA[:]
    random.Random(2000).shuffle(qa)
    half = len(qa) // 2
    qa_split = qa[:half] if split == "test" else qa[half:]
    pool["no_tool"] = [no_tool_task(q, a) for q, a in qa_split[: counts["no_tool"]]]

    un = UNANSWERABLE[:]
    un_split = un[: len(un) // 2] if split == "test" else un[len(un) // 2:]
    pool["unanswerable"] = [unanswerable_task(q, k) for q, k in un_split[: counts["unanswerable"]]]

    tasks: list[Task] = []
    for category, items in pool.items():
        for item in items:
            h = hashlib.sha1(item["prompt"].encode("utf-8")).hexdigest()[:8]
            tasks.append(Task(
                id=f"{split}-{category}-{h}",
                category=category,
                split=split,
                difficulty=len(item["gold_calls"]),
                **item,
            ))
    return tasks


def task_counts(tasks: list[Task]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        out[t.category] = out.get(t.category, 0) + 1
    return out
