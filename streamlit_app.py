import datetime
import math
import statistics
import numpy as np

# Optional: use advanced forecasting if statsmodels is available.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None

def clean_sales_data(sales):
    """Remove outliers from historical sales data using the IQR method."""
    if len(sales) < 2:
        return sales
    q1 = np.percentile(sales, 25)
    q3 = np.percentile(sales, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    cleaned = [x for x in sales if lower_bound <= x <= upper_bound]
    return cleaned

def forecast_sales(sales, current_daily, lead_time):
    """
    Forecast daily sales at the end of the lead time.
    Converts monthly sales to daily by dividing by 30.
    Uses exponential smoothing if possible, otherwise falls back to an exponential growth approach.
    """
    daily_sales = [x / 30.0 for x in sales]
    if ExponentialSmoothing and len(daily_sales) >= 3:
        try:
            model = ExponentialSmoothing(daily_sales, trend='add', seasonal=None, initialization_method="estimated")
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=int(lead_time))
            forecasted_daily = forecast[-1]
            return forecasted_daily
        except Exception:
            pass
    # Fallback: use exponential growth via geometric mean.
    if sales[0] > 0:
        growth_rate = (sales[-1] / sales[0]) ** (1 / (len(sales) - 1)) - 1
    else:
        growth_rate = 0
    return current_daily * ((1 + growth_rate) ** (lead_time / 30.0))

def get_market_common(market_name):
    print(f"\n--- Enter common parameters for the {market_name} market ---")
    try:
        lead_time = float(input("  Lead time (days) until shipment arrives: "))
        buffer_days = float(input("  Buffer days (for safety stock): "))
        ordering_cost = float(input("  Fixed ordering cost for this market: "))
        holding_cost = float(input("  Holding cost per unit per year for this market: "))
    except ValueError:
        print("  Invalid input. Please enter numbers only.")
        return None
    return {"lead_time": lead_time, "buffer_days": buffer_days, "ordering_cost": ordering_cost, "holding_cost": holding_cost}

def get_product_data(product_name, market_name, common):
    include = input(f"\nInclude {product_name} for {market_name}? (default YES, type NO or 0 to skip): ").strip().lower()
    if include in ["no", "0"]:
        return None
    print(f"\n--- Enter details for {product_name} in {market_name} ---")
    try:
        current_stock = float(input("  Current stock (units): "))
        current_daily = float(input("  Current daily sales (units per day): "))
        sales_input = input("  Paste previous months' sales figures (comma separated): ")
        sales_figures = [float(x.strip()) for x in sales_input.split(",") if x.strip() != ""]
        if len(sales_figures) < 2:
            print("  Not enough data provided. Using default values: growth_rate = 0, margin factor = 1.05.")
            monthly_growth_rate = 0.0
            margin_factor = 1.05
        else:
            cleaned_sales = clean_sales_data(sales_figures)
            if len(cleaned_sales) < 2:
                cleaned_sales = sales_figures  # Fallback if cleaning is too strict.
            if cleaned_sales[0] > 0:
                monthly_growth_rate = (cleaned_sales[-1] / cleaned_sales[0]) ** (1 / (len(cleaned_sales) - 1)) - 1
            else:
                monthly_growth_rate = 0.0
            # Dynamic safety margin based on variability: use coefficient of variation.
            mean_sales = statistics.mean(cleaned_sales)
            if mean_sales > 0:
                cv = statistics.stdev(cleaned_sales) / mean_sales
            else:
                cv = 0.0
            margin_factor = min(max(1 + cv, 1.05), 1.2)
    except Exception as e:
        print("  Error in input. Please check your numbers and try again.")
        return None
    return {
        "product": product_name,
        "market": market_name,
        "current_stock": current_stock,
        "current_daily": current_daily,
        "sales_figures": sales_figures,
        "monthly_growth_rate": monthly_growth_rate,
        "margin_factor": margin_factor,
        "lead_time": common["lead_time"],
        "buffer_days": common["buffer_days"],
        "ordering_cost": common["ordering_cost"],
        "holding_cost": common["holding_cost"]
    }

def calculate_metrics(data, today):
    # Clean historical data and forecast sales.
    cleaned_sales = clean_sales_data(data["sales_figures"])
    forecasted_daily = forecast_sales(cleaned_sales, data["current_daily"], data["lead_time"])
    average_daily = (data["current_daily"] + forecasted_daily) / 2
    consumption_during_lead = average_daily * data["lead_time"]
    
    # Dynamic safety stock based on variability:
    daily_sales = [x / 30.0 for x in cleaned_sales] if cleaned_sales else [data["current_daily"]]
    sigma_daily = np.std(daily_sales) if len(daily_sales) > 1 else 0
    z = 1.65  # Service factor for ~95% service level.
    dynamic_safety_stock = z * sigma_daily * math.sqrt(data["lead_time"])
    safety_stock = dynamic_safety_stock * data["margin_factor"]
    
    reorder_point = consumption_during_lead + safety_stock
    days_until_reorder = (data["current_stock"] - reorder_point) / data["current_daily"]
    if days_until_reorder < 0:
        days_until_reorder = 0
    individual_reorder_date = today + datetime.timedelta(days=math.ceil(days_until_reorder))
    coverage_days = data["lead_time"] + data["buffer_days"]
    target_inventory = average_daily * coverage_days + safety_stock
    
    # Cost Optimization: EOQ (Economic Order Quantity)
    annual_demand = average_daily * 365
    if data["holding_cost"] > 0:
        eoq = math.sqrt((2 * annual_demand * data["ordering_cost"]) / data["holding_cost"])
    else:
        eoq = 0

    metrics = {
        "product": data["product"],
        "market": data["market"],
        "current_stock": data["current_stock"],
        "current_daily": data["current_daily"],
        "lead_time": data["lead_time"],
        "buffer_days": data["buffer_days"],
        "forecasted_daily": forecasted_daily,
        "average_daily": average_daily,
        "consumption_during_lead": consumption_during_lead,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "days_until_reorder": days_until_reorder,
        "individual_reorder_date": individual_reorder_date,
        "coverage_days": coverage_days,
        "target_inventory": target_inventory,
        "EOQ": eoq
    }
    return metrics

def compute_order(metrics, common_order_date, today):
    days_to_order = (common_order_date - today).days
    projected_stock_at_order = metrics["current_stock"] - (metrics["current_daily"] * days_to_order)
    consumption_after_order = metrics["average_daily"] * metrics["lead_time"]
    projected_inventory_at_arrival = projected_stock_at_order - consumption_after_order
    recommended_order = metrics["target_inventory"] - projected_inventory_at_arrival
    if recommended_order < 0:
        recommended_order = 0
    arrival_date = common_order_date + datetime.timedelta(days=math.ceil(metrics["lead_time"]))
    run_out_date = arrival_date + datetime.timedelta(days=math.ceil(metrics["coverage_days"]))
    next_order_date = run_out_date - datetime.timedelta(days=math.ceil(metrics["lead_time"]))
    return {
        "recommended_order": math.ceil(recommended_order),
        "arrival_date": arrival_date,
        "run_out_date": run_out_date,
        "next_order_date": next_order_date,
        "individual_reorder_date": metrics["individual_reorder_date"],
        "EOQ": metrics["EOQ"]
    }

def simulate_sensitivity(metrics, param, values, today):
    print(f"\nSensitivity Analysis for {metrics['product']} ({metrics['market']}) on parameter '{param}':")
    for val in values:
        modified = metrics.copy()
        modified[param] = val
        # Recalculate target inventory using modified parameter
        modified_target = modified["average_daily"] * (modified["lead_time"] + modified["buffer_days"]) + modified["safety_stock"]
        rec_order = modified_target - (modified["current_stock"] - modified["current_daily"] * modified["days_until_reorder"])
        print(f"  {param} = {val}: Recommended Order ~ {rec_order:.0f} units")

def main():
    print("Multi-Product, Multi-Market Synchronized Order Planner with Advanced Enhancements")
    print("-----------------------------------------------------------------------------------")
    today = datetime.date.today()
    print(f"Today's Date: {today.strftime('%Y-%m-%d')}")
    
    # Define your product list.
    product_names = [
        "6L Food Waste Bag",
        "10L Food Waste Bag",
        "30L Food Waste Bag",
        "Nappy Sack",
        "Dog Poop Bag"
    ]
    markets = ["UK", "European"]
    market_common = {}
    for market in markets:
        common = get_market_common(market)
        if common is None:
            print(f"Skipping {market} market due to invalid inputs.")
            continue
        market_common[market] = common
    
    all_metrics = []
    for market in markets:
        if market not in market_common:
            continue
        for product in product_names:
            data = get_product_data(product, market, market_common[market])
            if data is None:
                continue
            metrics = calculate_metrics(data, today)
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid product data entered. Exiting.")
        return
    
    # Determine the common (synchronized) order date across all products/markets.
    individual_dates = [m["individual_reorder_date"] for m in all_metrics]
    common_order_date = min(individual_dates)
    
    orders = {}
    total_order = 0
    for metrics in all_metrics:
        order_info = compute_order(metrics, common_order_date, today)
        key = f"{metrics['product']} ({metrics['market']})"
        orders[key] = order_info
        total_order += order_info["recommended_order"]
    
    print("\n--- Synchronized Order Details ---")
    print(f"Common Order Date (place one big shipment by this date): {common_order_date.strftime('%Y-%m-%d')}\n")
    for key, order_info in orders.items():
        print(f"Product & Market: {key}")
        print(f"  Individual Reorder Date: {order_info['individual_reorder_date'].strftime('%Y-%m-%d')}")
        print(f"  Recommended Order Quantity: {order_info['recommended_order']} units")
        print(f"  Expected Shipment Arrival Date: {order_info['arrival_date'].strftime('%Y-%m-%d')}")
        print(f"  New Shipment Run Out Date: {order_info['run_out_date'].strftime('%Y-%m-%d')}")
        print(f"  Next Recommended Order Date: {order_info['next_order_date'].strftime('%Y-%m-%d')}")
        print(f"  EOQ for cost optimization: {order_info['EOQ']:.0f} units\n")
    print(f"Total Order Quantity for Manufacturer (across all products & markets): {total_order} units")
    
    # Optional: Run a simple sensitivity analysis (e.g., varying lead time by ±10%).
    for metrics in all_metrics:
        simulate_sensitivity(metrics, "lead_time", [metrics["lead_time"], metrics["lead_time"] * 1.1, metrics["lead_time"] * 0.9], today)

if __name__ == "__main__":
    main()
