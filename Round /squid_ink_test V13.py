import json
from typing import Dict, List, Any
import jsonpickle
import numpy as np
import math
import traceback
import pandas as pd

# Import necessary classes from datamodel
from datamodel import (
    OrderDepth, UserId, TradingState, Order, Trade, Symbol, ProsperityEncoder,
    Listing, Observation, Product, Position, Time, ConversionObservation
)

# ----------------------------------------------------------------------
# --- PASTE THE FULL, ORIGINAL LOGGER CLASS (with .strip() fix) HERE ---
# Ensure ALL original compress_* methods are included from the prompt.
class Logger:
    # ... (Full Logger class implementation including all compress methods) ...
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        log_line = sep.join(map(str, objects)) + end
        if len(self.logs) + len(log_line) < self.max_log_length * 3: self.logs += log_line
    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        try:
            # Ensure getattr defaults work or use actual methods if available
            _compress_state = getattr(self, 'compress_state', lambda s, td: [])
            _compress_orders = getattr(self, 'compress_orders', lambda o: [])
            base_payload = [ _compress_state(state, ""), _compress_orders({}), conversions, "", "" ]
            base_length = len(self.to_json(base_payload))
        except Exception as e: base_length = 500
        max_item_length = (self.max_log_length - base_length) // 3
        if max_item_length < 0: max_item_length = 0
        processed_logs = self.logs.replace('\n', ' | ').strip()
        truncated_state_trader_data = self.truncate(state.traderData if state.traderData is not None else "", max_item_length)
        truncated_trader_data = self.truncate(trader_data if trader_data is not None else "", max_item_length)
        truncated_logs = self.truncate(processed_logs, max_item_length)
        try:
            # Use actual methods if defined, otherwise defaults might cause format errors
            log_payload = [ self.compress_state(state, truncated_state_trader_data), self.compress_orders(orders), conversions, truncated_trader_data, truncated_logs, ]
            print(self.to_json(log_payload))
        except Exception as e: print(f"Error during logging/flushing: {e}")
        finally: self.logs = ""
    # --- INCLUDE ALL ORIGINAL compress_* methods here ---
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Ensure all necessary compress sub-methods are called correctly
        # Make sure self has these methods defined exactly as in prompt
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]
    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if listings: # Check if listings is not None or empty
            for listing in listings.values():
                compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed
    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths: # Check if not None or empty
            for symbol, order_depth in order_depths.items():
                # Ensure keys exist before accessing using getattr with default
                buy_orders_data = getattr(order_depth, 'buy_orders', {})
                sell_orders_data = getattr(order_depth, 'sell_orders', {})
                compressed[symbol] = [buy_orders_data, sell_orders_data]
        return compressed
    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades: # Check if not None or empty
            for arr in trades.values():
                if arr: # Check if list is not empty
                    for trade in arr:
                        # Ensure all attributes exist using getattr with default
                        buyer = getattr(trade, 'buyer', None)
                        seller = getattr(trade, 'seller', None)
                        timestamp = getattr(trade, 'timestamp', 0)
                        compressed.append([
                            trade.symbol, trade.price, trade.quantity,
                            buyer, seller, timestamp
                        ])
        return compressed
    def compress_observations(self, observations: Observation) -> list[Any]:
        # Ensure observations object and its attributes exist using getattr
        plain_obs = {}
        conv_obs = {}
        if observations:
            plain_obs = getattr(observations, 'plainValueObservations', {})
            conv_obs_data = getattr(observations, 'conversionObservations', {})
            if conv_obs_data:
                 for product, observation in conv_obs_data.items():
                     # Ensure sub-object attributes exist using getattr with default
                     conv_obs[product] = [
                         getattr(observation, attr, None) for attr in [
                             'bidPrice', 'askPrice', 'transportFees', 'exportTariff',
                             'importTariff', 'sugarPrice', 'sunlightIndex'
                         ]
                     ]
        return [plain_obs, conv_obs]
    def compress_orders(self, orders: dict[Symbol, List[Order]]) -> list[list[Any]]:
        compressed = []
        if orders: # Check if not None or empty
            for arr in orders.values():
                 if arr: # Check if list is not empty
                     for order in arr:
                         # Ensure price is int if required by backend spec, float might be used internally
                         compressed.append([order.symbol, int(order.price), order.quantity])
        return compressed
    def to_json(self, value: Any) -> str:
        # Use the provided ProsperityEncoder
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))
    def truncate(self, value: str, max_length: int) -> str:
        # Ensure value is a string for length check and slicing
        if not isinstance(value, str): value = str(value)
        if len(value) <= max_length: return value
        return value[: max_length - 3] + "..."

logger = Logger()
# ----------------------------------------------------------------------


SQUID_INK = "SQUID_INK"

# --- Strategy Parameters V13: Tighter Inventory Control ---
FAIR_VALUE_EMA_PERIOD = 10
QUOTE_OFFSET = 2
ASYMMETRIC_FACTOR = 0.5
INVENTORY_SKEW_FACTOR = 0.2  # Increased skew
POSITION_LIMIT = 50
ORDER_SIZE = 10
MAX_POSITION_THRESHOLD = 30  # Tighter "hard lock" liquidation threshold
# ****************************

class Trader:
    """
    Market Making Strategy V13: Tighter inventory control via increased skew
    and lower liquidation threshold.
    """
    def __init__(self):
        self.position_limit = {SQUID_INK: POSITION_LIMIT}
        self.trader_state = {} # {product: {'history': [], 'last_ema': None}}

    # --- calculation methods (calculate_mid_price, get_ema) ---
    def calculate_mid_price(self, order_depth: OrderDepth) -> float | None:
        # ... (same as V12) ...
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders: return None
        try: best_ask = min(order_depth.sell_orders.keys()); best_bid = max(order_depth.buy_orders.keys())
        except ValueError: return None
        if best_ask <= best_bid: return None
        return (best_ask + best_bid) / 2

    def get_ema(self, prices: List[float], period: int) -> float | None:
        # ... (same as V12) ...
        if len(prices) < period: return None
        try:
            series = pd.Series(prices); ema = series.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            return ema if not pd.isna(ema) else None
        except Exception as e: logger.print(f"Error EMA: {e}"); return None
    # --- End calculation methods ---

    def squid_ink_strategy(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = SQUID_INK
        limit = self.position_limit[product]

        order_depth = state.order_depths.get(product)
        current_pos = state.position.get(product, 0)

        # --- Get Product State ---
        prod_state = self.trader_state.setdefault(product, {'history': [], 'last_ema': None})
        history = prod_state['history']
        last_ema = prod_state.get('last_ema')

        # --- Price and History Update ---
        current_mid_price = self.calculate_mid_price(order_depth)
        best_ask: int | None = None; best_bid: int | None = None; spread: int | None = None
        if order_depth:
            if order_depth.sell_orders: 
                try: best_ask = min(order_depth.sell_orders.keys())
                except ValueError: best_ask = None
            if order_depth.buy_orders: 
                try: best_bid = max(order_depth.buy_orders.keys())
                except ValueError: best_bid = None
            if best_ask is not None and best_bid is not None and best_ask > best_bid: spread = best_ask - best_bid

        if current_mid_price is None: logger.print(f"{product}: Mid price fail. Skip."); return orders
        history.append(current_mid_price); max_hist_len = FAIR_VALUE_EMA_PERIOD * 3
        if len(history) > max_hist_len: history = history[-max_hist_len:]
        prod_state['history'] = history

        # --- Calculate Fair Value (EMA) & Trend Direction ---
        if len(history) < FAIR_VALUE_EMA_PERIOD: logger.print(f"{product}: Data ({len(history)}/{FAIR_VALUE_EMA_PERIOD})"); return orders
        fair_value_ema = self.get_ema(history, FAIR_VALUE_EMA_PERIOD)
        if fair_value_ema is None: logger.print(f"{product}: EMA calc fail."); return orders

        ema_trend_direction = 0 # 0: Flat/Unknown, 1: Up, -1: Down
        if last_ema is not None:
            # Add a small threshold to define trend vs flat
            ema_diff_threshold = 0.1
            if fair_value_ema > last_ema + ema_diff_threshold : ema_trend_direction = 1
            elif fair_value_ema < last_ema - ema_diff_threshold: ema_trend_direction = -1
        prod_state['last_ema'] = fair_value_ema # Store current EMA for next tick

        # --- Logging ---
        mid_str=f"{current_mid_price:.1f}"; bbo_str=f"{best_bid}/{best_ask}" if best_bid and best_ask else "N/A"
        ema_str=f"{fair_value_ema:.1f}"; trend_str = ["DOWN", "FLAT", "UP"][ema_trend_direction + 1]
        logger.print(f"{product}@{state.timestamp}: Pos={current_pos}, Mid={mid_str}, EMA={ema_str}({trend_str}), BBO={bbo_str}")

        # --- Market Making Logic ---

        # ** Liquidation Logic ** (Triggering at 30 now)
        if abs(current_pos) >= MAX_POSITION_THRESHOLD:
            log_reason = f"Pos Limit Breach ({current_pos}>={MAX_POSITION_THRESHOLD})"
            logger.print(f"{product}: EXIT ({log_reason}). Liquidating.")
            exit_qty = -current_pos; exit_price = None
            if exit_qty > 0 and best_ask is not None: exit_price = best_ask # Need to buy
            elif exit_qty < 0 and best_bid is not None: exit_price = best_bid # Need to sell
            if exit_price is not None: orders.append(Order(product, int(exit_price), exit_qty)); logger.print(f"{product}: Liq Order: Qty={exit_qty}, Px={exit_price}")
            else: logger.print(f"{product}: Cannot liquidate - missing BBO")
            self.trader_state[product] = prod_state # Save history, last_ema
            return orders

        # ** Normal Quoting Logic **
        inventory_skew = current_pos * INVENTORY_SKEW_FACTOR # Increased skew factor

        # Asymmetric adjustment based on EMA trend
        bid_trend_adj = 0; ask_trend_adj = 0
        if ema_trend_direction == 1: # EMA Rising
            bid_trend_adj = ASYMMETRIC_FACTOR; ask_trend_adj = -ASYMMETRIC_FACTOR
        elif ema_trend_direction == -1: # EMA Falling
            bid_trend_adj = -ASYMMETRIC_FACTOR; ask_trend_adj = ASYMMETRIC_FACTOR

        buy_offset = QUOTE_OFFSET + bid_trend_adj
        sell_offset = QUOTE_OFFSET + ask_trend_adj
        target_buy_price_adjusted = fair_value_ema - buy_offset - inventory_skew
        target_sell_price_adjusted = fair_value_ema + sell_offset - inventory_skew
        final_buy_price = math.floor(target_buy_price_adjusted)
        final_sell_price = math.ceil(target_sell_price_adjusted)

        # Ensure minimum spread between our own quotes
        min_quote_spread = 1
        if final_sell_price - final_buy_price < min_quote_spread:
            adjusted_mid = (target_buy_price_adjusted + target_sell_price_adjusted) / 2.0
            final_buy_price = math.floor(adjusted_mid - min_quote_spread / 2.0)
            final_sell_price = math.ceil(adjusted_mid + min_quote_spread / 2.0)
            # logger.print(f"{product}: Quotes too close/crossed. Reset: Buy={final_buy_price}, Sell={final_sell_price}")

        # --- Place Orders ---
        available_buy_capacity = limit - current_pos
        available_sell_capacity = limit + current_pos

        # Place Buy Order
        if available_buy_capacity > 0 and final_buy_price > 0 :
             buy_order_qty = min(ORDER_SIZE, available_buy_capacity)
             # Don't place buy strictly above best ask
             if best_ask is None or final_buy_price < best_ask:
                 logger.print(f"{product}: Place BUY LIMIT: Qty={buy_order_qty}, Px={final_buy_price} (EMA={ema_str}, Skew={inventory_skew:.2f}, TrendAdj={bid_trend_adj:.1f})")
                 orders.append(Order(product, final_buy_price, buy_order_qty))
             # else: logger.print(f"{product}: Skip BUY LIMIT {final_buy_price} >= Best Ask {best_ask}")

        # Place Sell Order
        if available_sell_capacity > 0 and final_sell_price > 0 :
             sell_order_qty = -min(ORDER_SIZE, available_sell_capacity)
             # Don't place sell strictly below best bid
             if best_bid is None or final_sell_price > best_bid:
                 logger.print(f"{product}: Place SELL LIMIT: Qty={sell_order_qty}, Px={final_sell_price} (EMA={ema_str}, Skew={inventory_skew:.2f}, TrendAdj={ask_trend_adj:.1f})")
                 orders.append(Order(product, final_sell_price, sell_order_qty))
             # else: logger.print(f"{product}: Skip SELL LIMIT {final_sell_price} <= Best Bid {best_bid}")

        # --- Save State ---
        self.trader_state[product] = prod_state

        return orders

    def run(self, state: TradingState):
        """ Main entry point """
        result: Dict[Symbol, List[Order]] = {}
        # --- Load Trader State ---
        loaded_state = {}
        if state.traderData:
            try:
                decoded_data = jsonpickle.decode(state.traderData, keys=True)
                if isinstance(decoded_data, dict): loaded_state = decoded_data
            except Exception as e: logger.print(f"Error decode traderData: {e}")
        self.trader_state = loaded_state

        # --- Execute Strategy ---
        try:
            self.trader_state.setdefault(SQUID_INK, {'history': [], 'last_ema': None}) # Ensure key exists
            squid_ink_orders = self.squid_ink_strategy(state)
            result[SQUID_INK] = squid_ink_orders
        except Exception as e:
            logger.print(f"Error in strategy: {e}"); logger.print(traceback.format_exc())
            result[SQUID_INK] = []

        # --- Save Trader State ---
        try: traderData = jsonpickle.encode(self.trader_state, keys=True)
        except Exception as e: logger.print(f"Error encode traderData: {e}"); traderData = ""

        conversions = 0
        # --- Flush Logs ---
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
