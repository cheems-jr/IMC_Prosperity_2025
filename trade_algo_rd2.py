from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Any, Dict, List
import statistics
import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ------------------------------LOGGER DONE-----------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

from math import log, sqrt, exp
from statistics import NormalDist
class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"
VOLCANIC_ROCK = 'VOLCANIC_ROCK'
VOLCANIC_ROCK_VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
VOLCANIC_ROCK_VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
VOLCANIC_ROCK_VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
VOLCANIC_ROCK_VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
VOLCANIC_ROCK_VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'

PRODUCTS = [
    RAINFOREST_RESIN,
    KELP,
    SQUID_INK,
    CROISSANTS,
    JAMS,
    DJEMBES,
    PICNIC_BASKET1,
    PICNIC_BASKET2,
    VOLCANIC_ROCK,
    VOLCANIC_ROCK_VOUCHER_9500,
    VOLCANIC_ROCK_VOUCHER_9750,
    VOLCANIC_ROCK_VOUCHER_10000,
    VOLCANIC_ROCK_VOUCHER_10250,
    VOLCANIC_ROCK_VOUCHER_10500
]

DEFAULT_PRICES = {
    RAINFOREST_RESIN: 10000,
    KELP: 2020,
    SQUID_INK: 2000,
    CROISSANTS: 4297,
    JAMS: 6593,
    DJEMBES: 13436,
    PICNIC_BASKET1: 59051,
    PICNIC_BASKET2: 30408,
    VOLCANIC_ROCK: 10503,
    VOLCANIC_ROCK_VOUCHER_9500: 1003.5,
    VOLCANIC_ROCK_VOUCHER_9750: 754.5,
    VOLCANIC_ROCK_VOUCHER_10000: 505.5,
    VOLCANIC_ROCK_VOUCHER_10250: 273.5,
    VOLCANIC_ROCK_VOUCHER_10500: 99.5,
}

lookback_std_dev_pb1 = 10
lookback_ema_pb1 = 20
lookback_std_dev_pb2 = 10
lookback_ema_pb2 = 20
lookback_std_dev_bb = 10
lookback_ema_bb = 20
z_threshold = 7
z_threshold_2 = 5
z_threshold_bb = 5
spread_mean = 49
spread_mean_p2 = 25
spread_mean_bb = 25

SQUID_INK = "SQUID_INK"

# --- Strategy Parameters V12: Adaptive Market Making ---
FAIR_VALUE_EMA_PERIOD = 10   # Smoother EMA
QUOTE_OFFSET = 2         # Wider base offset
ASYMMETRIC_FACTOR = 0.5    # How much to adjust quotes based on EMA slope (in ticks)
INVENTORY_SKEW_FACTOR = 0.15 # Keep skew moderate
POSITION_LIMIT = 50
ORDER_SIZE = 10          # Increased order size
MAX_POSITION_THRESHOLD = 40 # Relaxed liquidation threshold

class Trader:
    #@TODO: instead of VWAP, try using EMA for KELP's fair price -> calculate the EMA based on P_t and EMA_{t-1}
    #this requires passing the current EMA to the next step via traderdata at the end (compare to VWAP)
    #put bid orders at EMA-1 and ask orders at EMA+1
    def __init__(self):
        #self.kelp_prices = []
        #self.kelp_vwap = []
        
        self.vwap_history = []

        self.ema_prices = dict()
        for product in PRODUCTS[1:]:
            self.ema_prices[product] = {
                'price_history' : [],
                'ema_20' : None,
                'ema_5' : None
            }
        
        self.pb1_ema = []
        self.pb1_diffs = []
        #initialize std dev at high value s.t. initial timestamp does not lead to uninformed trades
        self.pb1_std_dev = 100
            
        self.pb2_ema = []
        self.pb2_diffs= []
        self.pb2_std_dev = 100

        self.bb_ema = []
        self.bb_diffs= []
        self.bb_std_dev = 100

        self.ema_param = 0.5

        self.base_iv_history = []  # Track base IV over time
        self.vol_smile_coeffs = {}  # Store parabola coefficients per timestamp
        self.residual_threshold = 0.005  # 3% IV deviation from curve
        self.base_iv_window = 20  # For regime detection
        self.voucher_coeffs = {}

        self.position_limit = {
            RAINFOREST_RESIN : 50,
            KELP : 50,
            SQUID_INK: 50,
            CROISSANTS: 250,
            JAMS: 350,
            DJEMBES: 60,
            PICNIC_BASKET1: 60,
            PICNIC_BASKET2: 100,
            VOLCANIC_ROCK: 400,
            VOLCANIC_ROCK_VOUCHER_9500: 200,
            VOLCANIC_ROCK_VOUCHER_9750: 200,
            VOLCANIC_ROCK_VOUCHER_10000: 200,
            VOLCANIC_ROCK_VOUCHER_10250: 200,
            VOLCANIC_ROCK_VOUCHER_10500: 200,

        }

        self.vouchers = [
            VOLCANIC_ROCK_VOUCHER_9500,
            VOLCANIC_ROCK_VOUCHER_9750,
            VOLCANIC_ROCK_VOUCHER_10000,
            VOLCANIC_ROCK_VOUCHER_10250,
            VOLCANIC_ROCK_VOUCHER_10500,
        ]

        self.strike = {
            VOLCANIC_ROCK_VOUCHER_9500: 9500,
            VOLCANIC_ROCK_VOUCHER_9750: 9750,
            VOLCANIC_ROCK_VOUCHER_10000: 10000,
            VOLCANIC_ROCK_VOUCHER_10250: 10250,
            VOLCANIC_ROCK_VOUCHER_10500: 10500,
        }
        self.voucher_z_threshold = {
            VOLCANIC_ROCK_VOUCHER_9500: 5,
            VOLCANIC_ROCK_VOUCHER_9750: 5,
            VOLCANIC_ROCK_VOUCHER_10000: 5,
            VOLCANIC_ROCK_VOUCHER_10250: 5,
            VOLCANIC_ROCK_VOUCHER_10500: 5,
        }
        self.mean_volatility = {
            VOLCANIC_ROCK_VOUCHER_9500: 0.1814496610488287,
            VOLCANIC_ROCK_VOUCHER_9750: 0.22107020961926602,
            VOLCANIC_ROCK_VOUCHER_10000: 0.2103950534654465,
            VOLCANIC_ROCK_VOUCHER_10250: 0.19169856564902132,
            VOLCANIC_ROCK_VOUCHER_10500: 0.18728347050089258,
        }

        self.volatility_tracker = []

        self.vol_window = 5

        self.atr_tracker = dict()
        for product in PRODUCTS[1:]:
            self.atr_tracker[product] = {
                'tr_history': [],
                'atr': None,
                'prev_close': None
            }

    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0) 
    
    def update_vwap(self, trades: List[Trade]):
        if not trades:
            return None

        total_qty = sum([t.quantity for t in trades])
        total_value = sum([t.price * t.quantity for t in trades])
        vwap = total_value / total_qty if total_qty > 0 else None

        if vwap:
            self.vwap_history.append(vwap)
            if len(self.vwap_history) > 100:
                self.vwap_history.pop(0)

        return vwap

    def calculate_volatility(self):
        if len(self.vwap_history) < 2:
            return 0
        return statistics.stdev(self.vwap_history)

    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2

    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        # mm_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 50])
        # mm_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 50])
        if [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]:
            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        else:
            baaf = 0
        if [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]:
            bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        else: bbbf = 0
        

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1)
        if baaf and bbbf:
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order

        return orders
    
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
        
    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential movitng average of the prices of each product.
        """
        for product in PRODUCTS[1:3]:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue
            
            alpha_5 = 2 / (5 + 1)
            alpha_20 = 2 / (20 + 1)

            if self.ema_prices[product]['ema_5'] is None:
                self.ema_prices[product]['ema_5'] = mid_price
            else:
                self.ema_prices[product]['ema_5'] = (alpha_5 * mid_price + 
                                                (1 - alpha_5) * self.ema_prices[product]['ema_5'])

            # Update 20-period EMA
            if self.ema_prices[product]['ema_20'] is None:
                self.ema_prices[product]['ema_20'] = mid_price
            else:
                self.ema_prices[product]['ema_20'] = (alpha_20 * mid_price + 
                                                    (1 - alpha_20) * self.ema_prices[product]['ema_20'])
                

    def kelp_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of kelp.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_kelp = self.get_position(KELP, state)

        bid_volume = self.position_limit[KELP] - position_kelp
        ask_volume = - self.position_limit[KELP] - position_kelp

        orders = []

        if position_kelp == 0:
            # Not long nor short
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP]['ema_20'] - 1), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP]['ema_20'] + 1), ask_volume))
        
        if position_kelp > 0:
            # Long position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP]['ema_20'] - 2), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP]['ema_20']), ask_volume))

        if position_kelp < 0:
            # Short position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP]['ema_20']), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP]['ema_20'] + 2), ask_volume))

        return orders
    
        # --- calculation methods ---
    def calculate_mid_price(self, order_depth: OrderDepth) -> float | None:
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders: return None
        try: best_ask = min(order_depth.sell_orders.keys()); best_bid = max(order_depth.buy_orders.keys())
        except ValueError: return None
        if best_ask <= best_bid: return None
        return (best_ask + best_bid) / 2

    def get_ema(self, prices: List[float], period: int) -> float | None:
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
        last_ema = prod_state.get('last_ema') # Get EMA from previous tick

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
            if fair_value_ema > last_ema: ema_trend_direction = 1
            elif fair_value_ema < last_ema: ema_trend_direction = -1
        prod_state['last_ema'] = fair_value_ema # Store current EMA for next tick

        # --- Logging ---
        mid_str=f"{current_mid_price:.1f}"; bbo_str=f"{best_bid}/{best_ask}" if best_bid and best_ask else "N/A"
        ema_str=f"{fair_value_ema:.1f}"; trend_str = ["DOWN", "FLAT", "UP"][ema_trend_direction + 1]
        logger.print(f"{product}@{state.timestamp}: Pos={current_pos}, Mid={mid_str}, EMA={ema_str}({trend_str}), BBO={bbo_str}")

        # --- Market Making Logic ---

        # ** Liquidation Logic **
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
        inventory_skew = current_pos * INVENTORY_SKEW_FACTOR

        # Asymmetric adjustment based on EMA trend
        bid_trend_adj = 0; ask_trend_adj = 0
        if ema_trend_direction == 1: # EMA Rising -> More aggressive bid, less aggressive ask
            bid_trend_adj = ASYMMETRIC_FACTOR
            ask_trend_adj = -ASYMMETRIC_FACTOR
        elif ema_trend_direction == -1: # EMA Falling -> Less aggressive bid, more aggressive ask
            bid_trend_adj = -ASYMMETRIC_FACTOR
            ask_trend_adj = ASYMMETRIC_FACTOR

        # Calculate base offsets
        buy_offset = QUOTE_OFFSET + bid_trend_adj
        sell_offset = QUOTE_OFFSET + ask_trend_adj

        # Calculate target prices with skew AND asymmetry
        target_buy_price_adjusted = fair_value_ema - buy_offset - inventory_skew
        target_sell_price_adjusted = fair_value_ema + sell_offset - inventory_skew # Skew subtracts from both

        # Round to nearest tick
        final_buy_price = math.floor(target_buy_price_adjusted)
        final_sell_price = math.ceil(target_sell_price_adjusted)

        # Ensure minimum spread between our own quotes
        min_quote_spread = 1 # Minimum allowed spread
        if final_sell_price - final_buy_price < min_quote_spread:
            # If crossed or too close, widen symmetrically around the adjusted midpoint
            adjusted_mid = (target_buy_price_adjusted + target_sell_price_adjusted) / 2.0
            final_buy_price = math.floor(adjusted_mid - min_quote_spread / 2.0)
            final_sell_price = math.ceil(adjusted_mid + min_quote_spread / 2.0)
            logger.print(f"{product}: Quotes too close/crossed. Reset around AdjMid {adjusted_mid:.2f}: Buy={final_buy_price}, Sell={final_sell_price}")

        # --- Place Orders ---
        available_buy_capacity = limit - current_pos
        available_sell_capacity = limit + current_pos

        # Place Buy Order
        if available_buy_capacity > 0 and final_buy_price > 0 :
             buy_order_qty = min(ORDER_SIZE, available_buy_capacity)
             # Simplified check: Don't place buy strictly above best ask
             if best_ask is None or final_buy_price < best_ask:
                 logger.print(f"{product}: Place BUY LIMIT: Qty={buy_order_qty}, Px={final_buy_price} (EMA={ema_str}, Skew={inventory_skew:.2f}, TrendAdj={bid_trend_adj:.1f})")
                 orders.append(Order(product, final_buy_price, buy_order_qty))
             else: logger.print(f"{product}: Skip BUY LIMIT {final_buy_price} >= Best Ask {best_ask}")

        # Place Sell Order
        if available_sell_capacity > 0 and final_sell_price > 0 :
             sell_order_qty = -min(ORDER_SIZE, available_sell_capacity)
             # Simplified check: Don't place sell strictly below best bid
             if best_bid is None or final_sell_price > best_bid:
                 logger.print(f"{product}: Place SELL LIMIT: Qty={sell_order_qty}, Px={final_sell_price} (EMA={ema_str}, Skew={inventory_skew:.2f}, TrendAdj={ask_trend_adj:.1f})")
                 orders.append(Order(product, final_sell_price, sell_order_qty))
             else: logger.print(f"{product}: Skip SELL LIMIT {final_sell_price} <= Best Bid {best_bid}")

        # --- Save State ---
        self.trader_state[product] = prod_state # Save updated state

        return orders
    
    def get_pb1_diff(self, state: TradingState):
        mid_price_pb1 = self.get_mid_price(PICNIC_BASKET1, state)
        mid_price_indiv = 6*self.get_mid_price(CROISSANTS, state) + 3*self.get_mid_price(JAMS, state) + self.get_mid_price(DJEMBES, state)

        return mid_price_pb1 - mid_price_indiv
    
    def update_pb1_params(self, state: TradingState):

        alpha_pb1 = 2 / (lookback_ema_pb1 + 1)

        self.pb1_diffs.append(self.get_pb1_diff(state))

        if self.pb1_ema == []:
            self.pb1_ema.append(self.pb1_diffs[-1])

        else:
            self.pb1_ema.append(alpha_pb1 * self.pb1_diffs[-1] + (1-alpha_pb1) * self.pb1_ema[-1])
        
        if len(self.pb1_diffs) > lookback_std_dev_pb1:
            self.pb1_diffs.pop(0)

        if len(self.pb1_ema) > lookback_ema_pb1:
            self.pb1_ema.pop(0)
        
        if len(self.pb1_ema) >= 2 and len(self.pb1_diffs) >= 2:
            self.pb1_std_dev = statistics.stdev(np.array(self.pb1_diffs, dtype=float) - np.array(self.pb1_ema, dtype=float)[-10:])

    def basket_1_strategy(self, state: TradingState):
        
        orders: List[Order] = []

        self.update_pb1_params(state)
        
        spread = self.get_pb1_diff(state)

        order_depth_pb1 = state.order_depths.get(PICNIC_BASKET1, OrderDepth({}, {}))
        best_bid_pb1 = max(order_depth_pb1.buy_orders.keys()) if order_depth_pb1.buy_orders else None
        best_ask_pb1 = min(order_depth_pb1.sell_orders.keys()) if order_depth_pb1.sell_orders else None

        order_depth_cr = state.order_depths.get(CROISSANTS, OrderDepth({}, {}))
        best_bid_cr = max(order_depth_cr.buy_orders.keys()) if order_depth_cr.buy_orders else None
        best_ask_cr = min(order_depth_cr.sell_orders.keys()) if order_depth_cr.sell_orders else None

        order_depth_jams = state.order_depths.get(JAMS, OrderDepth({}, {}))
        best_bid_jams = max(order_depth_jams.buy_orders.keys()) if order_depth_jams.buy_orders else None
        best_ask_jams = min(order_depth_jams.sell_orders.keys()) if order_depth_jams.sell_orders else None

        order_depth_dje = state.order_depths.get(DJEMBES, OrderDepth({}, {}))
        best_bid_dje = max(order_depth_dje.buy_orders.keys()) if order_depth_dje.buy_orders else None
        best_ask_dje = min(order_depth_dje.sell_orders.keys()) if order_depth_dje.sell_orders else None

        z_score = (spread - spread_mean) / self.pb1_std_dev


        #basket overvalued -> short basket, long on individuals
        #if self.pb1_std_dev is not None and (diff_actual - self.pb1_ema[-1]) > 4.5 * self.pb1_std_dev:
        if z_score >= z_threshold:
            logger.print(f"Basket short-regime, diff-EMA {spread}-{self.pb1_ema[-1]} > 4.5 * {self.pb1_std_dev}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(-math.floor(order_depth_cr.sell_orders[best_ask_cr]/6), -math.floor(order_depth_jams.sell_orders[best_ask_jams]/3), -order_depth_dje.sell_orders[best_ask_dje], order_depth_pb1.buy_orders[best_bid_pb1])
            logger.print(f"Initial pkg_amt_indiv: {pkg_amt_indiv}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) > self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if (6*pkg_amt_indiv + self.get_position(CROISSANTS, state)) > self.position_limit[CROISSANTS]:
                pkg_amt_indiv = math.floor((self.position_limit[CROISSANTS] - self.get_position(CROISSANTS, state))/6)
            if (3*pkg_amt_indiv + self.get_position(JAMS, state)) > self.position_limit[JAMS]:
                pkg_amt_indiv = math.floor((self.position_limit[JAMS] - self.get_position(JAMS, state))/3)
            if (pkg_amt_indiv + self.get_position(DJEMBES, state)) > self.position_limit[DJEMBES]:
                pkg_amt_indiv = math.floor(self.position_limit[DJEMBES] - self.get_position(DJEMBES, state))
            logger.print(f"Final pkg_amt_indiv: {pkg_amt_indiv}")

            orders.append(Order(PICNIC_BASKET1, best_bid_pb1, -pkg_amt_indiv))
            logger.print(f"Going short on PB1 with order {orders[-1]}")
            '''orders.append(Order(CROISSANTS, best_ask_cr, 6*pkg_amt_indiv))
            logger.print(f"Going long on CR with order {orders[-1]}")
            orders.append(Order(JAMS, best_ask_jams, 3*pkg_amt_indiv))
            logger.print(f"Going long on JAMS with order {orders[-1]}")
            orders.append(Order(DJEMBES, best_ask_dje, pkg_amt_indiv))
            logger.print(f"Going long on DJE with order {orders[-1]}")'''

        #if self.pb1_std_dev is not None and (diff_actual - self.pb1_ema[-1]) < -4.5 * self.pb1_std_dev:
        if z_score <= z_threshold:
            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(math.floor(order_depth_cr.buy_orders[best_bid_cr]/6), math.floor(order_depth_jams.buy_orders[best_bid_jams]/3), order_depth_dje.buy_orders[best_bid_dje], -order_depth_pb1.sell_orders[best_ask_pb1])
            logger.print(f"Initial pkg_amt_indiv: {pkg_amt_indiv}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (-pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) < -self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if -6*pkg_amt_indiv + self.get_position(CROISSANTS, state) < -self.position_limit[CROISSANTS]:
                pkg_amt_indiv = math.floor((self.position_limit[CROISSANTS] - self.get_position(CROISSANTS, state))/6)
            logger.print(f"pkg_amt_indiv after if-1: {pkg_amt_indiv}")
            if -3*pkg_amt_indiv + self.get_position(JAMS, state) < -self.position_limit[JAMS]:
                pkg_amt_indiv = math.floor((self.position_limit[JAMS] - self.get_position(JAMS, state))/3)
            logger.print(f"pkg_amt_indiv after if-2: {pkg_amt_indiv}")
            if -pkg_amt_indiv + self.get_position(DJEMBES, state) < -self.position_limit[DJEMBES]:
                pkg_amt_indiv = math.floor(self.position_limit[DJEMBES] - self.get_position(DJEMBES, state))
            logger.print(f"Final pkg_amt_indiv: {pkg_amt_indiv}")

            orders.append(Order(PICNIC_BASKET1, best_ask_pb1, pkg_amt_indiv))
            logger.print(f"Going long on PB1 with order {orders[-1]}")
            '''orders.append(Order(CROISSANTS, best_bid_cr, -6*pkg_amt_indiv))
            logger.print(f"Going short on CR with order {orders[-1]}")
            orders.append(Order(JAMS, best_bid_cr, -3*pkg_amt_indiv))
            logger.print(f"Going short on JAMS with order {orders[-1]}")
            orders.append(Order(DJEMBES, best_bid_cr, -pkg_amt_indiv))
            logger.print(f"Going short on DJE with order {orders[-1]}")'''  

        logger.print(f"Returning orders: {orders}") 
        return orders

    def basket_2_strategy(self, state: TradingState):
        
        orders: List[Order] = []

        self.update_pb2_params(state)
        
        spread = self.get_pb2_diff(state)

        order_depth_pb2 = state.order_depths.get(PICNIC_BASKET2, OrderDepth({}, {}))
        best_bid_pb2 = max(order_depth_pb2.buy_orders.keys()) if order_depth_pb2.buy_orders else None
        best_ask_pb2 = min(order_depth_pb2.sell_orders.keys()) if order_depth_pb2.sell_orders else None

        order_depth_cr = state.order_depths.get(CROISSANTS, OrderDepth({}, {}))
        best_bid_cr = max(order_depth_cr.buy_orders.keys()) if order_depth_cr.buy_orders else None
        best_ask_cr = min(order_depth_cr.sell_orders.keys()) if order_depth_cr.sell_orders else None

        order_depth_jams = state.order_depths.get(JAMS, OrderDepth({}, {}))
        best_bid_jams = max(order_depth_jams.buy_orders.keys()) if order_depth_jams.buy_orders else None
        best_ask_jams = min(order_depth_jams.sell_orders.keys()) if order_depth_jams.sell_orders else None

        if self.pb2_std_dev == 0:
            return []
        
        z_score = (spread - spread_mean_p2) / self.pb2_std_dev


        #basket overvalued -> short basket, long on individuals
        #if self.pb1_std_dev is not None and (diff_actual - self.pb1_ema[-1]) > 4.5 * self.pb1_std_dev:
        if z_score >= z_threshold_2:
            logger.print(f"Basket short-regime, diff-EMA {spread}-{self.pb2_ema[-1]} > 4.5 * {self.pb2_std_dev}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(-math.floor(order_depth_cr.sell_orders[best_ask_cr]/4), -math.floor(order_depth_jams.sell_orders[best_ask_jams]/2), order_depth_pb2.buy_orders[best_bid_pb2])
            logger.print(f"Initial pkg_amt_indiv: {pkg_amt_indiv}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) > self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if (4*pkg_amt_indiv + self.get_position(CROISSANTS, state)) > self.position_limit[CROISSANTS]:
                pkg_amt_indiv = math.floor((self.position_limit[CROISSANTS] - self.get_position(CROISSANTS, state))/4)
            if (2*pkg_amt_indiv + self.get_position(JAMS, state)) > self.position_limit[JAMS]:
                pkg_amt_indiv = math.floor((self.position_limit[JAMS] - self.get_position(JAMS, state))/2)
            logger.print(f"Final pkg_amt_indiv: {pkg_amt_indiv}")

            orders.append(Order(PICNIC_BASKET2, best_bid_pb2, -pkg_amt_indiv))
            logger.print(f"Going short on PB1 with order {orders[-1]}")
            orders.append(Order(CROISSANTS, best_ask_cr, 4*pkg_amt_indiv))
            logger.print(f"Going long on CR with order {orders[-1]}")
            orders.append(Order(JAMS, best_ask_jams, 2*pkg_amt_indiv))
            logger.print(f"Going long on JAMS with order {orders[-1]}")

        #if self.pb1_std_dev is not None and (diff_actual - self.pb1_ema[-1]) < -4.5 * self.pb1_std_dev:
        if z_score <= z_threshold_2:
            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(math.floor(order_depth_cr.buy_orders[best_bid_cr]/4), math.floor(order_depth_jams.buy_orders[best_bid_jams]/2), -order_depth_pb2.sell_orders[best_ask_pb2])
            logger.print(f"Initial pkg_amt_indiv: {pkg_amt_indiv}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (-pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) < -self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if -4*pkg_amt_indiv + self.get_position(CROISSANTS, state) < -self.position_limit[CROISSANTS]:
                pkg_amt_indiv = math.floor((self.position_limit[CROISSANTS] - self.get_position(CROISSANTS, state))/4)
            logger.print(f"pkg_amt_indiv after if-1: {pkg_amt_indiv}")
            if -2*pkg_amt_indiv + self.get_position(JAMS, state) < -self.position_limit[JAMS]:
                pkg_amt_indiv = math.floor((self.position_limit[JAMS] - self.get_position(JAMS, state))/2)
            logger.print(f"pkg_amt_indiv after if-2: {pkg_amt_indiv}")

            orders.append(Order(PICNIC_BASKET2, best_ask_pb2, pkg_amt_indiv))
            logger.print(f"Going long on PB1 with order {orders[-1]}")
            orders.append(Order(CROISSANTS, best_bid_cr, -4*pkg_amt_indiv))
            logger.print(f"Going short on CR with order {orders[-1]}")
            orders.append(Order(JAMS, best_bid_cr, -2*pkg_amt_indiv))
            logger.print(f"Going short on JAMS with order {orders[-1]}")

        logger.print(f"Returning orders: {orders}") 
        return orders
    
    def update_pb2_params(self, state: TradingState):

        alpha_pb2 = 2 / (lookback_ema_pb2 + 1)
        self.pb2_diffs.append(self.get_pb2_diff(state))

        if self.pb2_ema == []:
            self.pb2_ema.append(self.pb2_diffs[-1])

        else:
            self.pb2_ema.append(alpha_pb2 * self.pb2_diffs[-1] + (1-alpha_pb2) * self.pb2_ema[-1])
        
        if len(self.pb2_diffs) > lookback_std_dev_pb2:
            self.pb2_diffs.pop(0)

        if len(self.pb2_ema) > lookback_ema_pb2:
            self.pb2_ema.pop(0)
        
        if len(self.pb2_ema) >= 2 and len(self.pb2_diffs) >= 2:
            self.pb2_std_dev = statistics.stdev(np.array(self.pb2_diffs, dtype=float) - np.array(self.pb2_ema, dtype=float)[-10:])

    

    def get_pb2_diff(self, state: TradingState):
        mid_price_pb2 = self.get_mid_price(PICNIC_BASKET2, state)
        mid_price_indiv = 4*self.get_mid_price(CROISSANTS, state) + 2*self.get_mid_price(JAMS, state)

        return mid_price_pb2 - mid_price_indiv

    def basket_basket_strategy(self, state: TradingState):
        
        orders: List[Order] = []

        self.update_bb_params(state)
        
        spread = self.get_bb_diff(state)

        order_depth_pb1 = state.order_depths.get(PICNIC_BASKET1, OrderDepth({}, {}))
        best_bid_pb1 = max(order_depth_pb1.buy_orders.keys()) if order_depth_pb1.buy_orders else None
        best_ask_pb1 = min(order_depth_pb1.sell_orders.keys()) if order_depth_pb1.sell_orders else None

        order_depth_pb2 = state.order_depths.get(PICNIC_BASKET2, OrderDepth({}, {}))
        best_bid_pb2 = max(order_depth_pb2.buy_orders.keys()) if order_depth_pb2.buy_orders else None
        best_ask_pb2 = min(order_depth_pb2.sell_orders.keys()) if order_depth_pb2.sell_orders else None

        order_depth_djembes = state.order_depths.get(DJEMBES, OrderDepth({}, {}))
        best_bid_djembes = max(order_depth_djembes.buy_orders.keys()) if order_depth_djembes.buy_orders else None
        best_ask_djembes = min(order_depth_djembes.sell_orders.keys()) if order_depth_djembes.sell_orders else None

        if self.bb_std_dev == 0:
            return []
        
        z_score = (spread - spread_mean_bb) / self.bb_std_dev


        #basket overvalued -> short basket, long on individuals
        #if self.pb1_std_dev is not None and (diff_actual - self.pb1_ema[-1]) > 4.5 * self.pb1_std_dev:
        if z_score >= z_threshold_bb:
            logger.print(f"Basket short-regime, diff-EMA {spread}-{self.bb_ema[-1]} > 4.5 * {self.bb_std_dev}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(-math.floor(order_depth_pb2.sell_orders[best_ask_pb2]/2), -math.floor(order_depth_djembes.sell_orders[best_ask_djembes]/2), order_depth_pb1.buy_orders[best_bid_pb1])
            logger.print(f"Initial pkg_amt_indiv: {pkg_amt_indiv}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) > self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if (4*pkg_amt_indiv + self.get_position(PICNIC_BASKET2, state)) > self.position_limit[PICNIC_BASKET2]:
                pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET2] - self.get_position(PICNIC_BASKET2, state))/2)
            if (2*pkg_amt_indiv + self.get_position(DJEMBES, state)) > self.position_limit[DJEMBES]:
                pkg_amt_indiv = math.floor((self.position_limit[DJEMBES] - self.get_position(DJEMBES, state))/2)
            logger.print(f"Final pkg_amt_indiv: {pkg_amt_indiv}")

            orders.append(Order(PICNIC_BASKET1, best_bid_pb1, -pkg_amt_indiv))
            logger.print(f"Going short on PB1 with order {orders[-1]}")
            orders.append(Order(PICNIC_BASKET2, best_ask_pb2, 2*pkg_amt_indiv))
            logger.print(f"Going long on CR with order {orders[-1]}")
            orders.append(Order(DJEMBES, best_ask_djembes, 2*pkg_amt_indiv))
            logger.print(f"Going long on JAMS with order {orders[-1]}")

        #if self.pb1_std_dev is not None and (diff_actual - self.pb1_ema[-1]) < -4.5 * self.pb1_std_dev:
        if z_score <= z_threshold_bb:
            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(math.floor(order_depth_pb2.buy_orders[best_bid_pb2]/2), math.floor(order_depth_djembes.buy_orders[best_bid_djembes]/2), -order_depth_pb1.sell_orders[best_ask_pb1])
            logger.print(f"Initial pkg_amt_indiv: {pkg_amt_indiv}")
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (-pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) < -self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if -2*pkg_amt_indiv + self.get_position(PICNIC_BASKET2, state) < -self.position_limit[PICNIC_BASKET2]:
                pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET2] - self.get_position(PICNIC_BASKET2, state))/2)
            logger.print(f"pkg_amt_indiv after if-1: {pkg_amt_indiv}")
            if -2*pkg_amt_indiv + self.get_position(DJEMBES, state) < -self.position_limit[DJEMBES]:
                pkg_amt_indiv = math.floor((self.position_limit[DJEMBES] - self.get_position(DJEMBES, state))/2)
            logger.print(f"pkg_amt_indiv after if-2: {pkg_amt_indiv}")

            orders.append(Order(PICNIC_BASKET1, best_ask_pb1, pkg_amt_indiv))
            logger.print(f"Going long on PB1 with order {orders[-1]}")
            orders.append(Order(PICNIC_BASKET2, best_bid_pb2, -2*pkg_amt_indiv))
            logger.print(f"Going short on CR with order {orders[-1]}")
            orders.append(Order(DJEMBES, best_bid_djembes, -2*pkg_amt_indiv))
            logger.print(f"Going short on JAMS with order {orders[-1]}")

        logger.print(f"Returning orders: {orders}") 
        return orders
    
    def update_bb_params(self, state: TradingState):

        alpha_bb = 2 / (lookback_ema_bb + 1)
        self.bb_diffs.append(self.get_bb_diff(state))

        if self.bb_ema == []:
            self.bb_ema.append(self.bb_diffs[-1])

        else:
            self.bb_ema.append(alpha_bb * self.bb_diffs[-1] + (1-alpha_bb) * self.bb_ema[-1])
        
        if len(self.bb_diffs) > lookback_std_dev_bb:
            self.bb_diffs.pop(0)

        if len(self.bb_ema) > lookback_ema_bb:
            self.bb_ema.pop(0)
        
        if len(self.bb_ema) >= 2 and len(self.bb_diffs) >= 2:
            self.bb_std_dev = statistics.stdev(np.array(self.bb_diffs, dtype=float) - np.array(self.bb_ema, dtype=float)[-10:])

    

    def get_bb_diff(self, state: TradingState):
        '''#picnic basket 1
        best_bid = max(state.order_depths[PICNIC_BASKET1].buy_orders)
        best_ask = min(state.order_depths[PICNIC_BASKET1].sell_orders)
        best_bid_vol = abs(state.order_depths[PICNIC_BASKET1].buy_orders[best_bid])
        best_ask_vol = abs(state.order_depths[PICNIC_BASKET1].sell_orders[best_ask])
        mid_price_pb1 = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

        #croissants
        best_bid_croissants = max(state.order_depths[CROISSANTS].buy_orders)
        best_ask_croissants = min(state.order_depths[CROISSANTS].sell_orders)
        best_bid_vol_croissants = abs(state.order_depths[CROISSANTS].buy_orders[best_bid_croissants])
        best_ask_vol_croissants = abs(state.order_depths[CROISSANTS].sell_orders[best_ask_croissants])
        mid_price_croissants = (best_bid_croissants * best_ask_vol_croissants + best_ask_croissants * best_bid_vol_croissants) / (best_bid_vol_croissants + best_ask_vol_croissants)

        #jams
        best_bid_jams = max(state.order_depths[JAMS].buy_orders)
        best_ask_jams = min(state.order_depths[JAMS].sell_orders)
        best_bid_vol_jams = abs(state.order_depths[JAMS].buy_orders[best_bid_jams])
        best_ask_vol_jams = abs(state.order_depths[JAMS].sell_orders[best_ask_jams])
        mid_price_jams = (best_bid_jams * best_ask_vol_jams + best_ask_jams * best_bid_vol_jams) / (best_bid_vol_jams + best_ask_vol_jams)

        #djembes
        best_bid_djembes = max(state.order_depths[DJEMBES].buy_orders)
        best_ask_djembes = min(state.order_depths[DJEMBES].sell_orders)
        best_bid_vol_djembes = abs(state.order_depths[DJEMBES].buy_orders[best_bid_djembes])
        best_ask_vol_djembes = abs(state.order_depths[DJEMBES].sell_orders[best_ask_djembes])
        mid_price_djembes = (best_bid_djembes * best_ask_vol_djembes + best_ask_djembes * best_bid_vol_djembes) / (best_bid_vol_djembes + best_ask_vol_djembes)

        synth_mid = 6 * mid_price_croissants + 3 * mid_price_jams + mid_price_djembes
        return mid_price_pb1 - synth_mid'''
        mid_price_pb1 = self.get_mid_price(PICNIC_BASKET1, state)
        mid_price_indiv = 2*self.get_mid_price(PICNIC_BASKET2, state) + 2*self.get_mid_price(DJEMBES, state)

        return mid_price_pb1 - mid_price_indiv
    
    def fit_vol_smile(self, timestamp):
        # Fit parabola if we have enough observations
        data = self.vol_smile_coeffs.get(timestamp, {'m_t': [], 'iv': []})
        if len(data['m_t']) >= 3:  # Minimum 3 points for quadratic fit
            return np.polyfit(data['m_t'], data['iv'], 2)
        return None
    
    def store_iv_observations(self, timestamp, m_t, iv, voucher):
        # Store IV observations for all strikes to fit parabola
        if voucher not in self.voucher_coeffs:
            self.voucher_coeffs[voucher] = {'m_t': [], 'iv': []}
        if timestamp not in self.vol_smile_coeffs:
            self.vol_smile_coeffs[timestamp] = {'m_t': [], 'iv': []}
        self.vol_smile_coeffs[timestamp]['m_t'].append(m_t)
        self.vol_smile_coeffs[timestamp]['iv'].append(iv)
        self.voucher_coeffs[voucher]['m_t'].append(m_t)
        self.voucher_coeffs[voucher]['iv'].append(iv)
    
    def VOLCANIC_strat(self, state: TradingState):

        VOLCANIC_position = state.position[VOLCANIC_ROCK] if VOLCANIC_ROCK in state.position else 0
        VOLCANIC_order_depths = state.order_depths[VOLCANIC_ROCK]
        VOLCANIC_mid_price = (max(VOLCANIC_order_depths.buy_orders.keys()) + min(VOLCANIC_order_depths.sell_orders.keys())) / 2



        for voucher in self.vouchers:
            strike = self.strike[voucher]

            VOLCANIC_voucher_position = state.position[voucher] if voucher in state.position else 0

            VOLCANIC_voucher_order_depths = state.order_depths[voucher]

            VOLCANIC_voucher_mid_price = self.get_voucher_mid_price(VOLCANIC_voucher_order_depths, voucher)
            
            tte = state.timestamp / (1000000 * 365) + 4/365
            m_t = np.log(strike/VOLCANIC_mid_price)/np.sqrt(tte)
            volatility = BlackScholes.implied_volatility(VOLCANIC_voucher_mid_price, VOLCANIC_mid_price, strike, tte)
            self.store_iv_observations(state.timestamp, m_t, volatility, voucher)

        coeffs = self.fit_vol_smile(state.timestamp)
        base_iv = coeffs[-1] if coeffs.any() else None 

        VOLCANIC_orders = []
        VOLCANIC_voucher_orders = []

        self.base_iv_history.append(base_iv)
        
        voucher_order_book = {}
        voucher_make_book = {}
        orders_book = []
        for voucher in self.vouchers:
            strike = self.strike[voucher]
            VOLCANIC_voucher_position = state.position[voucher] if voucher in state.position else 0
            VOLCANIC_voucher_order_depths = state.order_depths[voucher]
            VOLCANIC_voucher_mid_price = self.get_voucher_mid_price(VOLCANIC_voucher_order_depths, voucher)


            
            data = self.voucher_coeffs.get(voucher, {'m_t': [], 'iv': []})
            iv = data['iv'][0]
            delta = BlackScholes.delta(VOLCANIC_mid_price, strike, tte, iv)
            if coeffs is not None:
                fair_iv = np.polyval(coeffs, m_t)
                iv_residual = volatility - fair_iv
            else: iv_residual = 0
            

            VOLCANIC_voucher_take_orders, VOLCANIC_voucher_make_orders = self.curve_aware_voucher_orders(
                voucher, iv_residual, base_iv, 
                VOLCANIC_voucher_order_depths, VOLCANIC_voucher_position
            )
            voucher_order_book[voucher] = VOLCANIC_voucher_take_orders
            voucher_make_book[voucher] = VOLCANIC_voucher_make_orders
            
            # Modified delta hedging using curve information
            delta = self.adjusted_delta_hedge(coeffs, delta, base_iv)

            VOLCANIC_orders = self.VOLCANIC_orders(VOLCANIC_order_depths, VOLCANIC_voucher_orders, VOLCANIC_position, VOLCANIC_voucher_position, delta)
            orders_book += VOLCANIC_orders
        logger.print(f"Tring to place order {voucher_order_book}")
        return voucher_order_book, voucher_make_book, orders_book
    
    def curve_aware_voucher_orders(self, voucher, iv_residual, base_iv, 
                                 VOLCANIC_voucher_order_depths, VOLCANIC_voucher_position):
        # Determine regime using base IV
        base_iv_z = self.calculate_base_iv_zscore(base_iv)
        
        # Adaptive residual threshold based on regime
        threshold = self.residual_threshold * (1 + abs(base_iv_z))
        logger.print(f"IV Residual: {iv_residual:.4f}, Threshold: {threshold:.4f}, Z: {base_iv_z:.2f}")
        
        if base_iv_z >= 1:  # High volatility regime
            if iv_residual > threshold:
                if abs(VOLCANIC_voucher_position) < self.position_limit[voucher]:
                    voucher_target_position = -self.position_limit[voucher]
                    if len(VOLCANIC_voucher_order_depths.buy_orders) > 0:
                        target_quantity = abs(voucher_target_position - VOLCANIC_voucher_position)
                        best_bid = max(VOLCANIC_voucher_order_depths.sell_orders.keys())
                        quantity = min(abs(VOLCANIC_voucher_order_depths.sell_orders[best_bid]), target_quantity)
                        quote_quantity = target_quantity - quantity
                        if quote_quantity == 0:
                            return [Order(voucher, best_bid, -quantity)]
                        else:
                            return [Order(voucher, best_bid, -quantity)], [Order(voucher, best_bid, -quote_quantity)]
            elif iv_residual < -threshold:
                if abs(VOLCANIC_voucher_position) < self.position_limit[voucher]:
                    voucher_target_position = self.position_limit[voucher]
                    if len(VOLCANIC_voucher_order_depths.sell_orders) > 0:
                        target_quantity = abs(voucher_target_position - VOLCANIC_voucher_position)
                        best_ask = min(VOLCANIC_voucher_order_depths.sell_orders.keys())
                        quantity = min(abs(VOLCANIC_voucher_order_depths.sell_orders[best_ask]), target_quantity)
                        quote_quantity = target_quantity - quantity
                        if quote_quantity == 0:
                            return [Order(voucher, best_ask, quantity)]
                        else:
                            return [Order(voucher, best_ask, quantity)], [Order(voucher, best_ask, quote_quantity)]
                
        elif base_iv_z < -1:  # Low volatility regime
            if iv_residual > threshold:
                if abs(VOLCANIC_voucher_position) < self.position_limit[voucher]:
                    voucher_target_position = -self.position_limit[voucher]
                    if len(VOLCANIC_voucher_order_depths.buy_orders) > 0:
                        target_quantity = abs(voucher_target_position - VOLCANIC_voucher_position)
                        best_bid = max(VOLCANIC_voucher_order_depths.sell_orders.keys())
                        quantity = min(abs(VOLCANIC_voucher_order_depths.sell_orders[best_bid]), target_quantity)
                        quote_quantity = target_quantity - quantity
                        if quote_quantity == 0:
                            return [Order(voucher, best_bid, -quantity)]
                        else:
                            return [Order(voucher, best_bid, -quantity)], [Order(voucher, best_bid, -quote_quantity)]
            elif iv_residual < -threshold:
                if abs(VOLCANIC_voucher_position) < self.position_limit[voucher]:
                    voucher_target_position = self.position_limit[voucher]
                    if len(VOLCANIC_voucher_order_depths.sell_orders) > 0:
                        target_quantity = abs(voucher_target_position - VOLCANIC_voucher_position)
                        best_ask = min(VOLCANIC_voucher_order_depths.sell_orders.keys())
                        quantity = min(abs(VOLCANIC_voucher_order_depths.sell_orders[best_ask]), target_quantity)
                        quote_quantity = target_quantity - quantity
                        if quote_quantity == 0:
                            return [Order(voucher, best_ask, quantity)]
                        else:
                            return [Order(voucher, best_ask, quantity)], [Order(voucher, best_ask, quote_quantity)]
        
        return None, None
    
    def calculate_base_iv_zscore(self, current_iv):
        # Maintain rolling window of base IV values
        if len(self.base_iv_history) > self.base_iv_window:
            self.base_iv_history.pop(0)
            
        if len(self.base_iv_history) < self.base_iv_window:
            return 0
            
        mean = np.mean(self.base_iv_history)
        std = np.std(self.base_iv_history)
        return (current_iv - mean) / std if std != 0 else 0
    
    def adjusted_delta_hedge(self, coeffs, original_delta, base_iv):
        # Adjust delta based on curve convexity
        if coeffs is None or len(coeffs) < 3:
            return original_delta
            
        # a coefficient controls convexity
        convexity_adj = 1 + 0.1 * coeffs[0]  # Example adjustment
        # Base IV adjustment for volatility regimes
        vol_adj = 1 + (base_iv - np.mean(self.base_iv_history[-20:]))/base_iv
        
        return original_delta * convexity_adj * vol_adj

    def get_voucher_mid_price(self, order_depths: OrderDepth, voucher):
        if (
            len(order_depths.buy_orders) > 0
            and len(order_depths.buy_orders) > 0
        ):
            bb = max(order_depths.buy_orders.keys())
            ba = min(order_depths.sell_orders.keys())
            DEFAULT_PRICES[voucher] = (bb + ba) / 2
            return (bb + ba) / 2
        else:
            return DEFAULT_PRICES[voucher]
 
    def VOLCANIC_orders(self, 
        VOLCANIC_order_depth: OrderDepth, 
        VOLCANIC_voucher_orders: List[Order],
        VOLCANIC_position: int,
        VOLCANIC_voucher_position: int,
        delta: float
    ) -> List[Order]:
        if VOLCANIC_voucher_orders == None or len(VOLCANIC_voucher_orders) == 0:
            VOLCANIC_voucher_position_post_trade = VOLCANIC_voucher_position
        else:
            VOLCANIC_voucher_position_post_trade = VOLCANIC_voucher_position + sum(order.quantity for order in VOLCANIC_voucher_orders)

        target_VOLCANIC_position = -delta * VOLCANIC_voucher_position_post_trade
        

        if target_VOLCANIC_position == VOLCANIC_position:
            return []

        target_VOLCANIC_quantity = target_VOLCANIC_position - VOLCANIC_position

        orders : List[Order] = []

        if target_VOLCANIC_quantity > 0:
            best_ask = min(VOLCANIC_order_depth.sell_orders.keys())
            quantity = min(abs(target_VOLCANIC_quantity), self.position_limit[VOLCANIC_ROCK] - VOLCANIC_position)

            if quantity > 0:
                orders.append(Order(VOLCANIC_ROCK, best_ask, round(quantity)))
        
        elif target_VOLCANIC_quantity < 0:
            best_bid = min(VOLCANIC_order_depth.buy_orders.keys())
            quantity = min(abs(target_VOLCANIC_quantity), self.position_limit[VOLCANIC_ROCK] + VOLCANIC_position)

            if quantity > 0:
                orders.append(Order(VOLCANIC_ROCK, best_bid, -round(quantity)))

        return orders


    
    def kelp_orders(self, order_depth: OrderDepth, timespan:int, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            #fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])
            
            fair_value = self.ema_prices["KELP"]
            #mmmid_price

            # take all orders we can
            # for ask in order_depth.sell_orders.keys():
            #     if ask <= fair_value - kelp_take_width:
            #         ask_amount = -1 * order_depth.sell_orders[ask]
            #         if ask_amount <= 50:
            #             quantity = min(ask_amount, position_limit - position)
            #             if quantity > 0:
            #                 orders.append(Order("KELP", ask, quantity))
            #                 buy_order_volume += quantity
            
            # for bid in order_depth.buy_orders.keys():
            #     if bid >= fair_value + kelp_take_width:
            #         bid_amount = order_depth.buy_orders[bid]
            #         if bid_amount <= 50:
            #             quantity = min(bid_amount, position_limit + position)
            #             if quantity > 0:
            #                 orders.append(Order("KELP", bid, -1 * quantity))
            #                 sell_order_volume += quantity

            # only taking best bid/ask
        
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 50:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 50:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 2)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", bbbf + 1, buy_quantity))  # Buy order

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", baaf - 1, -sell_quantity))  # Sell order

        return orders

    def run(self, state: TradingState):
        result = {}

        self.update_ema_prices(state)

        rainforest_resin_fair_value = 10000  # Participant should calculate this value
        rainforest_resin_width = 1
        rainforest_resin_position_limit = 50
        
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10

        


        if RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = state.position[RAINFOREST_RESIN] if RAINFOREST_RESIN in state.position else 0
            rainforest_resin_orders = self.rainforest_resin_orders(state.order_depths[RAINFOREST_RESIN], DEFAULT_PRICES[RAINFOREST_RESIN], rainforest_resin_width, rainforest_resin_position, self.position_limit[RAINFOREST_RESIN])
            result[RAINFOREST_RESIN] = rainforest_resin_orders
        
        try:
            result[KELP] = self.kelp_strategy(state)
        except Exception as e:
            print("Error in kelp strategy")
            print(e)

        try:
            result[SQUID_INK] = self.squid_ink_strategy(state)
        except Exception as e:
            print("Error in squid ink strategy")
            print(e)

        
        result[PICNIC_BASKET1] = self.basket_1_strategy(state)
        result[PICNIC_BASKET2] = self.basket_2_strategy(state)
        #result[PICNIC_BASKET1] = self.basket_basket_strategy(state)

        
        if KELP in state.order_depths:
            kelp_position = state.position[KELP] if KELP in state.position else 0
            kelp_orders = self.kelp_strategy(state)
            #self.kelp_orders(state.order_depths["KELP"], kelp_timespan, kelp_make_width, kelp_take_width, kelp_position, kelp_position_limit)
            result[KELP] = kelp_orders

        
        traderData = jsonpickle.encode( {"pb1_ema": self.pb1_ema, "pb1_std_dev": self.pb1_std_dev} )
        traderData = jsonpickle.encode( {"pb2_ema": self.pb2_ema, "pb2_std_dev": self.pb2_std_dev} )
        traderData = jsonpickle.encode( {"bb_ema": self.bb_ema, "bb_std_dev": self.bb_std_dev} )

        take_book, make_book, order_book = self.VOLCANIC_strat(state)
        for voucher in self.vouchers:
            if take_book[voucher] != None or make_book[voucher] != None:
                result[voucher] = take_book[voucher] + make_book[voucher]

        if order_book != None:
            result[VOLCANIC_ROCK] = order_book
            # print(f"COCONUT: {result[Product.COCONUT]}")


        conversions = 1

        logger.print(self.vwap_history)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    