from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Any, Dict, List
import statistics

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

RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

PRODUCTS = [
    RAINFOREST_RESIN,
    KELP,
    SQUID_INK,
    CROISSANTS,
    JAMS,
    DJEMBES,
    PICNIC_BASKET1,
    PICNIC_BASKET2
]

DEFAULT_PRICES = {
    RAINFOREST_RESIN: 10000,
    KELP: 2020,
    SQUID_INK: 2000,
    CROISSANTS: 4297,
    JAMS: 6593,
    DJEMBES: 13436,
    PICNIC_BASKET1: 59051,
    PICNIC_BASKET2: 30408
}

lookback_std_dev_pb1 = 10
lookback_ema_pb1 = 20
lookback_std_dev_pb2 = 10
lookback_ema_pb2 = 20
z_threshold = 7
z_threshold_2 = 5
spread_mean = 49
spread_mean_p2 = 25

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

        self.ema_param = 0.5



        self.position_limit = {
            RAINFOREST_RESIN : 50,
            KELP : 50,
            SQUID_INK: 50,
            CROISSANTS: 250,
            JAMS: 350,
            DJEMBES: 60,
            PICNIC_BASKET1: 60,
            PICNIC_BASKET2: 100
        }

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
        
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

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
        for product in PRODUCTS[1:]:
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

    def calculate_atr(self, state: TradingState, product: str, period: int = 14) -> float:
        """
        Calculates the Average True Range (ATR) for a given product.
        
        Args:
            state: TradingState object with market data.
            product: Asset symbol (e.g., "SQUID_INK").
            period: Lookback window for ATR (default=14).
        
        Returns:
            Current ATR value.
        """
        
        # Get current candle data (high, low, close)
        current_high = max(state.order_depths[product].sell_orders.keys())
        current_low = min(state.order_depths[product].buy_orders.keys())
        current_close = state.market_trades[product][-1].price if state.market_trades.get(product) else None
        
        if not current_close:
            return self.atr_tracker[product]['atr'] or 0.0  # Return previous ATR if no trades
        
        # --- Calculate True Range (TR) ---
        tr = 0.0
        if self.atr_tracker[product]['prev_close'] is None:
            tr = current_high - current_low  # First TR = High - Low
        else:
            prev_close = self.atr_tracker[product]['prev_close']
            tr = max(
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close)
            )
        
        # Update TR history
        self.atr_tracker[product]['tr_history'].append(tr)
        if len(self.atr_tracker[product]['tr_history']) > period:
            self.atr_tracker[product]['tr_history'].pop(0)
        
        # --- Calculate ATR ---
        if len(self.atr_tracker[product]['tr_history']) < period:
            # Initial ATR = Simple Average of TRs
            atr = sum(self.atr_tracker[product]['tr_history']) / len(self.atr_tracker[product]['tr_history'])
        else:
            # EMA of TR (smoothing factor alpha = 2/(period+1))
            alpha = 2 / (period + 1)
            atr = self.atr_tracker[product]['atr'] or tr  # Use previous ATR or current TR if no history
            atr = (tr - atr) * alpha + atr
        
        # Update tracker
        self.atr_tracker[product]['atr'] = atr
        self.atr_tracker[product]['prev_close'] = current_close
        
        return atr
    
    def squid_ink_strategy(self, state: TradingState):
        orders: List[Order] = []
        product = SQUID_INK
        position = self.get_position(product, state)
        order_depth = state.order_depths.get(product, OrderDepth({}, {}))
        recent_trades = state.market_trades.get(product, [])
        vwap = self.update_vwap(recent_trades)
        volatility = self.calculate_volatility()

        ema_short = self.ema_prices[product]['ema_5']
        ema_long = self.ema_prices[product]['ema_20']

        atr =  self.calculate_atr(state, SQUID_INK, period=14)

        fair_value = 0.7 * vwap + 0.3 * ema_long

        # --- Trend Filter ---
        trend_up = ema_short > ema_long
        
        # --- Entry/Exit Thresholds ---
        entry_threshold = 0.9 * atr
        exit_threshold = 0.5 * atr

        # --- Get Best Bid/Ask, mid_price ---
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        mid_price = self.get_mid_price(SQUID_INK, state)
            
        # Long Entry (Oversold)
        if (mid_price < fair_value) and trend_up:
            buy_price = best_ask
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            quantity = min(best_ask_amount, self.position_limit - position)
            orders.append(Order(product, buy_price, quantity))
        
        # Short Entry (Overbought)
        elif (mid_price > fair_value) and not trend_up:
            sell_price = best_bid
            best_bid_amount = order_depth.buy_orders[best_bid]
            quantity = min(best_bid_amount, self.position_limit + position)
            orders.append(Order(product, sell_price, -quantity))
        
        # Exit Conditions
        if position > 0 and mid_price >= fair_value - exit_threshold:
            orders.append(Order(product, best_bid, -position))  # Close long
        elif position < 0 and mid_price <= fair_value + exit_threshold:
            orders.append(Order(product, best_ask, position))  # Close short
        
        return orders
    
    def get_pb1_diff(self, state: TradingState):
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



        """
        #calculate the mid price difference between basket 1 and its constituents
        delta_pb1_indiv = self.get_mid_price(PICNIC_BASKET1, state) - (6*self.get_mid_price(CROISSANTS, state) + 3*self.get_mid_price(JAMS, state) + self.get_mid_price(DJEMBES, state))

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

        logger.print(f"Current delta: {delta_pb1_indiv}")
        #if delta_pb1_indiv > 50+49:
            #logger.print(f"CR - sell order depth: {order_depth_cr.sell_orders}, best ask: {best_ask_cr}")
            #logger.print(f"CR - # of sell orders @ best_ask: {order_depth_cr.sell_orders[best_ask_cr]}")
            #logger.print(f"JAMS - sell order depth: {order_depth_jams.sell_orders}, best ask: {best_ask_jams}")
            #logger.print(f"JAMS - # of sell orders @ best_ask: {order_depth_jams.sell_orders[best_ask_jams]}")
            #logger.print(f"DJE - sell order depth: {order_depth_dje.sell_orders}, best ask: {best_ask_dje}")
            #logger.print(f"DJE - # of sell orders @ best_ask: {order_depth_dje.sell_orders[best_ask_dje]}")

        if delta_pb1_indiv < -50+49:
            logger.print(f"CR - buy order depth: {order_depth_cr.buy_orders}, best bid: {best_bid_cr}")
            logger.print(f"CR - # of buy orders @ best_bid: {order_depth_cr.buy_orders[best_bid_cr]}")
            logger.print(f"JAMS - buy order depth: {order_depth_jams.buy_orders}, best bid: {best_bid_jams}")
            logger.print(f"JAMS - # of buy orders @ best_bid: {order_depth_jams.buy_orders[best_bid_jams]}")
            logger.print(f"DJE - buy order depth: {order_depth_dje.buy_orders}, best bid: {best_bid_dje}")
            logger.print(f"DJE - # of buy orders @ best_bid: {order_depth_dje.buy_orders[best_bid_dje]}")
        
        #first simple strat: when diff > 50, go short on basket & go long on individuals in largest 6-3-1 constellation
        #right now we're not considering any deviations from the 6-3-1 split or the fair values of anything
        #could be that we're now going long on an overvalued constituent or not fully utilising our limits due to the 6-3-1
        if delta_pb1_indiv > 50+49 and (-order_depth_cr.sell_orders[best_ask_cr] >= 6 and -order_depth_jams.sell_orders[best_ask_jams] >= 3 and -order_depth_dje.sell_orders[best_ask_dje] >= 1):
        #if delta_pb1_indiv > 50 and (-order_depth_cr.sell_orders[best_ask_cr] >= 6 and -order_depth_jams.sell_orders[best_ask_jams] >= 3 and -order_depth_dje.sell_orders[best_ask_dje] >= 1):  
            logger.print(f"Entered case delta ({delta_pb1_indiv}) > 50+49")
            #short as many baskets as possible for now
            pb1_amt = self.get_position(PICNIC_BASKET1, state) + self.position_limit[PICNIC_BASKET1]
            orders.append(Order(PICNIC_BASKET1, best_bid_pb1, -pb1_amt))

            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(-math.floor(order_depth_cr.sell_orders[best_ask_cr]/6), -math.floor(order_depth_jams.sell_orders[best_ask_jams]/3), -order_depth_dje.sell_orders[best_ask_dje])
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

            #long maximum amount of 6-3-1 splits
            if pkg_amt_indiv > 0:
                #in the end we go long on 10*pkg_amt_indiv, which is equivalent to 1 basket -> basket_amt = pkg_amt_indiv/10
                #orders.append(Order(PICNIC_BASKET1, best_bid_pb1, -pkg_amt_indiv/10))
                #logger.print(f"Shorting PB1 with order {orders[-1]}")
                orders.append(Order(CROISSANTS, best_ask_cr, 6*pkg_amt_indiv))
                logger.print(f"Going long on CR with order {orders[-1]}")
                orders.append(Order(JAMS, best_ask_jams, 3*pkg_amt_indiv))
                logger.print(f"Going long on JAMS with order {orders[-1]}")
                orders.append(Order(DJEMBES, best_ask_dje, pkg_amt_indiv))
                logger.print(f"Going long on DJE with order {orders[-1]}")

        if delta_pb1_indiv < -50+49 and (order_depth_cr.buy_orders[best_bid_cr] >= 6 and order_depth_jams.buy_orders[best_bid_jams] >= 3 and order_depth_dje.buy_orders[best_bid_dje] >= 1):
        #if delta_pb1_indiv < -50 and (order_depth_cr.buy_orders[best_bid_cr] >= 6 and order_depth_jams.buy_orders[best_bid_jams] >= 3 and order_depth_dje.buy_orders[best_bid_dje] >= 1):
            logger.print(f"Entered case delta ({delta_pb1_indiv}) < -50+49")
            orders.append(Order(PICNIC_BASKET1, best_ask_pb1, self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))

            #what is the lower limit to how many 6-3-1 splits we can buy based on the order book?
            pkg_amt_indiv = min(math.floor(order_depth_cr.buy_orders[best_bid_cr]/6), math.floor(order_depth_jams.buy_orders[best_bid_jams]/3), order_depth_dje.buy_orders[best_bid_dje])
            #what is the lower limit to how many 6-3-1 splits we can buy based on the current positions and limits?
            #if (-pkg_amt_indiv + self.get_position(PICNIC_BASKET1, state)) < -self.position_limit[PICNIC_BASKET1]:
                #pkg_amt_indiv = math.floor((self.position_limit[PICNIC_BASKET1] - self.get_position(PICNIC_BASKET1, state)))
            if -6*pkg_amt_indiv + self.get_position(CROISSANTS, state) < -self.position_limit[CROISSANTS]:
                pkg_amt_indiv = math.floor((self.position_limit[CROISSANTS] - self.get_position(CROISSANTS, state))/6)
            if -3*pkg_amt_indiv + self.get_position(JAMS, state) < -self.position_limit[JAMS]:
                pkg_amt_indiv = math.floor((self.position_limit[JAMS] - self.get_position(JAMS, state))/3)
            if -pkg_amt_indiv + self.get_position(DJEMBES, state) < self.position_limit[DJEMBES]:
                pkg_amt_indiv = math.floor(self.position_limit[DJEMBES] - self.get_position(DJEMBES, state))

            #long maximum amount of 6-3-1 splits
            if pkg_amt_indiv > 0:
                #orders.append(Order(PICNIC_BASKET1, best_ask_pb1, pkg_amt_indiv/10))
                orders.append(Order(CROISSANTS, best_bid_cr, -6*pkg_amt_indiv))
                orders.append(Order(JAMS, best_bid_cr, -3*pkg_amt_indiv))
                orders.append(Order(DJEMBES, best_bid_cr, -pkg_amt_indiv))

        #in no-price-diff-arbitrage-regime, go neutral
        if -50+49 <= delta_pb1_indiv <= 50+49:
        #if -50 <= delta_pb1_indiv <= 50:
            if self.get_position(PICNIC_BASKET1, state) > 0:
                orders.append(Order(PICNIC_BASKET1, best_bid_cr, -self.get_position(PICNIC_BASKET1, state)))
            elif self.get_position(PICNIC_BASKET1, state) < 0:
                orders.append(Order(PICNIC_BASKET1, best_ask_cr, self.get_position(PICNIC_BASKET1, state)))

            if self.get_position(CROISSANTS, state) > 0:
                orders.append(Order(CROISSANTS, best_bid_cr, -self.get_position(CROISSANTS, state)))
            elif self.get_position(CROISSANTS, state) < 0:
                orders.append(Order(CROISSANTS, best_ask_cr, self.get_position(CROISSANTS, state)))

            if self.get_position(JAMS, state) > 0:
                orders.append(Order(JAMS, best_bid_cr, -self.get_position(JAMS, state)))
            elif self.get_position(JAMS, state) < 0:
                orders.append(Order(JAMS, best_ask_cr, self.get_position(JAMS, state)))

            if self.get_position(DJEMBES, state) > 0:
                orders.append(Order(DJEMBES, best_bid_cr, -self.get_position(DJEMBES, state)))
            elif self.get_position(DJEMBES, state) < 0:
                orders.append(Order(DJEMBES, best_ask_cr, self.get_position(DJEMBES, state)))
            """

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
        mid_price_pb2 = self.get_mid_price(PICNIC_BASKET2, state)
        mid_price_indiv = 4*self.get_mid_price(CROISSANTS, state) + 2*self.get_mid_price(JAMS, state)

        return mid_price_pb2 - mid_price_indiv

    """
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

        return orders"""

    def run(self, state: TradingState):
        result = {}

        self.update_ema_prices(state)

        #rainforest_resin_fair_value = 10000  # Participant should calculate this value
        rainforest_resin_width = 1
        #rainforest_resin_position_limit = 50
        
        """
        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10
        """
        
        # traderData = jsonpickle.decode(state.traderData)
        # print(state.traderData)
        # self.kelp_prices = traderData["kelp_prices"]
        # self.kelp_vwap = traderData["kelp_vwap"]

        """if RAINFOREST_RESIN in state.order_depths:
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
            print(e)"""

        
        result[PICNIC_BASKET1] = self.basket_1_strategy(state)
        result[PICNIC_BASKET2] = self.basket_2_strategy(state)

        """
        if KELP in state.order_depths:
            kelp_position = state.position[KELP] if KELP in state.position else 0
            kelp_orders = self.kelp_strategy(TradingState)
            #self.kelp_orders(state.order_depths["KELP"], kelp_timespan, kelp_make_width, kelp_take_width, kelp_position, kelp_position_limit)
            result[KELP] = kelp_orders"""

        
        traderData = jsonpickle.encode( {"pb1_ema": self.pb1_ema, "pb1_std_dev": self.pb1_std_dev} )
        traderData = jsonpickle.encode( {"pb2_ema": self.pb2_ema, "pb2_std_dev": self.pb2_std_dev} )

        #jsonpickle.encode( { "kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap })


        conversions = 1

        logger.print(self.vwap_history)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    