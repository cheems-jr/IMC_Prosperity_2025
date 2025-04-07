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

PRODUCTS = [
    RAINFOREST_RESIN,
    KELP,
    SQUID_INK
]

DEFAULT_PRICES = {
    RAINFOREST_RESIN: 10000,
    KELP: 2020,
    SQUID_INK: 2000
}



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
            self.ema_prices[product] = None

        self.ema_param = 0.5

        self.position_limit = {
            RAINFOREST_RESIN : 50,
            KELP : 50,
            SQUID_INK: 50
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
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS[1:]:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

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
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP] - 1), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP] + 1), ask_volume))
        
        if position_kelp > 0:
            # Long position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP] - 2), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP]), ask_volume))

        if position_kelp < 0:
            # Short position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP]), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP] + 2), ask_volume))

        return orders
    
    def squid_ink_strategy(self, state: TradingState):
        orders: List[Order] = []
        product = SQUID_INK
        position = self.get_position(product, state)
        order_depth = state.order_depths.get(product, OrderDepth({}, {}))
        recent_trades = state.market_trades.get(product, [])
        vwap = self.update_vwap(recent_trades)
        volatility = self.calculate_volatility()

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Estimate mid-price
        mid_price = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
        elif best_bid:
            mid_price = best_bid
        elif best_ask:
            mid_price = best_ask
        elif self.ema_prices.get(product) is not None:
            mid_price = self.ema_prices[product]
        else:
            return []

        # Blend VWAP and mid-price if available
        alpha = 0.3 if volatility < 2 else 0.1  # Reduce VWAP influence further in high volatility
        if mid_price and vwap:
            fair_value = alpha * vwap + (1 - alpha) * mid_price
        elif mid_price:
            fair_value = mid_price
        else:
            return []

        buy_limit = self.position_limit[product] - position
        sell_limit = -self.position_limit[product] - position

        # --- More conservative and signal-based approach ---
        signal_threshold = 0.5 + volatility * 0.2 # Increase threshold with volatility
        order_size = max(1, int(10 / (volatility + 1))) # Reduce order size with volatility

        # Check for a significant deviation from fair value to initiate trades
        if best_ask is not None and best_ask < fair_value - signal_threshold and buy_limit > 0:
            buy_quantity = min(-order_depth.sell_orders.get(best_ask, 0), buy_limit, order_size)
            if buy_quantity > 0:
                orders.append(Order(product, best_ask, buy_quantity))

        if best_bid is not None and best_bid > fair_value + signal_threshold and sell_limit < 0:
            sell_quantity = min(order_depth.buy_orders.get(best_bid, 0), -sell_limit, order_size)
            if sell_quantity > 0:
                orders.append(Order(product, best_bid, -sell_quantity))

        # Place passive orders closer to the fair value with reduced size
        passive_spread = 0.2 + volatility * 0.1 # Wider spread in higher volatility
        passive_buy_price = int(fair_value - passive_spread)
        passive_sell_price = int(fair_value + passive_spread)
        passive_order_size = min(2, order_size)

        if buy_limit > 0 and (best_bid is None or passive_buy_price > best_bid):
            orders.append(Order(product, passive_buy_price, passive_order_size))

        if sell_limit < 0 and (best_ask is None or passive_sell_price < best_ask):
            orders.append(Order(product, passive_sell_price, -passive_order_size))

        return orders


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

        """
        if KELP in state.order_depths:
            kelp_position = state.position[KELP] if KELP in state.position else 0
            kelp_orders = self.kelp_strategy(TradingState)
            #self.kelp_orders(state.order_depths["KELP"], kelp_timespan, kelp_make_width, kelp_take_width, kelp_position, kelp_position_limit)
            result[KELP] = kelp_orders"""

        
        traderData = jsonpickle.encode( {"squid_ink_vwap": self.vwap_history} )
        #jsonpickle.encode( { "kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap })


        conversions = 1

        logger.print(self.vwap_history)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    