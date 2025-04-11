from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    #@TODO: Respect order limits 
    #       -> if |buy_qty| (us) > |sell_order_qty| (bot), check next order, fill if applicable and so on
    #       -> if |sell_qty| (us) > |buy_order_qty| (bot), check next order, fill if applicable and so on

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            print(f"\n----- Current product in run-method's order depth loop: {product} -----")
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 150;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
    
            if len(order_depth.sell_orders) != 0:
                print("Sell order depth : " + str(len(order_depth.sell_orders)))
                for order_key in order_depth.sell_orders:
                    print(f"ASK: Quantity {order_depth.sell_orders[order_key]} @ price {order_key}")
                ask_orders_list = list(order_depth.sell_orders.items())
                ask_orders_list.sort(key= lambda x: x[0])
                best_ask, best_ask_amount = ask_orders_list[0]
                if int(best_ask) < acceptable_price:
                    if (limits[product]-position[product] > 0):
                        if abs(position[product] + abs(best_ask_amount)) > abs(limits[product]):
                            orders.append(Order(product, best_ask, abs(limits[product]-position[product])))
                            print(f"--- BUY: {abs(limits[product]-position[product])}x {best_ask} ---")
                            print(f"--- Partial execution of counterparty's sell order due to limit of {limits[product]}")
                        else:
                            orders.append(Order(product, best_ask, -best_ask_amount))
                            print(f"--- BUY: {-best_ask_amount}x {best_ask} ---")
                else: 
                    print(f"--- NO BUY: Best ask price of {ask_orders_list[0][0]} is above or equal to the fair price of {acceptable_price} ---")
                
    
            if len(order_depth.buy_orders) != 0:
                print("Buy Order depth : " + str(len(order_depth.buy_orders)))
                for order_key in order_depth.buy_orders:
                    print(f"BID: Quantity {order_depth.buy_orders[order_key]} @ price {order_key}")
                buy_orders_list = list(order_depth.buy_orders.items())
                buy_orders_list.sort(key = lambda x: x[0], reverse=True)
                best_bid, best_bid_amount = buy_orders_list[0]
                if int(best_bid) > acceptable_price:
                    if (limits[product]+position[product] > 0):
                        if abs(position[product] - abs(best_bid_amount)) > abs(limits[product]):
                            orders.append(Order(product, best_bid, limits[product]+position[product]))
                            print(f"--- SELL: {limits[product]+position[product]}x {best_bid} ---")
                            print(f"--- Partial execution of counterparty's buy order due to limit of {limits[product]}")
                        else:
                            orders.append(Order(product, best_bid, -best_bid_amount))
                            print(f"--- SELL: {best_bid_amount}x {best_bid} ---")
                    else:
                        print(f"--- NO SELL: Limit reached (pos = {position[product]}), (limit = {-limits[product]})")

                else: 
                    print(f"--- NO SELL: Best bid price of {buy_orders_list[0][0]} is below or equal to the fair price of {acceptable_price} ---")
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData



# ------------------------------------ example generation ------------------------------------
from datamodel import Listing, Trade
timestamp = 1000

listings = {
    "PRODUCT1": Listing(
        symbol="PRODUCT1", 
        product="PRODUCT1", 
        denomination= "SEASHELLS"
    ),
    "PRODUCT2": Listing(
        symbol="PRODUCT2", 
        product="PRODUCT2", 
        denomination= "SEASHELLS"
    ),
}

order_depths = {
    "PRODUCT1": OrderDepth(
        buy_orders={10: 7, 9: 5},
        sell_orders={11: -4, 12: -8}
    ),
    "PRODUCT2": OrderDepth(
        buy_orders={142: 3, 141: 5},
        sell_orders={144: -5, 145: -8}
    ),	
}

own_trades = {
    "PRODUCT1": [],
    "PRODUCT2": []
}

market_trades = {
    "PRODUCT1": [
        Trade(
            symbol="PRODUCT1",
            price=11,
            quantity=4,
            buyer="",
            seller="",
            timestamp=900
        )
    ],
    "PRODUCT2": []
}

position = {
    "PRODUCT1": 3,
    "PRODUCT2": -5
}

limits = {
    "PRODUCT1": 5,
    "PRODUCT2": 5
}

observations = {}
traderData = ""

state = TradingState(
    traderData,
    timestamp,
  listings,
    order_depths,
    own_trades,
    market_trades,
    position,
    observations
)
# ------------------------------------ example generation ------------------------------------

#test current algo on example:
test = Trader()
test_res, test_conv, test_td = test.run(state)

print(f'Test result: {test_res}')
print(f'Test conversions: {test_conv}')
print(f'Test traderData: {test_td}')
