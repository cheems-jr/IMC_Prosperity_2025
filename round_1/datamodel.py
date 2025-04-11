import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

#example: Listing(symbol="PRODUCT1", product="PRODUCT1", denomination="SEASHELLS")
class Listing:

    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination
        
#required for conversions, includes all necessary info such as costs                 
class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex
        
"""
notes: 
must hold long or short before conversion request
conversion request cannot exceed current holdings
if we have a short pos of -10 (owing 10 units), can only request conversion of up to 10 (11 or more will be ignored)
when conversion occurs, transportation fees and tariffs apply -> see ConversionObservation object attributes
conversion requests are optional, can return 0 or Null if no desire for conversion 
"""
class Observation:

    def __init__(self, plainValueObservations: Dict[Product, ObservationValue], conversionObservations: Dict[Product, ConversionObservation]) -> None:
        #a dictionary made up of the products as keys and 'ObservationValues' (integers) as values, e.g {'Seashells': 9}
        self.plainValueObservations = plainValueObservations
        #a dictionary made up of the products as keys and 'ConversionObservations' useful for conversion requests, e.g. {'Seashells': ConversionObservations obj}
        #access properties via Observation(our_obj).conversionObservations['Seashell'].bidPrice e.g. ? (maybe a .val is needed, let's see)
        self.conversionObservations = conversionObservations
        
    def __str__(self) -> str:
        return "(plainValueObservations: " + jsonpickle.encode(self.plainValueObservations) + ", conversionObservations: " + jsonpickle.encode(self.conversionObservations) + ")"
     
#how much of what product is being traded at what price
class Order:

    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    

#order depth defined on a per-symbol basis
#collection of all outstanding buy & sell orders sent by the bots
# ---------------------------------------- MODIFIED THE CLASS DEFINITION, OTHERWISE THE TEST SCRIPT WON'T WORK ----------------------------------------
class OrderDepth:

    def __init__(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]):
        #buy and sell order properties of object of class OrderDepth are of type Dict where the keys and values are of type int
        #keys are price levels, values are order quantities, e.g. {9: 5, 10: 4} means total order qty of 5 at price lvl 9 and total order qty of 4 at price lvl 10
        #sell orders have negative order quantities, e.g. {10: -2}
        #buy order levels should always be below sell order levels (if not, bots should trade between each other)
        self.buy_orders = buy_orders
        self.sell_orders = sell_orders


class Trade:

    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId=None, seller: UserId=None, timestamp: int=0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        #only non-empty if our algo is buyer, then "SUBMISSION"
        self.buyer = buyer
        #only non-empty if our algo is seller, then "SUBMISSION"
        self.seller = seller
        self.timestamp = timestamp

    #user-friendly representation, used in print() and str()
    def __str__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

    #dev-friendly representation, use for debugging
    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

#(object) is legacy code, not really necessary, () would work as well (?)
class TradingState(object):

    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        #all buy and sell orders per product sent by other participants
        #structure is dictionary: {'Seashells': OrderDepth obj}
        #, where the OrderDepth object can supply us with all the buy and sell orders on the market for that particular product at that particular point in time (state)
        self.order_depths = order_depths
        #trades done by the algo SINCE LAST TRADINGSTATE. Dictionary of Trade objects with products as keys, e.g {'Seashells': Trade obj}
        self.own_trades = own_trades
        #trades done by other participants SINCE LAST TRADINGSTATE. Same data structure as own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    
class ProsperityEncoder(JSONEncoder):

        def default(self, o):
            return o.__dict__