"""Build the NL→Taxi benchmark — the project's measuring stick.

Strategy (per user direction): bias toward gold drawn from validated corpus
(data/taxi/valid_corpus.jsonl) + visible upstream .taxi files, then author
realistic NL descriptions. Lower risk of buggy gold; still tests the model's
ability to translate intent into idiomatic Taxi.

Stratification:
  - easy        (~40): single-block, common pattern, single domain, no context
  - schema_aware (~30): prompt references types in an in_context_schema fragment
  - open_ended  (~30): multi-block, ambiguous intent, novel composition

Each entry validated:
  - Easy/Open-ended:   gold_taxi compiles standalone via /validate
  - Schema-aware:      [in_context_schema, gold_taxi] compiles via /validate-multi

Output: data/taxi/benchmark.jsonl   one JSON record per line:
  {
    id, prompt, in_context_schema?, gold_taxi, construct_tags,
    difficulty, gold_source, domain
  }
"""
from __future__ import annotations

import json
from pathlib import Path

from cloudgym.taxi.validator import TaxiValidator

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "data/taxi/benchmark.jsonl"


# -------------------------------------------------------------------- entries
# Each entry is (id, prompt, gold_taxi, in_context_schema_or_None,
# construct_tags, difficulty, gold_source, domain).
# `gold_source` ∈ {"authored", "corpus", "upstream"} — how the gold was sourced.
# Domains we cover: trading, banking, healthcare, ride-share, e-commerce, iot,
# logistics, social, content, telecom, generic.

# ============================== EASY (single-block, no context) ==============================
EASY: list[dict] = [
    # --- identifier types and simple models ---
    dict(id="easy.001", domain="ecommerce", construct=["type", "model"], gold_source="authored",
         prompt="Define a Customer model with an id (string-based identifier type called CustomerId) and an email field typed as EmailAddress.",
         gold='''type CustomerId inherits String
type EmailAddress inherits String
model Customer {
  id : CustomerId
  email : EmailAddress
}'''),
    dict(id="easy.002", domain="banking", construct=["type", "model"], gold_source="authored",
         prompt="Create an Account model with an account number (string), the customer's full name, and a numeric balance using a Money decimal type.",
         gold='''type AccountNumber inherits String
type FullName inherits String
type Money inherits Decimal
model Account {
  number : AccountNumber
  holder : FullName
  balance : Money
}'''),
    dict(id="easy.003", domain="trading", construct=["type"], gold_source="authored",
         prompt="Declare three primitive-derived types for trading: TickerSymbol (string), Price (decimal), and TradeId (string).",
         gold='''type TickerSymbol inherits String
type Price inherits Decimal
type TradeId inherits String'''),
    dict(id="easy.004", domain="healthcare", construct=["type", "model"], gold_source="authored",
         prompt="Define a Patient model with a patient identifier, date of birth, and a string field for blood type.",
         gold='''type PatientId inherits String
type DateOfBirth inherits Date
type BloodType inherits String
model Patient {
  id : PatientId
  dateOfBirth : DateOfBirth
  bloodType : BloodType
}'''),
    dict(id="easy.005", domain="rideshare", construct=["type", "model"], gold_source="authored",
         prompt="Model a Ride with a ride id, the rider's id, the driver's id, and a numeric fare.",
         gold='''type RideId inherits String
type RiderId inherits String
type DriverId inherits String
type Fare inherits Decimal
model Ride {
  id : RideId
  rider : RiderId
  driver : DriverId
  fare : Fare
}'''),
    dict(id="easy.006", domain="iot", construct=["type", "model"], gold_source="authored",
         prompt="Define a temperature reading model: a device id (string), a celsius value (decimal), and an instant timestamp.",
         gold='''type DeviceId inherits String
type Celsius inherits Decimal
type ReadingTime inherits Instant
model TemperatureReading {
  device : DeviceId
  celsius : Celsius
  takenAt : ReadingTime
}'''),
    dict(id="easy.007", domain="logistics", construct=["type", "model"], gold_source="authored",
         prompt="Create a Shipment model with a tracking number, an origin city (string), a destination city (string), and a Status enum-style String field.",
         gold='''type TrackingNumber inherits String
type CityName inherits String
type ShipmentStatus inherits String
model Shipment {
  trackingNumber : TrackingNumber
  origin : CityName
  destination : CityName
  status : ShipmentStatus
}'''),
    dict(id="easy.008", domain="social", construct=["type", "model"], gold_source="authored",
         prompt="A user-profile model: handle (string), display name (string), follower count (integer).",
         gold='''type Handle inherits String
type DisplayName inherits String
type FollowerCount inherits Int
model UserProfile {
  handle : Handle
  displayName : DisplayName
  followers : FollowerCount
}'''),
    dict(id="easy.009", domain="content", construct=["type", "model"], gold_source="authored",
         prompt="Create an Article model with title, an author id, a published timestamp, and a word count integer.",
         gold='''type ArticleTitle inherits String
type AuthorId inherits String
type PublishedAt inherits Instant
type WordCount inherits Int
model Article {
  title : ArticleTitle
  author : AuthorId
  publishedAt : PublishedAt
  wordCount : WordCount
}'''),
    dict(id="easy.010", domain="telecom", construct=["type", "model"], gold_source="authored",
         prompt="Define a CallRecord with a caller phone number (string), callee phone number (string), and call duration in seconds (integer).",
         gold='''type PhoneNumber inherits String
type DurationSeconds inherits Int
model CallRecord {
  caller : PhoneNumber
  callee : PhoneNumber
  durationSeconds : DurationSeconds
}'''),

    # --- enums ---
    dict(id="easy.011", domain="trading", construct=["enum"], gold_source="authored",
         prompt="Define a TradeSide enum with two values, BUY and SELL.",
         gold='''enum TradeSide {
  BUY,
  SELL
}'''),
    dict(id="easy.012", domain="ecommerce", construct=["enum"], gold_source="authored",
         prompt="Create an OrderStatus enum: PENDING, PAID, SHIPPED, DELIVERED, CANCELLED.",
         gold='''enum OrderStatus {
  PENDING,
  PAID,
  SHIPPED,
  DELIVERED,
  CANCELLED
}'''),
    dict(id="easy.013", domain="banking", construct=["enum"], gold_source="authored",
         prompt="Make an enum for AccountType with CHECKING, SAVINGS, CREDIT, and INVESTMENT.",
         gold='''enum AccountType {
  CHECKING,
  SAVINGS,
  CREDIT,
  INVESTMENT
}'''),
    dict(id="easy.014", domain="healthcare", construct=["enum"], gold_source="authored",
         prompt="Define a ClaimStatus enum with SUBMITTED, IN_REVIEW, APPROVED, DENIED.",
         gold='''enum ClaimStatus {
  SUBMITTED,
  IN_REVIEW,
  APPROVED,
  DENIED
}'''),
    dict(id="easy.015", domain="logistics", construct=["enum"], gold_source="authored",
         prompt="A ShippingPriority enum with values STANDARD, EXPRESS, OVERNIGHT.",
         gold='''enum ShippingPriority {
  STANDARD,
  EXPRESS,
  OVERNIGHT
}'''),

    # --- simple services ---
    dict(id="easy.016", domain="ecommerce", construct=["service"], gold_source="authored",
         prompt="Define an OrderService with a single operation findOrderById that takes a string OrderId and returns a String.",
         gold='''type OrderId inherits String
service OrderService {
  operation findOrderById(id : OrderId) : String
}'''),
    dict(id="easy.017", domain="banking", construct=["service"], gold_source="authored",
         prompt="Create an AccountService exposing one operation called getBalance, which takes an AccountNumber (string) and returns a decimal.",
         gold='''type AccountNumber inherits String
service AccountService {
  operation getBalance(accountNumber : AccountNumber) : Decimal
}'''),
    dict(id="easy.018", domain="rideshare", construct=["service"], gold_source="authored",
         prompt="A RideService with two operations: requestRide (takes RiderId, returns RideId) and cancelRide (takes RideId, returns Boolean).",
         gold='''type RiderId inherits String
type RideId inherits String
service RideService {
  operation requestRide(rider : RiderId) : RideId
  operation cancelRide(ride : RideId) : Boolean
}'''),
    dict(id="easy.019", domain="content", construct=["service"], gold_source="authored",
         prompt="Define an ArticleService with an operation publishArticle that takes an ArticleId (string) and returns an Instant publishedAt timestamp.",
         gold='''type ArticleId inherits String
service ArticleService {
  operation publishArticle(article : ArticleId) : Instant
}'''),
    dict(id="easy.020", domain="iot", construct=["service"], gold_source="authored",
         prompt="A DeviceService with operation listDevices that returns an array of strings (device ids).",
         gold='''service DeviceService {
  operation listDevices() : String[]
}'''),

    # --- annotations on models ---
    dict(id="easy.021", domain="generic", construct=["annotation"], gold_source="authored",
         prompt="Define a simple annotation called PII with no fields.",
         gold='''annotation PII'''),
    dict(id="easy.022", domain="generic", construct=["annotation"], gold_source="authored",
         prompt="Create an Indexed annotation with a single string field called name.",
         gold='''annotation Indexed {
  name : String
}'''),
    dict(id="easy.023", domain="generic", construct=["annotation", "model"], gold_source="authored",
         prompt="Make a Deprecated annotation taking a reason string and a since version string, then apply it to a model called LegacyUser.",
         gold='''annotation Deprecated {
  reason : String
  since : String
}

@Deprecated(reason = "use User instead", since = "2.0")
model LegacyUser {
  id : String
}'''),

    # --- arrays / multi-valued fields ---
    dict(id="easy.024", domain="ecommerce", construct=["model"], gold_source="authored",
         prompt="A Cart model containing a customer id (string) and a list of product ids (an array of strings).",
         gold='''type CustomerId inherits String
type ProductId inherits String
model Cart {
  customer : CustomerId
  products : ProductId[]
}'''),
    dict(id="easy.025", domain="social", construct=["model"], gold_source="authored",
         prompt="Define a Post model with content (string), an author id, and a list of tag strings.",
         gold='''type Tag inherits String
type AuthorId inherits String
model Post {
  content : String
  author : AuthorId
  tags : Tag[]
}'''),
    dict(id="easy.026", domain="logistics", construct=["model"], gold_source="authored",
         prompt="A Manifest model: a manifest id, a carrier name, and an array of tracking numbers.",
         gold='''type ManifestId inherits String
type CarrierName inherits String
type TrackingNumber inherits String
model Manifest {
  id : ManifestId
  carrier : CarrierName
  trackingNumbers : TrackingNumber[]
}'''),

    # --- nullable / optional ---
    dict(id="easy.027", domain="ecommerce", construct=["model"], gold_source="authored",
         prompt="Update model: an order id, a status, and an optional notes string (use ?).",
         gold='''type OrderId inherits String
type OrderStatus inherits String
model OrderUpdate {
  id : OrderId
  status : OrderStatus
  notes : String?
}'''),
    dict(id="easy.028", domain="healthcare", construct=["model"], gold_source="authored",
         prompt="A Visit model: patient id, visit timestamp, optional doctor id, and reason (string).",
         gold='''type PatientId inherits String
type DoctorId inherits String
model Visit {
  patient : PatientId
  visitedAt : Instant
  doctor : DoctorId?
  reason : String
}'''),

    # --- nested models ---
    dict(id="easy.029", domain="ecommerce", construct=["model"], gold_source="authored",
         prompt="Define an Address model with street, city, country, and a postal code, then an Order model that contains a customer id and a shipping Address.",
         gold='''model Address {
  street : String
  city : String
  country : String
  postalCode : String
}

type CustomerId inherits String
model Order {
  customer : CustomerId
  shippingAddress : Address
}'''),
    dict(id="easy.030", domain="banking", construct=["model"], gold_source="authored",
         prompt="A Transaction model that has from-account, to-account, amount, and an embedded TransactionMeta (which contains a reference number and an optional memo).",
         gold='''type AccountNumber inherits String
type Money inherits Decimal
model TransactionMeta {
  reference : String
  memo : String?
}
model Transaction {
  fromAccount : AccountNumber
  toAccount : AccountNumber
  amount : Money
  meta : TransactionMeta
}'''),

    # --- inheriting type aliases ---
    dict(id="easy.031", domain="trading", construct=["type"], gold_source="authored",
         prompt="Define a Quantity type that inherits from Int and a Notional type that inherits from Decimal.",
         gold='''type Quantity inherits Int
type Notional inherits Decimal'''),
    dict(id="easy.032", domain="generic", construct=["type"], gold_source="authored",
         prompt="Two boolean-derived types: IsActive and IsArchived, both inheriting from Boolean.",
         gold='''type IsActive inherits Boolean
type IsArchived inherits Boolean'''),

    # --- model with all primitives ---
    dict(id="easy.033", domain="generic", construct=["model"], gold_source="authored",
         prompt="A demo Mixed model showing one field of each common primitive: a String name, an Int count, a Decimal price, a Boolean active, and an Instant createdAt.",
         gold='''model Mixed {
  name : String
  count : Int
  price : Decimal
  active : Boolean
  createdAt : Instant
}'''),

    # --- enum referenced by model ---
    dict(id="easy.034", domain="ecommerce", construct=["enum", "model"], gold_source="authored",
         prompt="Define a Currency enum with values USD, EUR, GBP, and a Price model that has an amount (decimal) and a currency field referencing the enum.",
         gold='''enum Currency {
  USD,
  EUR,
  GBP
}
model Price {
  amount : Decimal
  currency : Currency
}'''),
    dict(id="easy.035", domain="trading", construct=["enum", "model"], gold_source="authored",
         prompt="An OrderType enum (LIMIT, MARKET, STOP) and a TradeOrder model carrying a ticker (string), a quantity (int), and an OrderType.",
         gold='''enum OrderType {
  LIMIT,
  MARKET,
  STOP
}
type Ticker inherits String
model TradeOrder {
  ticker : Ticker
  quantity : Int
  orderType : OrderType
}'''),

    # --- service with multiple operations ---
    dict(id="easy.036", domain="banking", construct=["service"], gold_source="authored",
         prompt="A LoanService with three operations: applyForLoan (returns LoanId from a CustomerId), approveLoan (returns Boolean from LoanId), and rejectLoan (returns Boolean from LoanId).",
         gold='''type CustomerId inherits String
type LoanId inherits String
service LoanService {
  operation applyForLoan(customer : CustomerId) : LoanId
  operation approveLoan(loan : LoanId) : Boolean
  operation rejectLoan(loan : LoanId) : Boolean
}'''),

    # --- closed model ---
    dict(id="easy.037", domain="generic", construct=["model"], gold_source="authored",
         prompt="Define a closed model called Coordinates with a latitude and longitude, both decimals.",
         gold='''closed model Coordinates {
  latitude : Decimal
  longitude : Decimal
}'''),

    # --- documentation comment ---
    dict(id="easy.038", domain="generic", construct=["model"], gold_source="authored",
         prompt="Define an Email model with a from address, a to address, and a subject. Add a documentation comment above the model that says 'Represents a single outbound email.'",
         gold='''[[ Represents a single outbound email. ]]
model Email {
  fromAddress : String
  toAddress : String
  subject : String
}'''),

    # --- two-block schema ---
    dict(id="easy.039", domain="rideshare", construct=["model"], gold_source="authored",
         prompt="Define a Driver model (id, full name, license number) and a Vehicle model (id, model, plate, owned by a driver id).",
         gold='''type DriverId inherits String
type LicenseNumber inherits String
model Driver {
  id : DriverId
  fullName : String
  license : LicenseNumber
}
type VehicleId inherits String
type LicensePlate inherits String
model Vehicle {
  id : VehicleId
  modelName : String
  plate : LicensePlate
  owner : DriverId
}'''),

    # --- annotation with applied parameters ---
    dict(id="easy.040", domain="generic", construct=["annotation", "model"], gold_source="authored",
         prompt="Define an annotation called Sensitivity that takes a string level field. Apply it to a CreditCard model with @Sensitivity(level = 'HIGH'). The model itself has cardNumber and cvv strings.",
         gold='''annotation Sensitivity {
  level : String
}
@Sensitivity(level = "HIGH")
model CreditCard {
  cardNumber : String
  cvv : String
}'''),
]

# ============================== SCHEMA-AWARE (in_context_schema + extension prompt) ==============================
SCHEMA_AWARE: list[dict] = [
    dict(id="sa.001", domain="ecommerce", construct=["model"], gold_source="authored",
         in_context='''type CustomerId inherits String
type EmailAddress inherits String
model Customer {
  id : CustomerId
  email : EmailAddress
}''',
         prompt="Given the Customer model above, define an Order model that has an order id (string), a reference to a Customer id, and a numeric total.",
         gold='''type OrderId inherits String
model Order {
  id : OrderId
  customer : CustomerId
  total : Decimal
}'''),
    dict(id="sa.002", domain="banking", construct=["service"], gold_source="authored",
         in_context='''type AccountNumber inherits String
type Money inherits Decimal
model Account {
  number : AccountNumber
  balance : Money
}''',
         prompt="Given the Account schema, define an AccountService with an operation getAccountByNumber that takes an AccountNumber and returns an Account, and an operation getBalance that takes an AccountNumber and returns Money.",
         gold='''service AccountService {
  operation getAccountByNumber(number : AccountNumber) : Account
  operation getBalance(number : AccountNumber) : Money
}'''),
    dict(id="sa.003", domain="trading", construct=["model"], gold_source="authored",
         in_context='''type TickerSymbol inherits String
type Price inherits Decimal''',
         prompt="Using the existing TickerSymbol and Price types, define a Quote model containing a ticker, a bid price, and an ask price.",
         gold='''model Quote {
  ticker : TickerSymbol
  bid : Price
  ask : Price
}'''),
    dict(id="sa.004", domain="healthcare", construct=["service"], gold_source="authored",
         in_context='''type PatientId inherits String
type DoctorId inherits String
model Patient {
  id : PatientId
}
model Visit {
  patient : PatientId
  doctor : DoctorId
  visitedAt : Instant
}''',
         prompt="Given Patient and Visit, define a VisitService with operations: bookVisit (takes a Patient id and a Doctor id, returns a Visit) and listVisitsForPatient (takes a PatientId, returns Visit[]).",
         gold='''service VisitService {
  operation bookVisit(patient : PatientId, doctor : DoctorId) : Visit
  operation listVisitsForPatient(patient : PatientId) : Visit[]
}'''),
    dict(id="sa.005", domain="rideshare", construct=["model"], gold_source="authored",
         in_context='''type RideId inherits String
type RiderId inherits String
type DriverId inherits String
type Fare inherits Decimal
model Ride {
  id : RideId
  rider : RiderId
  driver : DriverId
  fare : Fare
}''',
         prompt="Given the Ride model, define a RideReceipt model with a ride id, the fare amount, an Instant issuedAt, and a reference to the rider's id.",
         gold='''model RideReceipt {
  ride : RideId
  amount : Fare
  issuedAt : Instant
  rider : RiderId
}'''),
    dict(id="sa.006", domain="iot", construct=["query"], gold_source="authored",
         in_context='''type DeviceId inherits String
type Celsius inherits Decimal
model TemperatureReading {
  device : DeviceId
  celsius : Celsius
  takenAt : Instant
}''',
         prompt="Given the TemperatureReading model, write a TaxiQL query that finds an array of TemperatureReading.",
         gold='''find { TemperatureReading[] }'''),
    dict(id="sa.007", domain="ecommerce", construct=["query"], gold_source="authored",
         in_context='''type CustomerId inherits String
model Customer {
  id : CustomerId
}
model Order {
  id : String
  customer : CustomerId
  total : Decimal
}''',
         prompt="Given Customer and Order, write a TaxiQL find query that returns Order[].",
         gold='''find { Order[] }'''),
    dict(id="sa.008", domain="banking", construct=["service", "annotation"], gold_source="authored",
         in_context='''type AccountNumber inherits String
type Money inherits Decimal''',
         prompt="Given AccountNumber and Money, define a TransferService with an operation transfer that takes a from AccountNumber, a to AccountNumber, and an amount Money, returning a String confirmation reference.",
         gold='''service TransferService {
  operation transfer(fromAccount : AccountNumber, toAccount : AccountNumber, amount : Money) : String
}'''),
    dict(id="sa.009", domain="logistics", construct=["model"], gold_source="authored",
         in_context='''type TrackingNumber inherits String
enum ShipmentStatus { PENDING, IN_TRANSIT, DELIVERED, FAILED }''',
         prompt="Using TrackingNumber and ShipmentStatus, define a Shipment model with the tracking number, a status, an origin city string and a destination city string.",
         gold='''model Shipment {
  trackingNumber : TrackingNumber
  status : ShipmentStatus
  origin : String
  destination : String
}'''),
    dict(id="sa.010", domain="social", construct=["service"], gold_source="authored",
         in_context='''type UserId inherits String
type Handle inherits String
model UserProfile {
  id : UserId
  handle : Handle
}''',
         prompt="Given the UserProfile schema, define a UserService with operations: getUserById (returns UserProfile from UserId), getUserByHandle (returns UserProfile from Handle), and listFollowers (takes UserId, returns UserProfile[]).",
         gold='''service UserService {
  operation getUserById(id : UserId) : UserProfile
  operation getUserByHandle(handle : Handle) : UserProfile
  operation listFollowers(user : UserId) : UserProfile[]
}'''),
    dict(id="sa.011", domain="content", construct=["model", "enum"], gold_source="authored",
         in_context='''type ArticleId inherits String
type AuthorId inherits String''',
         prompt="Given ArticleId and AuthorId, define a Status enum (DRAFT, PUBLISHED, ARCHIVED) and an Article model carrying an id, an author, a title, a status, and an Instant publishedAt.",
         gold='''enum Status {
  DRAFT,
  PUBLISHED,
  ARCHIVED
}
model Article {
  id : ArticleId
  author : AuthorId
  title : String
  status : Status
  publishedAt : Instant
}'''),
    dict(id="sa.012", domain="trading", construct=["model"], gold_source="authored",
         in_context='''type TickerSymbol inherits String
type Price inherits Decimal
type Quantity inherits Int
enum TradeSide { BUY, SELL }''',
         prompt="Using these existing types, define a Trade model that has a ticker, a side, a quantity, a price, and an executedAt Instant.",
         gold='''model Trade {
  ticker : TickerSymbol
  side : TradeSide
  quantity : Quantity
  price : Price
  executedAt : Instant
}'''),
    dict(id="sa.013", domain="healthcare", construct=["service"], gold_source="authored",
         in_context='''type ClaimId inherits String
type PatientId inherits String
enum ClaimStatus { SUBMITTED, IN_REVIEW, APPROVED, DENIED }
model Claim {
  id : ClaimId
  patient : PatientId
  status : ClaimStatus
}''',
         prompt="Given the Claim schema, define a ClaimService with operations: submitClaim (takes PatientId, returns ClaimId), getClaim (takes ClaimId, returns Claim), and updateClaimStatus (takes ClaimId and a ClaimStatus, returns Claim).",
         gold='''service ClaimService {
  operation submitClaim(patient : PatientId) : ClaimId
  operation getClaim(id : ClaimId) : Claim
  operation updateClaimStatus(id : ClaimId, status : ClaimStatus) : Claim
}'''),
    dict(id="sa.014", domain="iot", construct=["model"], gold_source="authored",
         in_context='''type DeviceId inherits String''',
         prompt="Given a DeviceId, define a Device model carrying the id, a manufacturer string, a model string, and an optional firmware version string.",
         gold='''model Device {
  id : DeviceId
  manufacturer : String
  modelName : String
  firmware : String?
}'''),
    dict(id="sa.015", domain="generic", construct=["model"], gold_source="authored",
         in_context='''closed model Coordinates {
  latitude : Decimal
  longitude : Decimal
}''',
         prompt="Given the closed Coordinates model, define a Location model with a name (string) and a coords field of type Coordinates.",
         gold='''model Location {
  name : String
  coords : Coordinates
}'''),
    dict(id="sa.016", domain="ecommerce", construct=["query", "projection"], gold_source="authored",
         in_context='''type ProductId inherits String
type CustomerId inherits String
model Product {
  id : ProductId
  name : String
  price : Decimal
}
model Order {
  id : String
  customer : CustomerId
  product : ProductId
}''',
         prompt="Given Product and Order, write a TaxiQL find query that returns Order[] projected as a structure with the order id and the product id.",
         gold='''find { Order[] } as {
  orderId : String
  productId : ProductId
}'''),
    dict(id="sa.017", domain="rideshare", construct=["service"], gold_source="authored",
         in_context='''type RiderId inherits String
type DriverId inherits String
type RideId inherits String''',
         prompt="Given the rider/driver/ride id types, define a DispatchService with operations: matchDriver (takes a RiderId, returns a DriverId) and confirmRide (takes RiderId and DriverId, returns RideId).",
         gold='''service DispatchService {
  operation matchDriver(rider : RiderId) : DriverId
  operation confirmRide(rider : RiderId, driver : DriverId) : RideId
}'''),
    dict(id="sa.018", domain="banking", construct=["model"], gold_source="authored",
         in_context='''type AccountNumber inherits String
type Money inherits Decimal
enum TransactionDirection { CREDIT, DEBIT }''',
         prompt="Using these types, define a LedgerEntry model carrying an account number, a direction, an amount, an instant timestamp, and an optional reference string.",
         gold='''model LedgerEntry {
  account : AccountNumber
  direction : TransactionDirection
  amount : Money
  occurredAt : Instant
  reference : String?
}'''),
    dict(id="sa.019", domain="content", construct=["model"], gold_source="authored",
         in_context='''type AuthorId inherits String
model Author {
  id : AuthorId
  displayName : String
}''',
         prompt="Given the Author model, define a Comment model that has a comment id (string), a body, an authorId reference, and an Instant postedAt.",
         gold='''model Comment {
  id : String
  body : String
  author : AuthorId
  postedAt : Instant
}'''),
    dict(id="sa.020", domain="logistics", construct=["service"], gold_source="authored",
         in_context='''type ShipmentId inherits String
enum ShipmentStatus { PENDING, IN_TRANSIT, DELIVERED, FAILED }''',
         prompt="Given the existing types, define a TrackingService with operations: getStatus (takes ShipmentId, returns ShipmentStatus) and updateStatus (takes ShipmentId and a new ShipmentStatus, returns Boolean).",
         gold='''service TrackingService {
  operation getStatus(shipment : ShipmentId) : ShipmentStatus
  operation updateStatus(shipment : ShipmentId, status : ShipmentStatus) : Boolean
}'''),
    dict(id="sa.021", domain="trading", construct=["query"], gold_source="authored",
         in_context='''type TickerSymbol inherits String
model Quote {
  ticker : TickerSymbol
  bid : Decimal
  ask : Decimal
}''',
         prompt="Given the Quote model, write a TaxiQL find query that returns an array of Quote.",
         gold='''find { Quote[] }'''),
    dict(id="sa.022", domain="healthcare", construct=["model"], gold_source="authored",
         in_context='''type PatientId inherits String
type DoctorId inherits String
type DiagnosisCode inherits String''',
         prompt="Given those identifier types, define a Diagnosis model containing a patient id, a doctor id, a code, an Instant diagnosedAt, and an optional notes string.",
         gold='''model Diagnosis {
  patient : PatientId
  doctor : DoctorId
  code : DiagnosisCode
  diagnosedAt : Instant
  notes : String?
}'''),
    dict(id="sa.023", domain="iot", construct=["service"], gold_source="authored",
         in_context='''type DeviceId inherits String
model TemperatureReading {
  device : DeviceId
  celsius : Decimal
  takenAt : Instant
}''',
         prompt="Given the TemperatureReading model, define a SensorService with operations: getLatestReading (takes DeviceId, returns TemperatureReading) and listReadingsForDevice (takes DeviceId, returns TemperatureReading[]).",
         gold='''service SensorService {
  operation getLatestReading(device : DeviceId) : TemperatureReading
  operation listReadingsForDevice(device : DeviceId) : TemperatureReading[]
}'''),
    dict(id="sa.024", domain="ecommerce", construct=["model", "annotation"], gold_source="authored",
         in_context='''annotation PII
type EmailAddress inherits String''',
         prompt="Given the PII annotation and EmailAddress type, define a User model with an id (string), a username, and an email field annotated with @PII.",
         gold='''model User {
  id : String
  username : String
  @PII email : EmailAddress
}'''),
    dict(id="sa.025", domain="social", construct=["model"], gold_source="authored",
         in_context='''type UserId inherits String
type PostId inherits String
model UserProfile {
  id : UserId
  handle : String
}
model Post {
  id : PostId
  author : UserId
  content : String
}''',
         prompt="Given the UserProfile and Post models, define a Like model with an id (string), a user reference, a post reference, and an Instant likedAt.",
         gold='''model Like {
  id : String
  user : UserId
  post : PostId
  likedAt : Instant
}'''),
    dict(id="sa.026", domain="banking", construct=["model"], gold_source="authored",
         in_context='''type CustomerId inherits String
type AccountNumber inherits String
model Customer {
  id : CustomerId
  fullName : String
}
model Account {
  number : AccountNumber
  customer : CustomerId
}''',
         prompt="Given Customer and Account, define a CustomerSummary model that includes the customer id, full name, and a list of account numbers (an array).",
         gold='''model CustomerSummary {
  customer : CustomerId
  fullName : String
  accounts : AccountNumber[]
}'''),
    dict(id="sa.027", domain="trading", construct=["service"], gold_source="authored",
         in_context='''type TickerSymbol inherits String
type Price inherits Decimal
model Quote {
  ticker : TickerSymbol
  bid : Price
  ask : Price
}''',
         prompt="Given the Quote model, define a QuoteService with operations: getQuote (takes TickerSymbol, returns Quote) and getQuotes (takes TickerSymbol[], returns Quote[]).",
         gold='''service QuoteService {
  operation getQuote(ticker : TickerSymbol) : Quote
  operation getQuotes(tickers : TickerSymbol[]) : Quote[]
}'''),
    dict(id="sa.028", domain="generic", construct=["query"], gold_source="authored",
         in_context='''type FirstName inherits String
type LastName inherits String
model Person {
  firstName : FirstName
  lastName : LastName
}''',
         prompt="Given the Person model, write a TaxiQL find query that returns Person[] projected to give just the first name (as a String).",
         gold='''find { Person[] } as {
  firstName : FirstName
}[]'''),
    dict(id="sa.029", domain="ecommerce", construct=["model"], gold_source="authored",
         in_context='''type ProductId inherits String
type Quantity inherits Int''',
         prompt="Given the ProductId and Quantity types, define a LineItem model containing a product id, a quantity, and a unit price (decimal). Then define an Invoice model with an id (string), a list of LineItem, and a total decimal.",
         gold='''model LineItem {
  product : ProductId
  quantity : Quantity
  unitPrice : Decimal
}
model Invoice {
  id : String
  items : LineItem[]
  total : Decimal
}'''),
    dict(id="sa.030", domain="logistics", construct=["model"], gold_source="authored",
         in_context='''type TrackingNumber inherits String
type CityName inherits String''',
         prompt="Given those types, define a Route model with an array of CityName waypoints, then a RoutedShipment model carrying a tracking number and a Route.",
         gold='''model Route {
  waypoints : CityName[]
}
model RoutedShipment {
  trackingNumber : TrackingNumber
  route : Route
}'''),
]


# ============================== OPEN-ENDED (multi-block, ambiguous, novel composition) ==============================
OPEN_ENDED: list[dict] = [
    dict(id="oe.001", domain="trading", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design a small trading domain: a Trade model that captures the trade id, ticker, side (BUY/SELL), quantity, price, and execution timestamp; the corresponding TradeSide enum; and a TradeService exposing findTradeById and listTradesByTicker.",
         gold='''type TradeId inherits String
type TickerSymbol inherits String
type Quantity inherits Int
type Price inherits Decimal
enum TradeSide {
  BUY,
  SELL
}
model Trade {
  id : TradeId
  ticker : TickerSymbol
  side : TradeSide
  quantity : Quantity
  price : Price
  executedAt : Instant
}
service TradeService {
  operation findTradeById(id : TradeId) : Trade
  operation listTradesByTicker(ticker : TickerSymbol) : Trade[]
}'''),
    dict(id="oe.002", domain="banking", construct=["model", "service", "enum", "annotation"], gold_source="authored",
         prompt="Design a small money-transfer system: a Transfer model with id, from-account, to-account, amount, currency, and a TransferStatus; the TransferStatus enum (INITIATED, COMPLETED, FAILED); a Currency enum (USD, EUR); and a TransferService with initiateTransfer and getTransferStatus operations. Mark the from-account and to-account fields with a @PII annotation that takes no parameters.",
         gold='''annotation PII
type AccountNumber inherits String
type TransferId inherits String
type Money inherits Decimal
enum TransferStatus {
  INITIATED,
  COMPLETED,
  FAILED
}
enum Currency {
  USD,
  EUR
}
model Transfer {
  id : TransferId
  @PII fromAccount : AccountNumber
  @PII toAccount : AccountNumber
  amount : Money
  currency : Currency
  status : TransferStatus
}
service TransferService {
  operation initiateTransfer(fromAccount : AccountNumber, toAccount : AccountNumber, amount : Money, currency : Currency) : TransferId
  operation getTransferStatus(id : TransferId) : TransferStatus
}'''),
    dict(id="oe.003", domain="healthcare", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design a claims-processing schema: ClaimStatus enum (SUBMITTED, IN_REVIEW, APPROVED, DENIED); a Claim model with id, patient id, an amount decimal, a list of diagnosis codes, and a status; a Patient model with id and full name; and a ClaimService offering submitClaim, getClaim, listClaimsForPatient, and approveClaim operations.",
         gold='''type PatientId inherits String
type ClaimId inherits String
type DiagnosisCode inherits String
type Money inherits Decimal
enum ClaimStatus {
  SUBMITTED,
  IN_REVIEW,
  APPROVED,
  DENIED
}
model Patient {
  id : PatientId
  fullName : String
}
model Claim {
  id : ClaimId
  patient : PatientId
  amount : Money
  diagnoses : DiagnosisCode[]
  status : ClaimStatus
}
service ClaimService {
  operation submitClaim(patient : PatientId, amount : Money, diagnoses : DiagnosisCode[]) : ClaimId
  operation getClaim(id : ClaimId) : Claim
  operation listClaimsForPatient(patient : PatientId) : Claim[]
  operation approveClaim(id : ClaimId) : Claim
}'''),
    dict(id="oe.004", domain="rideshare", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Build a basic ride-share domain: Driver, Rider, Vehicle, and Ride models; a RideStatus enum (REQUESTED, MATCHED, IN_PROGRESS, COMPLETED, CANCELLED); a RideService with requestRide, matchDriver, completeRide, and cancelRide operations.",
         gold='''type DriverId inherits String
type RiderId inherits String
type VehicleId inherits String
type RideId inherits String
type Fare inherits Decimal
enum RideStatus {
  REQUESTED,
  MATCHED,
  IN_PROGRESS,
  COMPLETED,
  CANCELLED
}
model Driver {
  id : DriverId
  fullName : String
}
model Rider {
  id : RiderId
  fullName : String
}
model Vehicle {
  id : VehicleId
  driver : DriverId
  plate : String
}
model Ride {
  id : RideId
  rider : RiderId
  driver : DriverId
  status : RideStatus
  fare : Fare
}
service RideService {
  operation requestRide(rider : RiderId) : RideId
  operation matchDriver(ride : RideId) : DriverId
  operation completeRide(ride : RideId, fare : Fare) : Ride
  operation cancelRide(ride : RideId) : Boolean
}'''),
    dict(id="oe.005", domain="ecommerce", construct=["model", "service"], gold_source="authored",
         prompt="Design an order-fulfilment schema: Customer, Address, Product, LineItem, Order, and Shipment models. Order references a Customer and contains a list of LineItem. Shipment references an Order and a destination Address. Add an OrderService with createOrder, getOrder, and shipOrder operations.",
         gold='''type CustomerId inherits String
type ProductId inherits String
type OrderId inherits String
type ShipmentId inherits String
type Quantity inherits Int
type Money inherits Decimal
model Address {
  street : String
  city : String
  country : String
  postalCode : String
}
model Customer {
  id : CustomerId
  fullName : String
  email : String
}
model Product {
  id : ProductId
  name : String
  price : Money
}
model LineItem {
  product : ProductId
  quantity : Quantity
  unitPrice : Money
}
model Order {
  id : OrderId
  customer : CustomerId
  items : LineItem[]
  total : Money
}
model Shipment {
  id : ShipmentId
  order : OrderId
  destination : Address
}
service OrderService {
  operation createOrder(customer : CustomerId, items : LineItem[]) : OrderId
  operation getOrder(id : OrderId) : Order
  operation shipOrder(order : OrderId, destination : Address) : ShipmentId
}'''),
    dict(id="oe.006", domain="iot", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design a sensor telemetry schema: a Device model with id, manufacturer, and a DeviceStatus (ONLINE, OFFLINE, FAULTED); a TemperatureReading model with device, celsius, and takenAt; a HumidityReading model with device, percent, takenAt; a SensorService with operations to list devices, register a device, and fetch the latest readings of each kind for a device.",
         gold='''type DeviceId inherits String
enum DeviceStatus {
  ONLINE,
  OFFLINE,
  FAULTED
}
model Device {
  id : DeviceId
  manufacturer : String
  status : DeviceStatus
}
model TemperatureReading {
  device : DeviceId
  celsius : Decimal
  takenAt : Instant
}
model HumidityReading {
  device : DeviceId
  percent : Decimal
  takenAt : Instant
}
service SensorService {
  operation listDevices() : Device[]
  operation registerDevice(manufacturer : String) : DeviceId
  operation latestTemperature(device : DeviceId) : TemperatureReading
  operation latestHumidity(device : DeviceId) : HumidityReading
}'''),
    dict(id="oe.007", domain="content", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design a blogging platform domain: an Author, Article, Comment, and Like model; an ArticleStatus enum (DRAFT, PUBLISHED, ARCHIVED); a BlogService with createDraft, publishArticle, addComment, listComments, and likeArticle operations.",
         gold='''type AuthorId inherits String
type ArticleId inherits String
type CommentId inherits String
enum ArticleStatus {
  DRAFT,
  PUBLISHED,
  ARCHIVED
}
model Author {
  id : AuthorId
  displayName : String
}
model Article {
  id : ArticleId
  author : AuthorId
  title : String
  body : String
  status : ArticleStatus
  publishedAt : Instant?
}
model Comment {
  id : CommentId
  article : ArticleId
  author : AuthorId
  body : String
  postedAt : Instant
}
model Like {
  user : AuthorId
  article : ArticleId
  likedAt : Instant
}
service BlogService {
  operation createDraft(author : AuthorId, title : String, body : String) : ArticleId
  operation publishArticle(article : ArticleId) : Article
  operation addComment(article : ArticleId, author : AuthorId, body : String) : CommentId
  operation listComments(article : ArticleId) : Comment[]
  operation likeArticle(user : AuthorId, article : ArticleId) : Boolean
}'''),
    dict(id="oe.008", domain="trading", construct=["model", "service", "query"], gold_source="authored",
         prompt="Design a market data domain: a Quote model with ticker, bid, ask, and timestamp; a Trade model with id, ticker, side, quantity, price, executedAt; a MarketService for fetching current quotes and historical trades by ticker; and a TaxiQL query that returns all trades for analysis.",
         gold='''type TickerSymbol inherits String
type Price inherits Decimal
type Quantity inherits Int
type TradeId inherits String
enum TradeSide {
  BUY,
  SELL
}
model Quote {
  ticker : TickerSymbol
  bid : Price
  ask : Price
  takenAt : Instant
}
model Trade {
  id : TradeId
  ticker : TickerSymbol
  side : TradeSide
  quantity : Quantity
  price : Price
  executedAt : Instant
}
service MarketService {
  operation currentQuote(ticker : TickerSymbol) : Quote
  operation historicalTrades(ticker : TickerSymbol) : Trade[]
}
find { Trade[] }'''),
    dict(id="oe.009", domain="logistics", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design a parcel-shipping system: a Carrier model, a Parcel model with weight, dimensions, status; a ShipmentStatus enum (PENDING, IN_TRANSIT, DELIVERED, FAILED, RETURNED); a Route model with origin and destination addresses and an array of waypoints; and a ShippingService with bookShipment, updateStatus, and getRoute operations.",
         gold='''type CarrierId inherits String
type ParcelId inherits String
enum ShipmentStatus {
  PENDING,
  IN_TRANSIT,
  DELIVERED,
  FAILED,
  RETURNED
}
model Address {
  street : String
  city : String
  country : String
  postalCode : String
}
model Carrier {
  id : CarrierId
  name : String
}
model Parcel {
  id : ParcelId
  weightKg : Decimal
  status : ShipmentStatus
}
model Route {
  origin : Address
  destination : Address
  waypoints : Address[]
}
service ShippingService {
  operation bookShipment(carrier : CarrierId, parcel : ParcelId, route : Route) : Boolean
  operation updateStatus(parcel : ParcelId, status : ShipmentStatus) : Parcel
  operation getRoute(parcel : ParcelId) : Route
}'''),
    dict(id="oe.010", domain="social", construct=["model", "service"], gold_source="authored",
         prompt="Build a social network's friend-request and timeline schema: User, FriendRequest with statuses (PENDING, ACCEPTED, REJECTED), Post, and a TimelineService that returns a user's recent posts and a SocialService that handles sending and responding to friend requests.",
         gold='''type UserId inherits String
type RequestId inherits String
type PostId inherits String
enum FriendRequestStatus {
  PENDING,
  ACCEPTED,
  REJECTED
}
model User {
  id : UserId
  handle : String
}
model FriendRequest {
  id : RequestId
  fromUser : UserId
  toUser : UserId
  status : FriendRequestStatus
}
model Post {
  id : PostId
  author : UserId
  content : String
  postedAt : Instant
}
service TimelineService {
  operation recentPosts(user : UserId) : Post[]
}
service SocialService {
  operation sendFriendRequest(fromUser : UserId, toUser : UserId) : RequestId
  operation respondToFriendRequest(request : RequestId, accept : Boolean) : FriendRequest
}'''),
    dict(id="oe.011", domain="banking", construct=["model", "service"], gold_source="authored",
         prompt="Design a card-issuing schema with a Card, a CardHolder, an enum for CardStatus (ISSUED, ACTIVE, BLOCKED, EXPIRED), and a CardService with issueCard, blockCard, listCardsForHolder operations. Mark the card number with a @Sensitive annotation.",
         gold='''annotation Sensitive
type CardId inherits String
type CardNumber inherits String
type HolderId inherits String
enum CardStatus {
  ISSUED,
  ACTIVE,
  BLOCKED,
  EXPIRED
}
model CardHolder {
  id : HolderId
  fullName : String
}
model Card {
  id : CardId
  holder : HolderId
  @Sensitive number : CardNumber
  status : CardStatus
  expiresAt : Instant
}
service CardService {
  operation issueCard(holder : HolderId) : CardId
  operation blockCard(card : CardId) : Card
  operation listCardsForHolder(holder : HolderId) : Card[]
}'''),
    dict(id="oe.012", domain="content", construct=["model", "service", "annotation"], gold_source="authored",
         prompt="Build a video-streaming catalogue: a Video model with id, title, durationSeconds, contentRating; a Subtitle model attached to a video; an enum ContentRating (G, PG, PG13, R); a CatalogueService with searchByTitle, getVideo, and listSubtitles operations.",
         gold='''type VideoId inherits String
type DurationSeconds inherits Int
enum ContentRating {
  G,
  PG,
  PG13,
  R
}
model Subtitle {
  video : VideoId
  language : String
  body : String
}
model Video {
  id : VideoId
  title : String
  durationSeconds : DurationSeconds
  rating : ContentRating
}
service CatalogueService {
  operation searchByTitle(title : String) : Video[]
  operation getVideo(id : VideoId) : Video
  operation listSubtitles(video : VideoId) : Subtitle[]
}'''),
    dict(id="oe.013", domain="generic", construct=["model", "service"], gold_source="authored",
         prompt="Design a generic Notification system: a Notification model with id, recipient (a UserId), channel (an enum EMAIL, SMS, PUSH), body, and sentAt; a NotificationService with sendNotification, listNotificationsForUser, and markRead operations.",
         gold='''type UserId inherits String
type NotificationId inherits String
enum NotificationChannel {
  EMAIL,
  SMS,
  PUSH
}
model Notification {
  id : NotificationId
  recipient : UserId
  channel : NotificationChannel
  body : String
  sentAt : Instant
  readAt : Instant?
}
service NotificationService {
  operation sendNotification(recipient : UserId, channel : NotificationChannel, body : String) : NotificationId
  operation listNotificationsForUser(user : UserId) : Notification[]
  operation markRead(id : NotificationId) : Notification
}'''),
    dict(id="oe.014", domain="iot", construct=["model", "service"], gold_source="authored",
         prompt="Design a smart-home system: a Home, a Room, a Device that belongs to a Room, and a HomeService with operations to register a home, list rooms, list devices in a room, and toggle a device's state. Devices have an isOn boolean.",
         gold='''type HomeId inherits String
type RoomId inherits String
type DeviceId inherits String
model Home {
  id : HomeId
  name : String
}
model Room {
  id : RoomId
  home : HomeId
  name : String
}
model Device {
  id : DeviceId
  room : RoomId
  name : String
  isOn : Boolean
}
service HomeService {
  operation registerHome(name : String) : HomeId
  operation listRooms(home : HomeId) : Room[]
  operation listDevices(room : RoomId) : Device[]
  operation toggleDevice(device : DeviceId) : Device
}'''),
    dict(id="oe.015", domain="ecommerce", construct=["model", "service", "query"], gold_source="authored",
         prompt="Design an inventory + order schema: a Product, a Warehouse, a StockLevel that ties a product to a warehouse with a quantity, an InventoryService with operations to update stock and check stock by product, plus a TaxiQL find query returning all StockLevel.",
         gold='''type ProductId inherits String
type WarehouseId inherits String
type Quantity inherits Int
model Product {
  id : ProductId
  name : String
  price : Decimal
}
model Warehouse {
  id : WarehouseId
  location : String
}
model StockLevel {
  product : ProductId
  warehouse : WarehouseId
  quantity : Quantity
}
service InventoryService {
  operation updateStock(product : ProductId, warehouse : WarehouseId, delta : Quantity) : StockLevel
  operation checkStock(product : ProductId) : StockLevel[]
}
find { StockLevel[] }'''),
    dict(id="oe.016", domain="trading", construct=["model", "service"], gold_source="authored",
         prompt="Design a portfolio domain: an Investor, a Position (ticker, quantity, average price), a Portfolio (with positions), and a PortfolioService with operations to fetch a portfolio by investor, add a trade, and compute total notional.",
         gold='''type InvestorId inherits String
type TickerSymbol inherits String
type Quantity inherits Int
type Price inherits Decimal
type PortfolioId inherits String
model Investor {
  id : InvestorId
  fullName : String
}
model Position {
  ticker : TickerSymbol
  quantity : Quantity
  averagePrice : Price
}
model Portfolio {
  id : PortfolioId
  investor : InvestorId
  positions : Position[]
}
service PortfolioService {
  operation getPortfolio(investor : InvestorId) : Portfolio
  operation addTrade(portfolio : PortfolioId, ticker : TickerSymbol, quantity : Quantity, price : Price) : Position
  operation totalNotional(portfolio : PortfolioId) : Decimal
}'''),
    dict(id="oe.017", domain="generic", construct=["model", "service", "annotation"], gold_source="authored",
         prompt="Build an audit-log schema: an Actor with id and role (enum ADMIN, USER, SYSTEM), an AuditLogEntry with actor, an action string, an instant when, and an optional payload string. Annotate the payload with @Sensitive (a parameterless annotation). Add a service with operations to record an entry and list entries by actor.",
         gold='''annotation Sensitive
type ActorId inherits String
type LogEntryId inherits String
enum ActorRole {
  ADMIN,
  USER,
  SYSTEM
}
model Actor {
  id : ActorId
  role : ActorRole
}
model AuditLogEntry {
  id : LogEntryId
  actor : ActorId
  action : String
  occurredAt : Instant
  @Sensitive payload : String?
}
service AuditService {
  operation recordEntry(actor : ActorId, action : String, payload : String?) : LogEntryId
  operation listEntriesForActor(actor : ActorId) : AuditLogEntry[]
}'''),
    dict(id="oe.018", domain="healthcare", construct=["model", "service"], gold_source="authored",
         prompt="Design a prescription system: a Drug catalogue entry, a Prescription that links a patient and a drug with a dosage and duration, a Pharmacy, and a PrescriptionService with createPrescription, fillPrescription, and listPrescriptionsForPatient operations.",
         gold='''type DrugId inherits String
type PatientId inherits String
type PrescriptionId inherits String
type PharmacyId inherits String
type DurationDays inherits Int
model Drug {
  id : DrugId
  name : String
  strengthMg : Decimal
}
model Pharmacy {
  id : PharmacyId
  name : String
  address : String
}
model Prescription {
  id : PrescriptionId
  patient : PatientId
  drug : DrugId
  dosageMg : Decimal
  durationDays : DurationDays
  filledAt : Instant?
}
service PrescriptionService {
  operation createPrescription(patient : PatientId, drug : DrugId, dosageMg : Decimal, durationDays : DurationDays) : PrescriptionId
  operation fillPrescription(prescription : PrescriptionId, pharmacy : PharmacyId) : Prescription
  operation listPrescriptionsForPatient(patient : PatientId) : Prescription[]
}'''),
    dict(id="oe.019", domain="ecommerce", construct=["model", "service", "annotation", "query"], gold_source="authored",
         prompt="Design a returns-and-refunds flow: a Return model that links to an Order, has a ReturnReason (enum DEFECTIVE, WRONG_ITEM, CHANGED_MIND, OTHER), a refunded amount, and a status (REQUESTED, APPROVED, REFUNDED, REJECTED). Annotate refundedAmount with @Sensitive (annotation has no fields). Add a ReturnsService with createReturn, approveReturn, and rejectReturn operations, plus a TaxiQL find returning all approved returns.",
         gold='''annotation Sensitive
type OrderId inherits String
type ReturnId inherits String
type Money inherits Decimal
enum ReturnReason {
  DEFECTIVE,
  WRONG_ITEM,
  CHANGED_MIND,
  OTHER
}
enum ReturnStatus {
  REQUESTED,
  APPROVED,
  REFUNDED,
  REJECTED
}
model Return {
  id : ReturnId
  order : OrderId
  reason : ReturnReason
  status : ReturnStatus
  @Sensitive refundedAmount : Money?
}
service ReturnsService {
  operation createReturn(order : OrderId, reason : ReturnReason) : ReturnId
  operation approveReturn(id : ReturnId, refundAmount : Money) : Return
  operation rejectReturn(id : ReturnId) : Return
}
find { Return[]( ReturnStatus == ReturnStatus.APPROVED ) }'''),
    dict(id="oe.020", domain="banking", construct=["model", "service"], gold_source="authored",
         prompt="Design a loan-origination flow: an Applicant, a LoanApplication with status (RECEIVED, UNDERWRITING, APPROVED, DENIED), an UnderwritingDecision attached to an application, and a LoanService with submitApplication, recordUnderwritingDecision, and getApplicationStatus operations.",
         gold='''type ApplicantId inherits String
type ApplicationId inherits String
type DecisionId inherits String
type Money inherits Decimal
enum ApplicationStatus {
  RECEIVED,
  UNDERWRITING,
  APPROVED,
  DENIED
}
model Applicant {
  id : ApplicantId
  fullName : String
  annualIncome : Money
}
model LoanApplication {
  id : ApplicationId
  applicant : ApplicantId
  amountRequested : Money
  status : ApplicationStatus
}
model UnderwritingDecision {
  id : DecisionId
  application : ApplicationId
  approved : Boolean
  rationale : String
  decidedAt : Instant
}
service LoanService {
  operation submitApplication(applicant : ApplicantId, amountRequested : Money) : ApplicationId
  operation recordUnderwritingDecision(application : ApplicationId, approved : Boolean, rationale : String) : DecisionId
  operation getApplicationStatus(application : ApplicationId) : ApplicationStatus
}'''),
    dict(id="oe.021", domain="rideshare", construct=["model", "service", "query"], gold_source="authored",
         prompt="Design a driver-payouts module: a Payout model carrying a driver id, period start and end timestamps, gross and net amounts; a PayoutService with operations to compute a driver's payout for a period and list past payouts; a TaxiQL find query returning all Payout.",
         gold='''type DriverId inherits String
type PayoutId inherits String
type Money inherits Decimal
model Payout {
  id : PayoutId
  driver : DriverId
  periodStart : Instant
  periodEnd : Instant
  gross : Money
  net : Money
}
service PayoutService {
  operation computePayout(driver : DriverId, periodStart : Instant, periodEnd : Instant) : Payout
  operation listPayoutsForDriver(driver : DriverId) : Payout[]
}
find { Payout[] }'''),
    dict(id="oe.022", domain="generic", construct=["model", "service", "annotation"], gold_source="authored",
         prompt="Build a feature-flag system: a Flag with name and a default boolean enabled state; an Override that scopes a flag for a specific user; an Audience enum (BETA, INTERNAL, PUBLIC); a FlagService with createFlag, setOverride, isFlagEnabledForUser operations.",
         gold='''type FlagId inherits String
type UserId inherits String
enum Audience {
  BETA,
  INTERNAL,
  PUBLIC
}
model Flag {
  id : FlagId
  name : String
  enabled : Boolean
  audience : Audience
}
model Override {
  flag : FlagId
  user : UserId
  enabled : Boolean
}
service FlagService {
  operation createFlag(name : String, audience : Audience) : FlagId
  operation setOverride(flag : FlagId, user : UserId, enabled : Boolean) : Override
  operation isFlagEnabledForUser(flag : FlagId, user : UserId) : Boolean
}'''),
    dict(id="oe.023", domain="logistics", construct=["model", "service"], gold_source="authored",
         prompt="Design a warehouse-receiving flow: an InboundShipment from a supplier, ReceivingNote that records what was actually received and any discrepancies, and a ReceivingService with recordInbound, recordReceipt, and listDiscrepancies operations.",
         gold='''type SupplierId inherits String
type InboundId inherits String
type ReceiptId inherits String
type ProductId inherits String
type Quantity inherits Int
model InboundShipment {
  id : InboundId
  supplier : SupplierId
  expectedAt : Instant
  expected : ProductId[]
  expectedQuantities : Quantity[]
}
model ReceivingNote {
  id : ReceiptId
  inbound : InboundId
  received : ProductId[]
  receivedQuantities : Quantity[]
  discrepancies : String?
  receivedAt : Instant
}
service ReceivingService {
  operation recordInbound(supplier : SupplierId, expected : ProductId[], expectedQuantities : Quantity[]) : InboundId
  operation recordReceipt(inbound : InboundId, received : ProductId[], receivedQuantities : Quantity[], discrepancies : String?) : ReceiptId
  operation listDiscrepancies(inbound : InboundId) : ReceivingNote[]
}'''),
    dict(id="oe.024", domain="trading", construct=["model", "service", "enum", "query"], gold_source="authored",
         prompt="Build an order-book domain: an OrderBookEntry with side (BUY/SELL), price, and remaining quantity; a Trade that matches two OrderBookEntries; a MatchEngineService with placeOrder, cancelOrder, and getOrderBook operations; and a TaxiQL find returning all OrderBookEntry.",
         gold='''type EntryId inherits String
type TickerSymbol inherits String
type Price inherits Decimal
type Quantity inherits Int
type TradeId inherits String
enum TradeSide {
  BUY,
  SELL
}
model OrderBookEntry {
  id : EntryId
  ticker : TickerSymbol
  side : TradeSide
  price : Price
  remainingQuantity : Quantity
}
model Trade {
  id : TradeId
  ticker : TickerSymbol
  buyEntry : EntryId
  sellEntry : EntryId
  price : Price
  quantity : Quantity
  executedAt : Instant
}
service MatchEngineService {
  operation placeOrder(ticker : TickerSymbol, side : TradeSide, price : Price, quantity : Quantity) : EntryId
  operation cancelOrder(entry : EntryId) : Boolean
  operation getOrderBook(ticker : TickerSymbol) : OrderBookEntry[]
}
find { OrderBookEntry[] }'''),
    dict(id="oe.025", domain="content", construct=["model", "service"], gold_source="authored",
         prompt="Build a podcasting domain: Podcast with id, title, hosts (an array of host ids); Episode with id, podcast id, title, audioUrl string, durationSeconds int, releasedAt Instant; PodcastService with createPodcast, addEpisode, listEpisodes operations.",
         gold='''type PodcastId inherits String
type HostId inherits String
type EpisodeId inherits String
type DurationSeconds inherits Int
model Podcast {
  id : PodcastId
  title : String
  hosts : HostId[]
}
model Episode {
  id : EpisodeId
  podcast : PodcastId
  title : String
  audioUrl : String
  durationSeconds : DurationSeconds
  releasedAt : Instant
}
service PodcastService {
  operation createPodcast(title : String, hosts : HostId[]) : PodcastId
  operation addEpisode(podcast : PodcastId, title : String, audioUrl : String, durationSeconds : DurationSeconds) : EpisodeId
  operation listEpisodes(podcast : PodcastId) : Episode[]
}'''),
    dict(id="oe.026", domain="iot", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design an industrial-IoT alerting schema: a Sensor with id and SensorType enum (TEMPERATURE, VIBRATION, PRESSURE); an AlertRule that triggers when a sensor's value crosses a threshold (with comparator GT, LT, EQ); an Alert that records when a rule fired; an AlertService with rule CRUD and listing recent alerts.",
         gold='''type SensorId inherits String
type RuleId inherits String
type AlertId inherits String
enum SensorType {
  TEMPERATURE,
  VIBRATION,
  PRESSURE
}
enum Comparator {
  GT,
  LT,
  EQ
}
model Sensor {
  id : SensorId
  sensorType : SensorType
}
model AlertRule {
  id : RuleId
  sensor : SensorId
  comparator : Comparator
  threshold : Decimal
}
model Alert {
  id : AlertId
  rule : RuleId
  observedValue : Decimal
  firedAt : Instant
}
service AlertService {
  operation createRule(sensor : SensorId, comparator : Comparator, threshold : Decimal) : RuleId
  operation deleteRule(rule : RuleId) : Boolean
  operation listRecentAlerts() : Alert[]
}'''),
    dict(id="oe.027", domain="banking", construct=["model", "service", "annotation", "query"], gold_source="authored",
         prompt="Design a KYC schema: a Customer with @PII (annotation, no params) annotations on identifying fields; a KycCheck model recording the type of check (enum DOCUMENT, BIOMETRIC, AML), result (PASS/FAIL/PENDING), and timestamp; a KycService with submitCheck and getKycStatusForCustomer operations; plus a find query for failed KYC checks.",
         gold='''annotation PII
type CustomerId inherits String
type CheckId inherits String
enum KycCheckType {
  DOCUMENT,
  BIOMETRIC,
  AML
}
enum KycResult {
  PASS,
  FAIL,
  PENDING
}
model Customer {
  id : CustomerId
  @PII fullName : String
  @PII dateOfBirth : Date
  @PII passportNumber : String?
}
model KycCheck {
  id : CheckId
  customer : CustomerId
  checkType : KycCheckType
  result : KycResult
  performedAt : Instant
}
service KycService {
  operation submitCheck(customer : CustomerId, checkType : KycCheckType) : CheckId
  operation getKycStatusForCustomer(customer : CustomerId) : KycCheck[]
}
find { KycCheck[]( KycResult == KycResult.FAIL ) }'''),
    dict(id="oe.028", domain="generic", construct=["model", "service", "enum"], gold_source="authored",
         prompt="Design a workflow / task-management schema: a Workflow with name, an array of Task; a Task with id, title, an Assignee (a UserId), a TaskStatus (TODO, IN_PROGRESS, DONE, BLOCKED); a WorkflowService with createWorkflow, advanceTask, and listOpenTasks operations.",
         gold='''type WorkflowId inherits String
type TaskId inherits String
type UserId inherits String
enum TaskStatus {
  TODO,
  IN_PROGRESS,
  DONE,
  BLOCKED
}
model Task {
  id : TaskId
  title : String
  assignee : UserId
  status : TaskStatus
}
model Workflow {
  id : WorkflowId
  name : String
  tasks : Task[]
}
service WorkflowService {
  operation createWorkflow(name : String) : WorkflowId
  operation advanceTask(task : TaskId, status : TaskStatus) : Task
  operation listOpenTasks(workflow : WorkflowId) : Task[]
}'''),
    dict(id="oe.029", domain="healthcare", construct=["model", "service", "query"], gold_source="authored",
         prompt="Design a clinical trial schema: a Trial, a Cohort that belongs to a Trial, an Enrolment that links a Patient to a Cohort, a TrialEvent recording adverse events; a TrialService with operations to create a trial, enrol a patient, record an event, and a find query returning all TrialEvent.",
         gold='''type TrialId inherits String
type CohortId inherits String
type EnrolmentId inherits String
type EventId inherits String
type PatientId inherits String
model Trial {
  id : TrialId
  name : String
  startedAt : Instant
}
model Cohort {
  id : CohortId
  trial : TrialId
  name : String
}
model Enrolment {
  id : EnrolmentId
  patient : PatientId
  cohort : CohortId
  enrolledAt : Instant
}
model TrialEvent {
  id : EventId
  patient : PatientId
  trial : TrialId
  description : String
  occurredAt : Instant
  severity : String
}
service TrialService {
  operation createTrial(name : String) : TrialId
  operation enrolPatient(patient : PatientId, cohort : CohortId) : EnrolmentId
  operation recordEvent(trial : TrialId, patient : PatientId, description : String, severity : String) : EventId
}
find { TrialEvent[] }'''),
    dict(id="oe.030", domain="generic", construct=["model", "service", "annotation"], gold_source="authored",
         prompt="Build a billing-and-subscriptions schema: a Subscriber, a Plan with id, name, price (decimal), and billing cadence (enum MONTHLY, QUARTERLY, ANNUAL); a Subscription linking subscriber and plan with a startedAt and renewedAt; an Invoice attached to a subscription with amount and dueAt; mark price and amount with @Sensitive (annotation has no fields). Service exposes subscribe, cancelSubscription, generateInvoice operations.",
         gold='''annotation Sensitive
type SubscriberId inherits String
type PlanId inherits String
type SubscriptionId inherits String
type InvoiceId inherits String
type Money inherits Decimal
enum BillingCadence {
  MONTHLY,
  QUARTERLY,
  ANNUAL
}
model Subscriber {
  id : SubscriberId
  email : String
}
model Plan {
  id : PlanId
  name : String
  @Sensitive price : Money
  cadence : BillingCadence
}
model Subscription {
  id : SubscriptionId
  subscriber : SubscriberId
  plan : PlanId
  startedAt : Instant
  renewedAt : Instant?
}
model Invoice {
  id : InvoiceId
  subscription : SubscriptionId
  @Sensitive amount : Money
  dueAt : Instant
}
service BillingService {
  operation subscribe(subscriber : SubscriberId, plan : PlanId) : SubscriptionId
  operation cancelSubscription(subscription : SubscriptionId) : Boolean
  operation generateInvoice(subscription : SubscriptionId) : InvoiceId
}'''),
]


# -------------------------------------------------------------------- main
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_entries: list[dict] = []
    rejected: list[tuple[str, str]] = []

    with TaxiValidator() as v:
        for difficulty, entries in [("easy", EASY), ("schema_aware", SCHEMA_AWARE), ("open_ended", OPEN_ENDED)]:
            for spec in entries:
                gold = spec["gold"]
                in_ctx = spec.get("in_context")
                # validate
                try:
                    if in_ctx:
                        res = v.validate_multi([("schema.taxi", in_ctx), ("gold.taxi", gold)])
                    else:
                        res = v.validate(gold, source_name=f"{spec['id']}.taxi")
                except Exception as e:
                    rejected.append((spec["id"], f"validator threw: {e}"))
                    continue
                if not res.is_valid:
                    msgs = "; ".join(e.detailMessage for e in res.errors[:3])
                    rejected.append((spec["id"], f"{res.error_count} errors: {msgs[:160]}"))
                    continue
                all_entries.append({
                    "id": spec["id"],
                    "difficulty": difficulty,
                    "domain": spec["domain"],
                    "construct_tags": spec["construct"],
                    "gold_source": spec["gold_source"],
                    "prompt": spec["prompt"],
                    "in_context_schema": in_ctx,
                    "gold_taxi": gold,
                })

    # write output
    with OUT_PATH.open("w") as f:
        for r in all_entries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    from collections import Counter
    print(f"=== benchmark build report ===")
    print(f"total accepted:  {len(all_entries)}")
    print(f"total rejected:  {len(rejected)}")
    if rejected:
        print(f"\n=== REJECTED (gold validation failed) ===")
        for rid, msg in rejected:
            print(f"  {rid}: {msg}")
    print(f"\nby difficulty: {dict(Counter(r['difficulty'] for r in all_entries))}")
    print(f"by domain:     {dict(Counter(r['domain'] for r in all_entries))}")
    print(f"\noutput: {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
