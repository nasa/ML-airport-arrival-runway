select
  gufi,
  departure_aerodrome_icao_name,
  departure_runway_actual_time,
  arrival_runway_actual_time,
  aircraft_engine_class,
  aircraft_type,
  carrier
from matm_flight_summary
where (arrival_runway_actual_time between :start_time and :end_time)
  and (arrival_aerodrome_icao_name = :airport)
