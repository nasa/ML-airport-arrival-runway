select
  gufi,
  "timestamp",
  arrival_fix_source_data,
  filed_flight
from matm_flight
where "timestamp" between :start_time and :end_time
  and arrival_aerodrome_icao_name = :airport
  and arrival_fix_source_data is not null
