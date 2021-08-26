with base as (
	select
	  gufi,
	  "timestamp",
	  arrival_runway_assigned,
	  lag(arrival_runway_assigned) over wnd as arrival_runway_assigned_lagged
	from matm_flight
	where "timestamp" between :start_time and :end_time
	  and arrival_aerodrome_icao_name = :airport
	  and arrival_runway_assigned is not null
	window wnd as (partition by gufi order by timestamp_fuser_processed, sequence_id rows between unbounded preceding and unbounded following)
)
select
  gufi,
  "timestamp",
  arrival_runway_assigned
from base
where (arrival_runway_assigned <> arrival_runway_assigned_lagged)
   or (arrival_runway_assigned_lagged is null and arrival_runway_assigned is not null)
   or (arrival_runway_assigned_lagged is not null and arrival_runway_assigned is null)
