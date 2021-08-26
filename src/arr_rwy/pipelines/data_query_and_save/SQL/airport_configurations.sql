select
  airport_id,
  datis_time,
  start_time as config_start_time,
  'D_' || replace(departure_runways, ', ', '_') || '_A_' || replace(arrival_runways, ', ', '_') as airport_configuration_name
from datis_parser_message
where airport_id = :airport
  and datis_time between (timestamp :start_time - interval '48 hours') and (timestamp :end_time + interval '24 hours')
  and start_time between (timestamp :start_time - interval '24 hours') and :end_time
